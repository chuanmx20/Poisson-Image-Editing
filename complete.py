import cv2
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import maxflow

def prepare_images(input_path, mask_path, patch_path):
    input_img = cv2.imread(input_path)
    mask_img = cv2.imread(mask_path, 0)
    patch_img = cv2.imread(patch_path)
    return input_img, mask_img, patch_img

def dilate_mask(mask_img, k=3):
    kernel = np.ones((k, k), np.uint8)
    dilated_mask = cv2.dilate(mask_img, kernel, iterations=3)
    return dilated_mask

def mask_image(input_img, mask_img):
    # 将mask_img转换为bool类型的mask
    mask_img = mask_img.astype(bool)
    masked_img = input_img.copy()
    masked_img[mask_img] = 0
    return masked_img

def get_patch(patch_shape, patch_img, best_match):
    # 根据best_match位置，从patch_img中取出对应的patch
    patch = patch_img[best_match[0]:best_match[0] + patch_shape[0], best_match[1]:best_match[1] + patch_shape[1]]
    return patch

def find_best_match(masked_region, patch_img):
    # 实现查找最佳匹配位置的逻辑
    # 将input_img被mask的区域在patch_img上平移，找到L2距离最小的位置
    best_position = (0, 0)
    min_error = float('inf')
    # naive方法
    # for i in range(patch_img.shape[0] - masked_region.shape[0]):
    #     for j in range(patch_img.shape[1] - masked_region.shape[1]):
    #         sub_patch = patch_img[i:i + masked_region.shape[0], j:j + masked_region.shape[1]]
    #         sub_patch[mask_img] = 0
    #         error = np.sum((sub_patch - masked_region) ** 2)
    #         if error < min_error:
    #             min_error = error
    #             best_position = (j, i)
    # 使用fft加速上述过程
    # 以masked_region为卷积核，patch_img为被卷积图像，计算L2距离，padding方式为valid，即卷积核不超过被卷积图像
    error = fftconvolve(patch_img, np.flip(np.flip(masked_region, 0), 1), mode='valid') ** 2
    # 找到L2距离最小的位置
    best_position = np.unravel_index(np.argmin(error), error.shape)
    return best_position

# 在待补全的图像和补丁图像（用于填补空缺的图像）之间找到最佳的融合边界
def find_boundary_graphcut(incomplete_img, patch_img):
    # 创建一个图
    g = maxflow.Graph[float]()

    # 为每个像素添加一个节点
    nodeids = g.add_grid_nodes(incomplete_img.shape[:2])

    # 创建边权重的计算函数
    def calculate_weight(s, t, A, B):
        return np.sum(np.abs(A[s] - B[s])) + np.sum(np.abs(A[t] - B[t]))

    # 为每对相邻像素添加边
    for i in range(incomplete_img.shape[0]):
        for j in range(incomplete_img.shape[1] - 1): # Horizontal edges
            s = (i, j)
            t = (i, j + 1)
            weight = calculate_weight(s, t, incomplete_img, patch_img)
            g.add_edge(nodeids[s], nodeids[t], weight, weight)

        for j in range(incomplete_img.shape[1]): # Vertical edges
            s = (i, j)
            t = (i + 1, j) if i < incomplete_img.shape[0] - 1 else (0, j) # wrap around for the last row
            weight = calculate_weight(s, t, incomplete_img, patch_img)
            g.add_edge(nodeids[s], nodeids[t], weight, weight)

    # 数据项边权重使用像素点在补丁和原图中的差值
    for i in range(incomplete_img.shape[0]):
        for j in range(incomplete_img.shape[1]):
            s = (i, j)
            weight = np.sum(np.abs(incomplete_img[s] - patch_img[s]))
            g.add_tedge(nodeids[s], weight, weight)

    # 计算最大流
    g.maxflow()

    # 获取分割后的图像区域
    sgm = g.get_grid_segments(nodeids)
    print(sgm)
    plt.imshow(sgm)    
    plt.show()
    # 返回计算出的最佳融合边界
    return sgm

def complete_image(input_path, mask_path, patch_path):
    # 准备图像
    input_img, mask_img, patch_img = prepare_images(input_path, mask_path, patch_path)
    # 膨胀mask
    dilated_mask = dilate_mask(mask_img)
    # 被mask的区域
    masked_region = mask_image(input_img, dilated_mask)
    incomplete_region = input_img - masked_region
    # 找到最佳匹配位置
    best_match = find_best_match(masked_region, patch_img)
    print(best_match)
    
    patch = get_patch(masked_region.shape, patch_img, best_match)
    print(masked_region.shape, patch_img.shape)
    patch = mask_image(patch, dilated_mask)
    
    plt.imshow(patch)
    plt.show()
    
    plt.imshow(incomplete_region)
    plt.show()
    find_boundary_graphcut(incomplete_region, patch)

completed_img = complete_image('data/completion/input1.jpg', 'data/completion/input1_mask.jpg', 'data/completion/input1_patch.jpg')