import cv2
import itertools
import numpy as np
import scipy
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import maxflow
from pyamg.gallery import poisson
import scipy
from scipy.sparse import csr_matrix
from skimage.color import rgb2gray
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d

def prepare_images(input_path, mask_path, patch_path):
    input_img = cv2.imread(input_path)
    mask_img = cv2.imread(mask_path, 0)
    mask_img = mask_img.astype(bool)
    mask_img = np.invert(mask_img)
    patch_img = cv2.imread(patch_path)
    return input_img, mask_img, patch_img

def dilate_mask(mask_img, k=3):
    kernel = np.ones((k, k), np.uint8)
    mask_img = mask_img.astype(np.uint8) * 255
    dilated_mask = cv2.dilate(mask_img, kernel, iterations=7)
    return np.clip(dilated_mask, 0, 1)

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
    print(masked_region.shape, patch_img.shape)
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
    # 对三个通道分别计算L2距离，然后求和
    # L2(f,g)=∑(f−g)^2
    # L2(f,g)=∑f^2+∑g^2−2∑fg
    # 计算每个通道的平方
    patch_squared = (patch_img / 255.0) ** 2
    masked_squared = (masked_region / 255.0) ** 2
    l2_distance_map = np.zeros((patch_img.shape[0] - masked_region.shape[0] + 1,
                            patch_img.shape[1] - masked_region.shape[1] + 1))
    
    # 对于每个颜色通道
    for c in range(3):
        # 计算卷积 f^2 和 g^2
        l2_distance_map += fftconvolve(patch_squared[:, :, c], np.ones_like(masked_squared[:, :, c]), mode='valid')
        
        # 计算 -2fg 部分，并加到 L2 距离矩阵中
        l2_distance_map -= 2 * fftconvolve(patch_img[:, :, c] / 255.0, masked_squared[:, :, c], mode='valid')

    # 将 g^2 的和加到每个位置
    l2_distance_map += np.sum(masked_squared)
        
    best_position = np.unravel_index(np.argmin(l2_distance_map), l2_distance_map.shape)
    return best_position

# 在待补全的图像和补丁图像（用于填补空缺的图像）之间找到最佳的融合边界
def find_boundary_graphcut(incomplete_img, patch_img, mask):
    mask = mask.astype(np.uint8) * 255
    # 创建一个图
    g = maxflow.Graph[float]()

    # 为每个像素添加一个节点
    nodeids = g.add_grid_nodes(incomplete_img.shape[:2])

    # 设置节点之间的边权重
    for i in range(incomplete_img.shape[0]):
        for j in range(incomplete_img.shape[1]):
            if j+1 < incomplete_img.shape[1]:  # 邻接像素在水平方向
                weight = np.sum(np.abs(incomplete_img[i,j] - patch_img[i,j])) + np.sum(np.abs(incomplete_img[i,j+1] - patch_img[i,j+1]))
                g.add_edge(nodeids[i,j], nodeids[i,j+1], weight, weight)
            if i+1 < incomplete_img.shape[0]:  # 邻接像素在垂直方向
                weight = np.sum(np.abs(incomplete_img[i,j] - patch_img[i,j])) + np.sum(np.abs(incomplete_img[i+1,j] - patch_img[i+1,j]))
                g.add_edge(nodeids[i,j], nodeids[i+1,j], weight, weight)

    # 设置源点和汇点的边权重
    g.add_grid_tedges(nodeids,255 - mask, mask)

    # 计算最大流
    g.maxflow()

    # 获取分割
    sgm = g.get_grid_segments(nodeids)
    return sgm

def is_boundatry(mask, x, y):
    """
    用于判断像素(x, y)是否是mask的边界像素，即(x, y)处于mask内部，但是(x, y)的邻居中有像素不在mask内部
    """
    if not mask[x, y]:
        return False
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        i, j = x + di, y + dj
        if i >= 0 and i < mask.shape[0] and j >= 0 and j < mask.shape[1]:
            if not mask[i, j]:
                return True
    return False

def poisson_blend(patch, sgm, dest):
    """
    使用泊松融合算法将补丁与不完整的图像融合。
    Args:
        patch (np.array): 用于补全的patch，已经经过mask处理
        sgm (_type_): 使用GraphCut得到的分割结果
        dest (_type_): 待补全的图像，已经经过mask处理
    """
    # 将mask转换为bool类型
    sgm = sgm.astype(bool)
    patch = patch.astype(np.float64)
    dest = dest.astype(np.float64)

    # 对于Poisson Blending，首先需要构建系数矩阵A和b
    # 首先给mask部分的像素编号，编号的顺序是从左到右，从上到下
    n_pixels = np.sum(sgm)
    pixel_dict = {} # 像素坐标到编号的映射
    pixel_index = 0
    for i, j in itertools.product(range(sgm.shape[0]), range(sgm.shape[1])):
        if sgm[i, j]:
            pixel_dict[(i, j)] = pixel_index
            pixel_index += 1
    
    # 针对每个channel，构建系数矩阵A和b
    for color in range(3):
        # 计算补丁图像散度
        patch_laplace = scipy.ndimage.convolve(patch[:, :, color], np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), mode='constant')
        # 构建系数矩阵A和指导向量b
        # A 就是一个对角矩阵，对角线上的元素为4，对应像素的相邻元素为-1
        A = lil_matrix((n_pixels, n_pixels), dtype=np.float64)
        b = np.zeros((n_pixels,), dtype=np.float64)
        for (i, j), pid in pixel_dict.items():
            if is_boundatry(sgm, i, j):
                # 融合边界的像素，直接将dest中的像素值作为b的值
                A[pid, pid] = 1
                b[pid] = dest[i, j, color]
                
            else:
                # 内部像素，使用拉普拉斯算子计算散度
                A[pid, pid] = 4
                if (i-1, j) in pixel_dict:
                    A[pid, pixel_dict[(i-1, j)]] = -1
                if (i+1, j) in pixel_dict:
                    A[pid, pixel_dict[(i+1, j)]] = -1
                if (i, j-1) in pixel_dict:
                    A[pid, pixel_dict[(i, j-1)]] = -1
                if (i, j+1) in pixel_dict:
                    A[pid, pixel_dict[(i, j+1)]] = -1
                b[pid] = patch_laplace[i, j]

        # 解线性方程组
        x = spsolve(A.tocsr(), b)
        # 将解的结果写回dest
        for (i, j), pid in pixel_dict.items():
            dest[i, j, color] = np.clip(x[pid], 0, 255)
    return dest

def complete_image(input_path, mask_path, patch_path, i):
    # 准备图像
    input_img, mask_img, patch_img = prepare_images(input_path, mask_path, patch_path)
    stage1 = input_img.copy()
    # 膨胀mask
    dilated_mask = dilate_mask(mask_img)
    # 被mask的区域
    incomplete_img = mask_image(input_img, dilated_mask)
    masked_region = input_img - incomplete_img
    # 找到最佳匹配位置
    best_match = find_best_match(masked_region, patch_img)
    # 裁切出patch
    patch = get_patch(masked_region.shape, patch_img, best_match)
    # 使用GraphCut找到融合边界
    sgm = find_boundary_graphcut(input_img, patch, dilated_mask)
    stage2 = stage1.copy()
    stage2[sgm] = 0
    stage3 = patch.copy()
    stage3[sgm==0] = 0
    # 使用Poisson Blending融合两部分图片
    blended = poisson_blend(patch, sgm, input_img).astype(np.uint8)
    return stage1, stage2, stage2+stage3, blended

plt.figure()
n = 4
test = False
if test:
    root_path = 'test/'
    result_path = 'test/result/'
else:
    root_path = 'data/completion/'
    result_path = 'data/completion/output/'
for i in range(1, n+1):
    input_img, incomplete, naive, blended = complete_image(f'{root_path}input{i}.jpg', f'{root_path}input{i}_mask.jpg', f'{root_path}input{i}_patch.jpg', i)
    cv2.imwrite(f'{result_path}input{i}_incomplete.jpg', incomplete)
    cv2.imwrite(f'{result_path}input{i}_naive.jpg', naive)
    cv2.imwrite(f'{result_path}input{i}_poisson.jpg', blended)
    plt.subplot(n, 4, i*4-3)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Input {i}')
    plt.axis('off')
    plt.subplot(n, 4, i*4-2)
    plt.imshow(cv2.cvtColor(incomplete, cv2.COLOR_BGR2RGB))
    plt.title(f'Incomplete {i}')
    plt.axis('off')
    plt.subplot(n, 4, i*4-1)
    plt.imshow(cv2.cvtColor(naive, cv2.COLOR_BGR2RGB))
    plt.title(f'Naive {i}')
    plt.axis('off')
    plt.subplot(n, 4, i*4)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title(f'Poisson Blended {i}')
    plt.axis('off')
plt.show()