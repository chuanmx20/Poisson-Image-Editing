<center>
  <font size=8>
    <b>图像补全大作业报告</b>
  </font>
  <br>
  <font size=3>钏茗喜 &nbsp 2020011035</font>
</center>



## 一、实现步骤

我实现了如下算法：

- 使用OpenCV的dilate方法扩充mask，然后将被扣掉区域使用FFT卷积的方法加速计算L2距离
- 使用maxflow库计算最优边界
- 使用scipy.sparse解泊松融合线性方程组

### 1. 最近匹配

**1.1 扩宽mask**

首先，需要将mask膨胀，得到扩充k个像素的mask区域，这里我使用的是OpenCV提供的接口

```python
cv2.dilate(mask_img, kernel, iterations=7)
```

**1.2 计算L2距离，找到距离最小的位置**

这部分的代码如下：

```python
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
```

对于最简单的实现，即逐步平移计算距离的方法，效率较低。这里我才用FFT卷积的方法，使用valid模式（即卷积核不会离开被卷积区域，没有padding）得到两个图在每个区域上的乘积，然后根据下面这个式子计算出每个位置的L2距离：
$$
L2(f, g) = \sum (f-g)^2
= \sum f^2 + \sum g^2 - 2\sum fg
$$
最后再使用argmin的方法找到距离最小的位置即可

### 2. 计算融合边界

此步骤对上一步的mask的边界进一步分割，使用GraphCut算法确定最优边界。我使用了maxflow库来计算最大流，具体步骤如下：

**2.1 构建图**

图的初始化使用`maxflow.Graph`，每一个像素点位置都是图中的一个节点，对于节点之间的边，使用如下公式：
$$
M(s,t,A,B)=||A(s)-B(s)||+||A(t)-B(t)||
$$


而对于源点到汇点之间的边，使用mask图像来计算，属于同一区域就是255，否则就是0（U8，255就是无穷大，0就是无穷小）。

**2.2 最大流分割图**

在图构建好以后，就可以计算出图中的最大流，得到最优融合边界了，我使用的是maxflow提供的`maxflow`和`get_grid_segments`方法

该部分真题代码如下

```python
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
```

### 3. 泊松融合

**3.1 算法概述**

泊松融合算法大致可以解释为两部分：

- 内部平滑
- 边缘一致

内部平滑指的是，在补全区域内部，补丁（前景）和原图像（背景）梯度相同；边界一致指的是两部分在边界上的像素值相同。

该算法的真正要处理的就是下面这个函数
$$
\min_{f} \iint_\Omega |\triangledown f-v|^2 with f|_{\partial\Omega} = f^*|_{\partial\Omega}
$$


其中$\Omega$表示融合的区域，$f$表示融合的结果，$f^*$表示的是背景图像，$v$表示前景图像。这个公式简单来说就是，在背景图像边缘不变（结果的边缘=背景边缘）的前提下，让融合结果在融合区域内部的梯度与补丁图像的梯度最小。

图像梯度的计算有很多不同的方法（如Laplace、Sobel、Scharr算子等），不同方法的侧重点也不太一样。Laplace算子能很好的体现出不同物体之间的边界，对于图像平滑化处理任务，使用Laplace算子作为图像梯度较为合适。

最终求解上述最小二乘问题，可以简化为求解Ax=b的过程，x的每一个元素都是待补全区域的一个像素点，b对应位置的每一个元素都是求解该点像素值的一个指导（即最小二乘的结果）。

对于区域内部的像素点，使用梯度计算来平滑化这一区域，即
$$
\triangledown f_{x,y} = 4f_{x, y} - f_{x-1,y} - f_{x+1,y} - f_{x,y-1} - f_{x,y+1}
$$
所以这一像素点在A矩阵中对应的一行就是对角元为4，A同一行其他四个邻居对应的列的元素为-1，假设该像素的编号是i，也就是$A_ix = b[i]$，这里按照前面说的让补丁内部梯度最小，所以b[i]就取补丁在该点的梯度。

对于边界上的点，使用背景图的像素颜色，即
$$
f_{x,y} = f^*_{x,y}
$$
这一像素点在A矩阵中对应的一行就是对角元为1，其他区域都是0。

按照上述方法构建方程组，再使用scipy.sparse方法求解方程组即可得到区域内的像素值了

**3.2 算法实现**

```python
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
```

## 二、运行结果

### 1. 初始图像

|                            背景图                            |                             补丁                             |                           待补全图                           |                           直接拼接                           |                           泊松融合                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![input1](/Users/chuanmx/Downloads/第一次大作业/data/completion/input1.jpg) | ![input1_patch](/Users/chuanmx/Downloads/第一次大作业/data/completion/input1_patch.jpg) | ![input1_incomplete](/Users/chuanmx/Downloads/第一次大作业/output/input1_incomplete.jpg) | ![input1_naive](/Users/chuanmx/Downloads/第一次大作业/output/input1_naive.jpg) | ![input1_poisson](/Users/chuanmx/Downloads/第一次大作业/output/input1_poisson.jpg) |
| ![input2](/Users/chuanmx/Downloads/第一次大作业/data/completion/input2.jpg) | ![input2_patch](/Users/chuanmx/Downloads/第一次大作业/data/completion/input2_patch.jpg) | ![input2_incomplete](/Users/chuanmx/Downloads/第一次大作业/output/input2_incomplete.jpg) | ![input2_naive](/Users/chuanmx/Downloads/第一次大作业/output/input2_naive.jpg) | ![input2_poisson](/Users/chuanmx/Downloads/第一次大作业/output/input2_poisson.jpg) |
| ![input3](/Users/chuanmx/Downloads/第一次大作业/data/completion/input3.jpg) | ![input3_patch](/Users/chuanmx/Downloads/第一次大作业/data/completion/input3_patch.jpg) | ![input3_incomplete](/Users/chuanmx/Downloads/第一次大作业/output/input3_incomplete.jpg) | ![input3_naive](/Users/chuanmx/Downloads/第一次大作业/output/input3_naive.jpg) | ![input3_poisson](/Users/chuanmx/Downloads/第一次大作业/output/input3_poisson.jpg) |
| ![input4](/Users/chuanmx/Downloads/第一次大作业/data/completion/input4.jpg) | ![input4_patch](/Users/chuanmx/Downloads/第一次大作业/data/completion/input4_patch.jpg) | ![input4_incomplete](/Users/chuanmx/Downloads/第一次大作业/output/input4_incomplete.jpg) | ![input4_naive](/Users/chuanmx/Downloads/第一次大作业/output/input4_naive.jpg) | ![input4_poisson](/Users/chuanmx/Downloads/第一次大作业/output/input4_poisson.jpg) |

在这些图片上的融合表现都相较于直接融合更加平滑

### 2. 其他图像

| 背景图                                                       |                             补丁                             |                           待补全图                           |                           直接拼接                           |                           泊松融合                           |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![input1](/Users/chuanmx/Downloads/第一次大作业/test/input1.jpg) | ![input1_patch](/Users/chuanmx/Downloads/第一次大作业/test/input1_patch.jpg) | ![input1_incomplete](/Users/chuanmx/Downloads/第一次大作业/test/result/input1_incomplete.jpg) | ![input1_naive](/Users/chuanmx/Downloads/第一次大作业/test/result/input1_naive.jpg) | ![input1_poisson](/Users/chuanmx/Downloads/第一次大作业/test/result/input1_poisson.jpg) |
| ![input2](/Users/chuanmx/Downloads/第一次大作业/test/input2.jpg) | ![input2_patch](/Users/chuanmx/Downloads/第一次大作业/test/input2_patch.jpg) | ![input2_incomplete](/Users/chuanmx/Downloads/第一次大作业/test/result/input2_incomplete.jpg) | ![input2_naive](/Users/chuanmx/Downloads/第一次大作业/test/result/input2_naive.jpg) | ![input2_poisson](/Users/chuanmx/Downloads/第一次大作业/test/result/input2_poisson.jpg) |
| ![input3](/Users/chuanmx/Downloads/第一次大作业/test/input3.jpg) | ![input3_patch](/Users/chuanmx/Downloads/第一次大作业/test/input3_patch.jpg) | ![input3_incomplete](/Users/chuanmx/Downloads/第一次大作业/test/result/input3_incomplete.jpg) | ![input3_naive](/Users/chuanmx/Downloads/第一次大作业/test/result/input3_naive.jpg) | ![input3_naive](/Users/chuanmx/Downloads/第一次大作业/test/result/input3_naive.jpg) |
| ![input4](/Users/chuanmx/Downloads/第一次大作业/test/input4.jpg) | ![input4_patch](/Users/chuanmx/Downloads/第一次大作业/test/input4_patch.jpg) | ![input4_incomplete](/Users/chuanmx/Downloads/第一次大作业/test/result/input4_incomplete.jpg) | ![input4_naive](/Users/chuanmx/Downloads/第一次大作业/test/result/input4_naive.jpg) | ![input4_poisson](/Users/chuanmx/Downloads/第一次大作业/test/result/input4_poisson.jpg) |

找到的图像有些扣的不是很干净，或者扣的太多了，但是还是可以看出更加平滑



## 三、其他

### 1. 目录简述

```
.
├── data	# 原来的数据
│   ├── completion 
│   │   └── output # 存放补全结果，input{i}_incomplete为扣掉mask的原图，input{i}_naive为直接粘贴的结果，input{i}_poisson为泊松融合结果
│   └── quadtree
├── complete.py	# 图像补全的代码，使用python complete.py即可运行
├── report.pdf	# 实验报告
└── test	# 自己找的测试数据，格式和原来一样
    └── result	# 测试数据的运行结果，格式同上output
    

```

### 2. 运行简述

代码位于complete.py，直接运行即可。如果要运行测试数据， 把206行处的`test`改为`True`即可