from functions import *
import cv2
import numpy as np


def Display(matrix):
    plt.imshow(matrix)
    plt.show()


def apply_perspective_transform(image, src_points, dst_size):
    """
    应用透视变换，将斜视角图像转换为俯视图，并转换为灰度图像。

    参数：
    image: 输入图像
    src_points: 图像中待变换的四个角点
    dst_size: 目标图像的大小 (width, height)

    返回：
    透视变换后的灰度图像
    """
    # 定义目标位置的四个点（俯视图的四个角点）
    dst_points = np.float32([
        [0, 0],  # 左上角
        [dst_size[0], 0],  # 右上角
        [dst_size[0], dst_size[1]],  # 右下角
        [0, dst_size[1]]  # 左下角
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, M, dst_size)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

    return gray_image


# 示例使用
image = cv2.imread('Figure_1.png')  # 读取图像

Display(image)

# 原始图像中待变换的四个点（例如，某个矩形区域的四个角）
src_points = np.float32([
    [330, 84],  # 左上角
    [420, 91],  # 右上角
    [404, 159],  # 右下角
    [312, 152]  # 左下角
])

# 目标图像大小 (宽度, 高度)，也就是俯视图的分辨率，改大小就是实现自适应池化的过程
dst_size = (1600, 1600)
time1 = time.time()

# 进行透视变换并转换为灰度图
result_image = apply_perspective_transform(image, src_points, dst_size)
time2 = time.time()

# 在线显示灰度图像
cv2.imshow('Transformed Gray Image', result_image)
# print(time2 - time1)

# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
