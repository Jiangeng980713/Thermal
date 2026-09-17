import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


def spatial_calibration(image, layer_num):

    match layer_num:

        case 0:
            src_points = np.float32([
                [99, 105],   # 左上角
                [333, 107],  # 右上角
                [332, 155],  # 右下角
                [95, 153]    # 左下角
            ])
        case 1:
            src_points = np.float32([
                [100, 102],  # 左上角
                [334, 104],  # 右上角
                [328, 151],  # 右下角
                [93, 148]    # 左下角
            ])
        case 2:
            src_points = np.float32([
                [105, 97],   # 左上角
                [328, 99],   # 右上角
                [330, 149],  # 右下角
                [92, 146]    # 左下角
            ])
        case 3:
            src_points = np.float32([
                [105, 93],   # 左上角
                [328, 95],   # 右上角
                [327, 148],  # 右下角
                [91, 144]    # 左下角
            ])
        case 4:
            src_points = np.float32([
                [108, 89],   # 左上角
                [328, 92],   # 右上角
                [329, 143],  # 右下角
                [91, 141]    # 左下角
            ])

    # 目标图像大小 (宽度, 高度)，也就是俯视图的分辨率，改大小就是实现自适应池化的过程
    dst_size = (175, 46)

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

    # rotate the figure and align the direction
    # transformed_image = np.rot90(transformed_image, k=-1)

    return transformed_image

# Layer 0  # 0608 - 0663 - 0672
# Layer 1  # 1126 - 1180 - 1194
# Layer 2  # 1641 - 1696 - 1704
# Layer 3  # 2146 - 2194 - 2211
# Layer 4  # 2632 - 2686 - 2704


layer_num = 4
num = "2632"
file = "D:\\test_data\\csv\\Rec-0217_" + num + "_163244.csv"
csv_temp = np.loadtxt(file, delimiter=',')

plt.imshow(csv_temp)
plt.show()

image = spatial_calibration(csv_temp, layer_num=layer_num)

plt.imshow(image)
plt.show()