from functions import *
import cv2
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


def Display(matrix):
    plt.imshow(matrix)
    plt.show()


def apply_perspective_transform(image):
    """
    应用透视变换，将斜视角图像转换为俯视图，并转换为灰度图像。

    参数：
    image: 输入图像
    src_points: 图像中待变换的四个角点
    dst_size: 目标图像的大小 (width, height)

    返回：
    透视变换后的灰度图像
    """

    # 原始图像中待变换的四个点（例如，某个矩形区域的四个角）
    src_points = np.float32([
        [330, 84],  # 左上角
        [420, 91],  # 右上角
        [404, 159],  # 右下角
        [312, 152]  # 左下角
    ])

    # 目标图像大小 (宽度, 高度)，也就是俯视图的分辨率，改大小就是实现自适应池化的过程
    dst_size = (1600, 1600)

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

    return transformed_image


# # load single data
# def load_csv(folder_path):
#
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
#
#     data_frames = []
#
#     for file in csv_files:
#         df = pd.read_csv(file)
#         data_frames.append(df)
#
#     # 将所有数据合并为一个DataFrame
#     all_data = pd.concat(data_frames, ignore_index=True)
#
#     return all_data

# load the figures and apply angle transformation
def load_all_figures(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    data_list = []

    # load the figs and transform them
    for file in csv_files:

        # load data
        data = np.loadtxt(file, delimiter=',')

        # transform angles
        data = apply_perspective_transform(data)

        # fuse the data
        data_list.append(data)

    # 将所有数据合并为一个大的numpy数组
    all_data = np.vstack(data_list)

    return all_data


# test for the angle transformation
if __name__ == "__main__":
    folder_path = 'xxxxxxx'
    sample = apply_perspective_transform(folder_path[0])
    Display(sample)
