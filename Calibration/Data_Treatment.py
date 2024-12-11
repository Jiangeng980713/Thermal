from functions import *
import cv2
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import sys


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
        [540, 294],  # 左上角
        [675, 295],  # 右上角
        [671, 318],  # 右下角
        [533, 320]  # 左下角
    ])

    # 目标图像大小 (宽度, 高度)，也就是俯视图的分辨率，改大小就是实现自适应池化的过程
    dst_size = (150, 46)

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


def load_and_save_figures_to_csv(folder_path):

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 遍历每个文件进行处理
    for file in csv_files:

        # 加载数据
        data = np.loadtxt(file, delimiter=',')

        # 对数据进行透视变换
        data_transformed = apply_perspective_transform(data)

        # 获取原文件名和路径
        base_name = os.path.basename(file)  # 提取文件名
        new_name = f"transformed_{base_name}"  # 创建新文件名
        save_path = os.path.join(folder_path, new_name)  # 保存路径

        # 保存数据到 CSV 文件
        np.savetxt(save_path, data_transformed, delimiter=',', fmt='%.6f')

    print(f"所有变换后的文件已保存到: {folder_path}")


# load the figures and transform to array
def calibration(folder_path):

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    data_list = []

    # load the figs and transform them
    for file in csv_files:

        # load data
        data = np.loadtxt(file, delimiter=',')

        # transform angles
        data = apply_perspective_transform(data)  # 46, 150

        # fuse the data
        data_list.append(data)

    # 将所有数据合并为一个大的numpy数组
    all_data = np.vstack(data_list)

    return all_data


# test in single figure the angle transformation  coffee-1_2919_112557.csv 温度很高 - 2700
if __name__ == "__main__":

    mode = 'single'

    if mode == 'single':     # 2203 - 2312
        data = np.loadtxt("D:\\小范围实验\\新建文件夹\\coffee-1_2312_112557.csv", delimiter=",")
        sample = apply_perspective_transform(data)
        Display(sample)
    else:
        data = calibration("D:\\小范围实验\\新建文件夹")
        print(data.shape)
