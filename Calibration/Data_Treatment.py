import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


def Display(matrix):
    plt.imshow(matrix)
    plt.show()


def spatial_calibration(image):
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
        [99, 105],  # 左上角
        [337, 107],  # 右上角
        [332, 158],  # 右下角
        [95, 156]  # 左下角
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

    return transformed_image

# def temporal_calibration():
#
#     # 设置文件夹路径和阈值
#     folder_path = "D:\\test\\calibration\\csv"  # 替换为实际文件夹路径
#     threshold = 900
#
#     # 初始化变量
#     segments = []
#     start_idx = None
#     idx = 0
#
#     # 遍历文件夹中的 CSV 文件，按文件名顺序读取
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith(".csv"):  # 只处理 CSV 文件
#             file_path = os.path.join(folder_path, file_name)
#             print(file_path)
#
#             # 读取 CSV 文件并转换为二维矩阵
#             matrix = pd.read_csv(file_path, header=None).values
#
#             # 判断是否超过阈值
#             if np.max(matrix) > threshold:
#                 if start_idx is None:
#                     start_idx = idx  # 标记开始编号
#             else:
#                 if start_idx is not None:
#                     segments.append((start_idx, idx - 1))  # 记录开始和结束编号
#                     start_idx = None
#
#             idx += 1  # 增加文件编号
#
#     # 如果最后一段没有结束
#     if start_idx is not None:
#         segments.append((start_idx, idx - 1))  # 补充最后的段
#
#     # 输出结果
#     for start, end in segments:
#         print(f"Start Index: {start}, End Index: {end}")


# load the figures and transform to array
def calibration(folder_path):

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    data_list = []

    # space calibration
    for file in csv_files:
        print('simulation', file)

        # load data
        data = np.loadtxt(file, delimiter=',')

        # transform angles
        data = spatial_calibration(data)  # 46, 175

        # fuse the data
        data_list.append(data)

    # 将所有数据合并为一个大的numpy数组
    all_data = np.vstack(data_list)
    np.save("simulation_data", all_data)


# test in single figure the angle transformation
if __name__ == "__main__":

    mode = 'single'

    if mode == 'single':
        data = np.loadtxt("D:\\test\\calibration\\csv\\Rec-0217_0160_194403.csv", delimiter=",")
        sample = spatial_calibration(data)
        # Display(data)
        Display(sample)
