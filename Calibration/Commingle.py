import time
from Functions_calibration import *
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime


def Calculate_MSE(path, display):

    thermal = Thermal()
    thermal.Reset()

    V0 = VS  # solid input speed -> constant speed for calibration (240 mm/min, 360 mm/min, 480 mm/min)

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 600  # changed in the physical world code

    stripe_count = 0
    global_count = 0

    mses = []
    global_counts = []
    high_reals = []
    high_simus = []

    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])

    # 构建存储文件夹
    current_folder = '.'

    folder_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    save_path = os.path.join(current_folder, folder_name)
    os.makedirs(save_path, exist_ok=True)

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y

        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0
            step_count = 0

            # calculate length period in one strip
            heat_length = COLD_STARTER[stripe_count] - HEAT_STARTER[stripe_count]
            wait_length = HEAT_STARTER[stripe_count + 1] - COLD_STARTER[stripe_count]

            # balance the FPS - simulation fps (CELL_SIZE_X+TIME_SLEEP) and physical fps
            heat_fps = Balance_FPS(CELL_SIZE_X, heat_length)
            wait_fps = Balance_FPS(TIME_SLEEP, wait_length)

            # heater is working
            for step in range(CELL_SIZE_X):

                # calculate the global step number
                global_count += 1

                # Execute One Step
                thermal.Step(P, V0, heat_loc, True)

                # calculate the thermal distribution
                simulation_data = thermal.current_T
                simulation_previous_data = thermal.previous_T
                concat_T = thermal.previous_T * (1 - thermal.Actuator) + simulation_data
                shadow = thermal.Actuator

                # calculate MSE - heating the layer
                if heat_fps != [] and step_count == heat_fps[0]:
                    file = str(path) + "\\" + str(csv_files[0])
                    csv_temp = np.loadtxt(file, delimiter=',')
                    physical_data = spatial_calibration(csv_temp, layer_num=layer)
                    physical_data = physical_data + 273.15   # offset the temperature

                    # 仅仅关注当前 actuator 之内的内容，之外内容没有关注
                    mse, error_rate_matrix = mean_squared_rate(simulation_data, physical_data, shadow)

                    # pop the used elements
                    csv_files.pop(0)
                    heat_fps.pop(0)

                    mses.append(mse)
                    global_counts.append(global_count)

                    # additional assignments
                    high_reals.append(np.max(physical_data))
                    high_simus.append(np.max(simulation_data))

                # Update Location
                heat_loc[0] += 1
                step_count += 1

                # # additional assignments
                if display:

                    # if global_count % 10 == 0:
                    if global_count == 1420 or global_count == 1450 or global_count == 2850:

                        # np.save('physical-' + str(global_count) + "_1trans_0.1trans_30body", physical_data)
                        # np.save('simulation-' + str(global_count) + "_1trans_0.1trans_30body", simulation_data)
                        # np.save('simulation_previous-' + str(global_count) + "_1trans_0.1trans_30body", simulation_previous_data)

                        # name_ = str(global_count) + str(physical_data)
                        name_ = 'physical_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path,  name_)  # 创建每个数组的保存路径
                        np.save(file_path, physical_data)  # 保存数组

                        # record data into files
                        name_ = 'simulation_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
                        np.save(file_path, simulation_data)  # 保存数组

                        name_ = 'simulation_previous_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
                        np.save(file_path, simulation_previous_data)  # 保存数组

                    # print('global_count', global_count)

                #     np.save('physical', physical_data)
                #     np.save('simulation', simulation_data)

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):
                thermal.Step(P, V0, heat_loc, False)
                global_count += 1

                # print(np.average(thermal.current_T - thermal.previous_T) * thermal.Actuator)
                # print(np.average((thermal.body - thermal.previous_T)))

                # calculate the thermal distribution
                simulation_data = thermal.current_T * thermal.Actuator
                concat_T = thermal.previous_T * (1 - thermal.Actuator) + simulation_data
                simulation_previous_data = thermal.previous_T
                shadow = thermal.Actuator

                # calculate MSE during waiting period
                if wait_fps != [] and step_count - CELL_SIZE_X == wait_fps[0]:
                    file = str(path) + "\\" + str(csv_files[0])
                    csv_temp = np.loadtxt(file, delimiter=',')
                    physical_data = spatial_calibration(csv_temp, layer_num=layer)
                    physical_data = physical_data + 273.15

                    # 仅仅关注当前actuator之内的内容，之外内容没有关注
                    mse, error_rate_matrix = mean_squared_rate(simulation_data, physical_data, shadow)

                    # pop the used elements
                    csv_files.pop(0)
                    wait_fps.pop(0)

                    # print('mse', mse)
                    mses.append(mse)
                    global_counts.append(global_count)

                    high_reals.append(np.max(physical_data))
                    high_simus.append(np.max(simulation_data))

                step_count += 1

                if display:
                    if global_count == 1420 or global_count == 1450 or global_count == 2850:

                        # np.save('physical-' + str(global_count), physical_data)
                        # np.save('simulation-' + str(global_count) + "_1trans_0.1trans_30body", simulation_data)
                        # np.save('simulation_previous' + str(global_count) + "_1trans_0.1trans_30body", simulation_previous_data)

                        # # 新建立一个文件夹，存储np文件
                        # # record data into files
                        name_ = 'physical_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
                        np.save(file_path, physical_data)  # 保存数组

                        name_ = 'simulation_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
                        np.save(file_path, simulation_data)  # 保存数组

                        name_ = 'simulation_previous_data_' + str(global_count) + '.npy'
                        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
                        np.save(file_path, simulation_previous_data)  # 保存数组

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            stripe_count += 1
            print('stripe_count', stripe_count)

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return mses, global_counts, high_reals, high_simus, save_path


# 均方误差比率
def mean_squared_rate(x_pred, x_real, shadow):
    epsilon = 1e-8
    error_rate_matrix = np.abs(x_pred - x_real) / (x_real + epsilon) * 100
    error_rate_matrix = error_rate_matrix * shadow
    mean_error_rate = np.mean(error_rate_matrix)

    return mean_error_rate, error_rate_matrix


def Display(matrix):

    plt.imshow(matrix)
    plt.show()


def spatial_calibration(image, layer_num):
    """
    应用透视变换，将斜视角图像转换为俯视图，并转换为灰度图像。

    参数：
    image: 输入图像
    src_points: 图像中待变换的四个角点
    dst_size: 目标图像的大小 (width, height)

    返回：
    透视变换后的灰度图像
    """

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
    transformed_image = np.rot90(transformed_image, k=-1)

    return transformed_image


def Balance_FPS(input_frame_count, target_frame_count):
    """
    Compute the optimal mapping from input frames to target frames with minimal error.

    Args:
        input_frame_count (int): Number of input frames.
        target_frame_count (int): Number of target frames.

    Returns:
        list: List of input frame indices corresponding to the target frames.
    """

    step = (input_frame_count - 1) / (target_frame_count)

    # Generate the indices
    corresponding_frames = [round(i * step) for i in range(target_frame_count)]

    return corresponding_frames


if __name__ == "__main__":
    # find the path
    path = "D:\\test_data\\csv"
    time1 = time.time()
    display = True

    mses, global_counts, high_reals, high_simus, save_path = Calculate_MSE(path, display)
    time2 = time.time()
    print('time', time1-time2)

    # record mses
    file_path = os.path.join(save_path, 'mses.txt')
    with open(file_path, 'w') as f:
        f.write(','.join(map(str, mses)))

    # record global_counts
    file_path = os.path.join(save_path, 'global_counts.txt')
    with open(file_path, 'w') as f:
        f.write(','.join(map(str, global_counts)))

    py_file_path = "../Parameter.py"
    txt_file_path = os.path.join(save_path, 'Parameter.txt')

    # 读取 .py 文件内容
    with open(py_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 将内容写入 .txt 文件
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(content)

    # with open('physical.txt', 'w') as f:
    #     f.write(','.join(map(str, high_reals)))
    #
    # with open('simulation.txt', 'w') as f:
    #     f.write(','.join(map(str, high_simus)))
