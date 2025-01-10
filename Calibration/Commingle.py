import time

from Functions_calibration import *
import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


def Calculate_MSE(path):
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
                simulation_data = thermal.current_T * thermal.Actuator
                concat_T = thermal.previous_T * (1 - thermal.Actuator) + simulation_data

                # calculate MSE
                if heat_fps != [] and step_count == heat_fps[0]:
                    file = str(path) + "\\" + str(csv_files[0])
                    csv_temp = np.loadtxt(file, delimiter=',')
                    physical_data = spatial_calibration(csv_temp)
                    mse = mean_squared_rate(simulation_data, physical_data)

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

                # additional assignments
                if global_count % 175 == 0:
                    Display(physical_data)
                    Display(simulation_data)
                    np.save('physical', physical_data)
                    np.save('simulation', simulation_data)

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):
                thermal.Step(P, V0, heat_loc, False)
                global_count += 1

                # calculate the thermal distribution
                simulation_data = thermal.current_T * thermal.Actuator
                concat_T = thermal.previous_T * (1 - thermal.Actuator) + simulation_data

                # calculate MSE during waiting period
                if wait_fps != [] and step_count - CELL_SIZE_X == wait_fps[0]:
                    file = str(path) + "\\" + str(csv_files[0])
                    csv_temp = np.loadtxt(file, delimiter=',')
                    physical_data = spatial_calibration(csv_temp)
                    mse = mean_squared_rate(simulation_data, physical_data)

                    # pop the used elements
                    csv_files.pop(0)
                    wait_fps.pop(0)

                    # print('mse', mse)
                    mses.append(mse)
                    global_counts.append(global_count)

                    high_reals.append(np.max(physical_data))
                    high_simus.append(np.max(simulation_data))

                step_count += 1

                if global_count % 175 == 0:
                    Display(physical_data)
                    Display(simulation_data)
                    np.save('physical', physical_data)
                    np.save('simulation', simulation_data)

                # print('step_count', step_count)
                # print('wait_fps', len(wait_fps))

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            stripe_count += 1
            print('stripe_count', stripe_count)

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return mses, global_counts, high_reals, high_simus


# 均方误差比率
def mean_squared_rate(x_pred, x_real):
    epsilon = 1e-8
    error_rate_matrix = np.abs(x_pred - x_real) / (np.abs(x_real) + epsilon) * 100
    # Display(np.abs(x_pred - x_real))
    # Display(x_pred)
    # Display(x_real)
    mean_error_rate = np.mean(error_rate_matrix)
    return mean_error_rate

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
        case 1:
            src_points = np.float32([
                [99, 105],  # 左上角
                [337, 107],  # 右上角
                [332, 158],  # 右下角
                [95, 156]  # 左下角
            ])
        case 2:
            src_points = np.float32([
                [99, 105],  # 左上角
                [337, 107],  # 右上角
                [332, 158],  # 右下角
                [95, 156]  # 左下角
            ])
        case 3:
            src_points = np.float32([
                [99, 105],  # 左上角
                [337, 107],  # 右上角
                [332, 158],  # 右下角
                [95, 156]  # 左下角
            ])
        case 4:
            src_points = np.float32([
                [99, 105],  # 左上角
                [337, 107],  # 右上角
                [332, 158],  # 右下角
                [95, 156]  # 左下角
            ])
        case 5:
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
    mses, global_counts, high_reals, high_simus = Calculate_MSE(path)
    time2 = time.time()
    print('time', time1-time2)

    with open('mses.txt', 'w') as f:
        f.write(','.join(map(str, mses)))

    with open('global_counts.txt', 'w') as f:
        f.write(','.join(map(str, global_counts)))

    with open('physical.txt', 'w') as f:
        f.write(','.join(map(str, high_reals)))

    with open('simulation.txt', 'w') as f:
        f.write(','.join(map(str, high_simus)))
