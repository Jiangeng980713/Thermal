import time
from Functions_calibration import *
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from Parameter import *
import pickle

# 每次运行之前注意 resume—checkpoint
# 每次运行之前注意 NOTE 需要改进内容

def Calculate_MSE(path, display, resume_checkpoint=None, new_NOTE_dir=None, resume_NOTE_dir=None):

    V0 = VS
    P = 600

    # 新模型自己的存储文件夹
    current_folder = '.'
    save_path = os.path.join(current_folder, str(new_NOTE_dir))
    os.makedirs(save_path, exist_ok=True)

    # 新模型自己的checkpoint文件夹
    checkpoint_path = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # ==================================================================================================================
    #                                               是否从checkpoint恢复
    # ==================================================================================================================

    if resume_checkpoint is None:
        checkpoint = None

    else:
        old_checkpoint_path = os.path.join(current_folder, str(resume_NOTE_dir), "checkpoints")
        checkpoint_name = os.path.join(old_checkpoint_path, resume_checkpoint)
        checkpoint = load_checkpoint(checkpoint_name)

        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    # ==================================================================================================================
    #                                               checkpoint 读取
    # ==================================================================================================================

    if checkpoint is None:

        thermal = Thermal()
        thermal.Reset_start()

        heat_loc = [INIT_X, INIT_Y, 0]

        stripe_count = 0
        global_count = 0

        mses = []
        global_counts = []
        high_reals = []
        high_simus = []

        csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])

        start_layer = 0
        start_stripe = 0

    else:

        thermal = checkpoint["thermal"]
        heat_loc = checkpoint["heat_loc"]
        stripe_count = checkpoint["stripe_count"]
        global_count = checkpoint["global_count"]
        mses = checkpoint["mses"]
        global_counts = checkpoint["global_counts"]
        high_reals = checkpoint["high_reals"]
        high_simus = checkpoint["high_simus"]
        csv_files = checkpoint["csv_files"]
        start_layer = checkpoint["layer"]
        start_stripe = checkpoint["stripe"]

        if not 0 <= start_stripe <= STRIPE_NUM:
            raise ValueError(f"Unexpected start_stripe: {start_stripe}")

        if not 0 <= start_layer < LAYER_HEIGHT:
            raise ValueError(f"Unexpected start_layer: {start_layer}")

        if start_stripe == STRIPE_NUM:

            # ★ 改进：如果后面还有下一层
            if start_layer + 1 < LAYER_HEIGHT:

                heat_loc[2] += 1
                thermal.reset()

                heat_loc[0] = INIT_X
                heat_loc[1] = INIT_Y

                start_layer += 1
                start_stripe = 0

            # ★ 改进：如果已经是最后一层最后一个stripe
            else:
                start_layer = LAYER_HEIGHT
                start_stripe = 0

        print("Resume from:" + checkpoint_name)
        print("layer =", start_layer)
        print("stripe =", start_stripe)
        print("global_count =", global_count)

    # ==================================================================================================================
    #                                                   开始计算
    # ==================================================================================================================

    for layer in range(start_layer, LAYER_HEIGHT):

        # 当checkpoint有东西（开始读一些东西）而且 layer = start_layer（从最开始层开始读）-》上面已经给 heat_loc 一个旧坐标，不需要归零
        if checkpoint is not None and layer == start_layer:
            pass
        else:   # 否则开始归零
            heat_loc[0], heat_loc[1] = INIT_X, INIT_Y

        if layer == start_layer:
            stripe_begin = start_stripe
        else:
            stripe_begin = 0

        for stripe in range(stripe_begin, STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0
            step_count = 0

            # calculate length period in one strip
            heat_length = COLD_STARTER[stripe_count] - HEAT_STARTER[stripe_count]
            wait_length = HEAT_STARTER[stripe_count + 1] - COLD_STARTER[stripe_count]

            # balance the FPS - simulation fps (CELL_SIZE_X+TIME_SLEEP) and physical fps
            heat_fps = Balance_FPS(CELL_SIZE_X, heat_length)
            wait_fps = Balance_FPS(TIME_SLEEP, wait_length)

########################################################################################################################

            # check whether last stripe boundary offset    # TODO：右侧不对的问题需要进行解决
            if stripe // (STRIPE_NUM - 1) == 1:

                right_bound = RIGHT_BOUND
                # print(stripe // (STRIPE_NUM - 1), right_bound)

            else:
                right_bound = False
                # print(stripe // (STRIPE_NUM - 1), right_bound)

########################################################################################################################
            # heater is working
            for step in range(CELL_SIZE_X):

                # calculate the global step number
                global_count += 1

                thermal.Step(P, V0, heat_loc, True, right_bound=right_bound)

                # if global_count > 100:
                #     Display(thermal.current_T)

                # if 100 < global_count < 200 and global_count % 10 == 0:
                #     np.save(f"thermal_T_{global_count}.npy", thermal.current_T)

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
                thermal.Step(P, V0, heat_loc, False, right_bound=right_bound)
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

            ############################################ 每一个 stripe 都进行一个保存 #####################################
            next_stripe = stripe + 1

            # ★ 改进：去掉 if next_stripe < STRIPE_NUM
            # 每一条stripe完成后都保存，包括S7
            checkpoint_data = {
                "thermal": thermal,
                "layer": layer,
                # ★ 改进：这里仍然保存“下一条需要计算的stripe
                "stripe": next_stripe,
                "global_count": global_count,
                "stripe_count": stripe_count,
                "heat_loc": heat_loc.copy(),
                "csv_files": csv_files.copy(),
                "mses": mses.copy(),
                "global_counts": global_counts.copy(),
                "high_reals": high_reals.copy(),
                "high_simus": high_simus.copy()
            }

            checkpoint_name = os.path.join(checkpoint_path, f"checkpoint_{stripe_count:03d}_L{layer + 1}_S{stripe + 1}.pkl")
            save_checkpoint(checkpoint_data, checkpoint_name)
            print('stripe', stripe)
            print('工作了5')
            ############################################ 每一个 stripe 都进行一个保存 #####################################

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return mses, global_counts, high_reals, high_simus, save_path


def save_checkpoint(data, checkpoint_name):
    with open(checkpoint_name, "wb") as f:
        pickle.dump(data, f)

    print("Checkpoint saved:", checkpoint_name)


def load_checkpoint(checkpoint_name):
    if os.path.exists(checkpoint_name):
        with open(checkpoint_name, "rb") as f:
            return pickle.load(f)

    return None


# 均方误差比率
def mean_squared_rate(x_pred, x_real, shadow):
    epsilon = 1e-8
    error_rate_matrix = np.abs(x_pred - x_real) / (x_real + epsilon) * 100
    error_rate_matrix = error_rate_matrix * shadow
    mean_error_rate = np.mean(error_rate_matrix)

    return mean_error_rate, error_rate_matrix


def Display(matrix):

    plt.imshow(matrix,cmap='jet')
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
    display = False

    # 老的文件夹放到哪里
    resume_NOTE_dir = "FULL_RIGHT_BOUND_DOUBLE_LAYER_NEW_MSE"

    # checkpoint的名字
    resume_checkpoint = None

    # 新的文件夹位置
    new_NOTE_dir = "VALIDATION"

    mses, global_counts, high_reals, high_simus, save_path = Calculate_MSE(path, display, resume_checkpoint, new_NOTE_dir, resume_NOTE_dir)
    time2 = time.time()
    print('time', time2-time1)

    # record mses
    file_path = os.path.join(save_path, 'mses.txt')
    with open(file_path, 'w') as f:
        f.write(','.join(map(str, mses)))

    # record global_counts
    file_path = os.path.join(save_path, 'global_counts.txt')
    with open(file_path, 'w') as f:
        f.write(','.join(map(str, global_counts)))

    # record parameter information
    py_file_path = "../Parameter.py"
    txt_file_path = os.path.join(save_path, 'Parameter.txt')

    # 读取 .py 文件内容
    with open(py_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 将内容写入 .txt 文件
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(content)
