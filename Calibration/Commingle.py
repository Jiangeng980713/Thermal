from Functions_calibration import *
from Data_Treatment import *


"remained work: 1. 加入时间戳  2. 构建MSE误差"

def Commingle(input_vector, input_figures):

    thermal = Thermal()

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 500

    thermal.Reset()
    count = 0
    mse_count = []

    # load and treat input figures
    real_figures = load_all_figures(input_figures)

    assert len(input_vector) == LAYER_HEIGHT * STRIPE_NUM, " V num do not match stripe num "

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y
        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0

            for step in range(CELL_SIZE_X):

                # calculate the global step number
                global_count = layer * STRIPE_NUM * CELL_SIZE_X + stripe * CELL_SIZE_X + step

                # Execute One Step
                thermal.Step(P, input_vector[count], heat_loc)

                # calculate the thermal distribution
                current_T = thermal.current_T
                real_T = real_T * thermal.Actuator

                if step - (step//70)*90 == 0:
                     print('step', step, step//10)
                     print('stripe, step', stripe, step)
                     thermal.Display(thermal.current_T)

                # Update Location
                heat_loc[0] += 1

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            count += 1

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return mse_count


def nearest_neighbor_match_varying_duration(fps_19, fps_30, duration_19, duration_30):

    """
    以较短的 19 FPS 或 30 FPS 视频持续时间为基准，进行最近邻匹配。

    参数:
    - fps_19: 19 FPS 视频的帧率
    - fps_30: 30 FPS 视频的帧率
    - duration_19: 19 FPS 视频的总时长（秒）
    - duration_30: 30 FPS 视频的总时长（秒）

    返回:
    - matches: 一个列表，表示在短视频时间范围内，19 FPS 视频每帧对应的 30 FPS 视频的帧编号
    """

    # 获取两个视频的较短持续时间
    min_duration = min(duration_19, duration_30)

    # 计算在较短时间范围内的两个视频的时间戳
    timestamps_19 = np.arange(0, min_duration, 1 / fps_19)
    timestamps_30 = np.arange(0, min_duration, 1 / fps_30)

    # 用于保存 19 FPS 视频中每帧对应的最近的 30 FPS 帧编号
    matches = []

    # 找到 30 FPS 视频中时间最接近当前 19 FPS 时间戳的帧索引
    for t in timestamps_19:
        closest_index = np.argmin(np.abs(timestamps_30 - t))
        matches.append(closest_index)

    return matches



def mse(y1, y2):
    return np.mean((y1 - y2) ** 2)


if __name__ == "__main__":

    # find the path
    path = 'xxxxxxx'
    name = "constant"
    fps_exp = 20

    # load the figures
    input_figures = load_all_figures(path)
    input_vector = np.full(35, 5e-3)

    # 示例用法
    V_in_exp = 200  # 200 mm/min

    fps_19 = 19
    fps_30 = 30

    # real time from the
    duration_R = len(input_figures) / fps_exp

    # stripe Length * stripe count * layer count * 1000 （mm->m）* 60 (min->s) / V (G code)
    duration_C = STRIPE_NUM * LAYER_HEIGHT * SIMU_L * 1000 * 60 / V_in_exp

    # print("在较短持续时间内 19 FPS 视频的帧对应的 30 FPS 视频的最近帧索引:", matched_frames)
    matched_frames = nearest_neighbor_match_varying_duration(fps_19, fps_30, duration_R, duration_C)

    # calculate the mse
    mse_total = Commingle(input_vector, input_figures)
    np.save('mse_total' + name, mse_total)
