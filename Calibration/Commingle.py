from Functions_calibration import *
from Data_Treatment import *


def commingle(input_vector):

    thermal = Thermal()
    V0 = VS  # solid input speed -> constant speed for calibration (240mm/min, 360mm/min, 480mm/min)

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 500
    global_count = 0
    thermal.Reset()
    count = 0

    # first layer include the upper layer, full layer include upper layer and the second layer
    T_first_layers = []
    T_full_layer_count = []

    assert len(input_vector) == LAYER_HEIGHT * STRIPE_NUM, " V num do not match stripe num "

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y
        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0

            # heater is working
            for step in range(CELL_SIZE_X):

                # calculate the global step number
                global_count += 1

                # Execute One Step
                thermal.Step(P, V0, heat_loc, True)

                # calculate the thermal distribution
                real_T = thermal.current_T * thermal.Actuator
                # concat_T = thermal.previous_T * (1 - thermal.Actuator)

                # Update Location
                heat_loc[0] += 1

                # record the temperature distribution
                T_first_layers.append(real_T)
                # T_full_layer_count.append(concat_T)

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):
                thermal.Step(P, V0, heat_loc, False)
                global_count += 1

                real_T = thermal.current_T * thermal.Actuator
                # concat_T = thermal.previous_T * (1 - thermal.Actuator) + real_T

                T_first_layers.append(real_T)
                # T_full_layer_count.append(concat_T)

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            count += 1

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return T_first_layers, T_full_layer_count


def MSE(simu, real):

    # assert length of the simulation equal
    assert len(simu) == len(real), "The lengths of 'simu' and 'real' do not match."

    mses = []

    for i in range(len(simu)):
        mse = mean_squared_rate(simu[i], real[i])
        mses.append(mse)

    return mses


# 均方误差比率
def mean_squared_rate(x_pred, x_real):
    epsilon = 1e-8
    error_rate_matrix = np.abs(x_pred - x_real) / (np.abs(x_real) + epsilon) * 100
    mean_error_rate = np.mean(error_rate_matrix)
    return mean_error_rate


# 最小误差帧映射函数
def get_min_error_mapping(input_frame_count, target_frame_count):

    """
    Compute the optimal mapping from input frames to target frames with minimal error.

    Args:
        input_frame_count (int): Number of input frames.
        target_frame_count (int): Number of target frames.

    Returns:
        list: List of input frame indices corresponding to the target frames.
    """

    # Calculate the step size
    input_frame_count = 175
    target_frame_count = 56     # maybe 57

    step = (input_frame_count - 1) / (target_frame_count - 1)

    # Generate the indices
    corresponding_frames = [round(i * step) for i in range(target_frame_count)]

    return corresponding_frames


if __name__ == "__main__":

    # find the path
    path = "D:\\test\\calibration\\csv"
    real_data = calibration(path)

