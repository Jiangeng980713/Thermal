from Functions_calibration import *
from Data_Treatment import *


def simulation(input_vector):

    thermal = Thermal()

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 500

    thermal.Reset()
    count = 0
    T_first_layer_count = []
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

                # # calculate the global step number
                # global_count = layer * STRIPE_NUM * CELL_SIZE_X + stripe * CELL_SIZE_X + step

                # Execute One Step
                thermal.Step(P, input_vector[count], heat_loc, True)

                # calculate the thermal distribution
                real_T = thermal.current_T * thermal.Actuator
                concat_T = thermal.previous_T * (1 - thermal.Actuator)

                # if step - (step//70)*90 == 0:
                #      print('step', step, step//10)
                #      print('stripe, step', stripe, step)
                #      thermal.Display(thermal.current_T)

                # Update Location
                heat_loc[0] += 1

                # record the temperature distribution
                T_first_layer_count.append(real_T)
                T_full_layer_count.append(concat_T)

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):
                thermal.Step(P, SLEEP_SPEED, heat_loc, False)

                real_T = thermal.current_T * thermal.Actuator
                concat_T = thermal.previous_T * (1- thermal.Actuator)

                T_first_layer_count.append(real_T)
                T_full_layer_count.append(concat_T)

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            count += 1

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return T_first_layer_count, T_full_layer_count


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


if __name__ == "__main__":

    # find the path
    path = 'xxxxxxx'
    real_data = calibration(path)

    # simulated surrogate model
    input_vector = np.full(35, 5e-3)
    simu_data_half, simu_data_full = simulation(input_vector)

    # calculate error sequence
    mse_sequence = MSE(simu_data_full, real_data)
    np.save("mse", mse_sequence)