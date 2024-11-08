from Functions_calibration import *


def Execute_calibration(input_vector, real_T):
    thermal = Thermal()

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 500

    thermal.Reset()
    count = 0
    mse_count = []

    assert len(input_vector) == LAYER_HEIGHT * STRIPE_NUM, " V num do not match stripe num"

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
                thermal.Step(P, input_vector[count], heat_loc, DISPLAY_COUNT)
                mse = MSE(thermal.current_T, real_T)
                mse_count.append(mse)

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


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # vector = np.random.uniform(V_MIN, V_MAX, 35)
    vector = np.full(35, 5E-3)
    time1 = time.time()
    real_T = 0
    mse_count = Execute_calibration(vector, real_T)
    time2 = time.time()
    print(time2 - time1)
    print("MSE_count", mse_count)
    print("MSE_value", sum(mse_count))
