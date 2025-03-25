from functions import *
import os
import numpy as np


def worker_agent(V):
    thermal = Thermal()
    thermal.Reset()

    V0 = V

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    # solid laser power
    P = 600  # changed in the physical world code

    stripe_count = 0
    global_count = 0

    losses = []
    global_counts = []

    # 构建存储文件夹
    current_folder = '.'
    folder_name = str(INNER_TRANS) + "_" + str(INNER_TRANS_) + '_' + str(NOTE)
    save_path = os.path.join(current_folder, folder_name)
    os.makedirs(save_path, exist_ok=True)

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y

        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0
            step_count = 0

            # check whether last stripe boundary offset
            if stripe // (STRIPE_NUM - 1) == 1:
                right_bound = RIGHT_BOUND

            else:
                right_bound = False

            # heater is working
            for step in range(CELL_SIZE_X):
                # Execute One Step
                _, _, _ = thermal.Step(P, V0, heat_loc, True, right_bound=right_bound)
                loss = loss_calculation(thermal.current_T, heat_loc)

                # Update Location
                heat_loc[0] += 1
                step_count += 1
                global_count += 1

                losses.append(loss)

                # # # additional assignments
                # if display:
                #
                #     # if global_count % 10 == 0:
                #     if global_count == 1420 or global_count == 1450 or global_count == 2850:
                #
                #         # np.save('physical-' + str(global_count) + "_1trans_0.1trans_30body", physical_data)
                #         # np.save('simulation-' + str(global_count) + "_1trans_0.1trans_30body", simulation_data)
                #         # np.save('simulation_previous-' + str(global_count) + "_1trans_0.1trans_30body", simulation_previous_data)
                #
                #         # name_ = str(global_count) + str(physical_data)
                #         name_ = 'physical_data_' + str(global_count) + '.npy'
                #         file_path = os.path.join(save_path,  name_)  # 创建每个数组的保存路径
                #         np.save(file_path, physical_data)  # 保存数组

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):

                # Waiting the heater between stripe
                _, _, _ = thermal.Step(P, V0, heat_loc, False, right_bound=right_bound)
                loss = 0

                global_count += 1
                step_count += 1

                losses.append(loss)

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            stripe_count += 1

            print('stripe_count', stripe_count)

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

    return losses, global_counts, save_path


# TODO: finish cost function
def loss_calculation(physical_matrix, loc):

    # further development
    location = [loc[0], loc[1]]

    # calculate the loss
    if loc[1] <= 3 * STRIPE_NUM:
        loss = physical_matrix[location[0], location[1]]
    else:
        loss = 0

    return loss


if __name__ == "__main__":
    vector = np.random.uniform(V_MIN, V_MAX, 35)
    cost = worker_agent(vector)
