from functions import *
import numpy as np


def worker_agent(P):
    thermal = Thermal()
    thermal.Reset()

    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER

    stripe_count = 0

    body_stripe_accumulation = []  # for a body layer equivalent

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y

        # multi_stripe_total = []       # for the inner layer equivalent

        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0
            step_count = 0
            single_stripe_Ts = []

            # check whether last stripe boundary offset
            if stripe // (STRIPE_NUM - 1) == 1:
                right_bound = RIGHT_BOUND

            else:
                right_bound = False

            # Extract effective P
            P_input = P[stripe_count]

            # Heater is working
            for step in range(CELL_SIZE_X):

                # Execute One Step
                _, _, _ = thermal.Step(P_input, VS, heat_loc, True, right_bound=right_bound)

                stripe_actuator = thermal.pick_up(heat_loc)

                # print("stripe_num", stripe)
                # if step == 150:
                #     thermal.Display(stripe_actuator)

                stripe_T = thermal.current_T * stripe_actuator

                # Update Location
                heat_loc[0] += 1
                step_count += 1

                single_stripe_Ts.append(np.sum(stripe_T))
                # print("single_stripe_Ts", len(single_stripe_Ts))

            # add the sleep time and wait for heater moving
            for step in range(TIME_SLEEP):
                # Waiting the heater between stripe
                _, _, _ = thermal.Step(P_input, VS, heat_loc, False, right_bound=right_bound)

                step_count += 1

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            stripe_count += 1

            # sum up stripe Temperature
            single_stripe_accumulation = np.sum(np.array(single_stripe_Ts))  # temperature accumulation in one layer

            # record layer-wise stripe heat
            # multi_stripe_total.append(single_stripe_total)

            # record body-wise stripe heat
            body_stripe_accumulation.append(single_stripe_accumulation)

        # one layer is done
        # loss = loss_calculation_layer(multi_stripe_total)             # calculate the layer-wise reward
        # losses.append(loss)

        heat_loc[2] += 1
        thermal.reset()

    # one manufacturing is done
    global_loss, stripe_loss = loss_calculation_body(body_stripe_accumulation)  # calculate the global reward

    # losses.append(loss)
    #
    # # sum up loss list
    # losses = sum(losses)

    return global_loss, stripe_loss, body_stripe_accumulation


def loss_calculation_body(stripe_Ts):

    body_average_thermal = sum(stripe_Ts) / (STRIPE_NUM * LAYER_HEIGHT)

    # print('body_average_thermal', body_average_thermal)

    loss_per = [T - body_average_thermal / body_average_thermal for T in stripe_Ts[:STRIPE_NUM * LAYER_HEIGHT]]

    global_loss = np.var(loss_per) ** 0.5

    # np.save('worker_loss_per', loss_per)

    return global_loss, loss_per


# def loss_calculation_layer(Multiple_stripe_T):
#     Layer_average_T = sum(Multiple_stripe_T) / (STRIPE_NUM * LAYER_HEIGHT)
#     DEMAND_T = 0
# 
#     stripe_loss = []
# 
#     for i in range(STRIPE_NUM):
#         single_stripe_total = Multiple_stripe_T[i]
#         loss = (single_stripe_total - Layer_average_T) ** 2
#         stripe_loss.append(loss)
# 
#     layer_loss = (sum(stripe_loss) ** 0.5) / Tm
# 
#     return layer_loss


# def loss_calculation(physical_matrix, actuator, loc):
#     thermal_matrix = physical_matrix * actuator
#
#     # Smart-scan loss - thermal equivalent 热均衡
#     if LOSS_EQUIVALENT:
#
#         # find all non-zero element
#         non_zero_elements = thermal_matrix[thermal_matrix != 0]
#         list_length = len(non_zero_elements)
#
#         if non_zero_elements.size > 0:
#             average_thermal = np.mean(non_zero_elements)
#         else:
#             raise ValueError("Error: 选取的子矩阵中没有非零元素！")
#
#         upper = np.sum((non_zero_elements - average_thermal) ** 2)
#
#         lower = list_length * Tm ** 2
#         loss = np.sqrt(upper / lower)
#
#     # Li Sun loss - thermal gradient 热梯度
#     else:
#
#         # 设定中心点
#         center_x, center_y = loc[0], loc[1]
#
#         # 计算边界，防止超出索引范围
#         x_start, x_end = max(center_x - LOSS_RADIUS, 0), min(center_x + LOSS_RADIUS + 1, thermal_matrix.shape[0])
#         y_start, y_end = max(center_y - LOSS_RADIUS, 0), min(center_y + LOSS_RADIUS + 1, thermal_matrix.shape[1])
#
#         # 提取子矩阵
#         sub_matrix = thermal_matrix[x_start:x_end, y_start:y_end]
#
#         # 获取非零元素的坐标
#         non_zero_indices = np.argwhere(sub_matrix != 0)
#
#         if non_zero_indices.size == 0:
#             raise ValueError("Error: 选取的子矩阵中没有非零元素！")
#
#         # 提取非零值
#         non_zero_values = sub_matrix[sub_matrix != 0]
#
#         # 获取最大最小值
#         max_val = np.max(non_zero_values)
#         min_val = np.min(non_zero_values)
#
#         # 找到最大值和最小值的索引（相对于子矩阵）
#         max_pos = tuple(non_zero_indices[np.argmax(non_zero_values)])
#         min_pos = tuple(non_zero_indices[np.argmin(non_zero_values)])
#         euclidean_dist = np.linalg.norm(np.array(max_pos) - np.array(min_pos))
#
#         if max_pos == min_pos:
#             raise ValueError("Error: 最大值与最小值的位置相同，无法计算有效的距离！")
#
#         loss = (max_val - min_val) / euclidean_dist
#
#     return loss


def Display(matrix):
    plt.imshow(matrix)
    plt.show()


if __name__ == "__main__":
    # vector = np.random.uniform(P_Min, P_Max, 35)

    vector = np.full((35,), 600)

    loss, loss_per, thermal_distribution = worker_agent(vector)
    print("loss", loss)
    print("loss_distribution", loss_per)
    print('thermal_distribution', thermal_distribution)
