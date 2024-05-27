from functions import *
from parameter import *
import numpy as np
import matplotlib.pyplot as plt


thermal = Thermal()

loc = [100, 39, 0]
P = 300
V = VS
x = []
y = []

""" 在块体中间进行加热，不涉及 diffusion 的传导"""
for i in range(1000):

    heat_matrix, _ = thermal.Heat_matrix(P, loc)
    # thermal.Display(heat_matrix)

    # X_delta_1 = ((thermal.T_upper + thermal.T_lower) @ thermal.current_T) / DELTA_X ** 2
    # Y_delta_1 = (thermal.current_T @ (thermal.T_left + thermal.T_right)) / DELTA_Y ** 2
    # Z_delta_1 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

    T_next_1 = (heat_matrix / Kt) * ALPHA * t + thermal.current_T
    thermal.current_T = T_next_1
    thermal.Display(T_next_1)

    # test_point = T_next_1[2, 2]
    # print('temp', X_delta_1[2, 2], Y_delta_1[2, 2], Z_delta_1[2][2])
    # y.append(test_point)
    # x.append(i)

    # 创建折线图
    # plt.plot(x, y)

    # 添加标题和轴标签
    # plt.title('折线图示例')
    # plt.xlabel('x轴')
    # plt.ylabel('y轴')

    # 显示图形
    # plt.show()


