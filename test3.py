from functions import *
from parameter import *
import numpy as np
import matplotlib.pyplot as plt

thermal = Thermal()

loc = [100, 39, 0]
P = 600
V = VS
x = []
y = []
temperature = []

def check_diffusion(loc):
    temp1 = np.zeros((10, 10))
    temp2 = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            x, y = loc[0] - 10 + i, loc[1] - 10 + j

            upper = thermal.current_T[x - 1, y]

            lower = thermal.current_T[x + 1, y]
            X = upper + lower
            left = thermal.current_T[x, y + 1]
            right = thermal.current_T[x, y - 1]
            Y = left + right
            temp1[i][j] = X - 2 * thermal.current_T[x, y]
            temp2[i][j] = Y - 2 * thermal.current_T[x, y]
    return temp1, temp2


cell = 100
time_ = 100

import time

time1 = time.time()
""" 在块体中间进行加热，不涉及 diffusion 的传导"""
for j in range(cell):
    for i in range(time_):

        heat_matrix = thermal.Heat_matrix(P, loc)
        T = thermal.current_T

        # diffuse temperature
        X_T = (thermal.T_upper + thermal.T_lower) @ thermal.current_T
        Y_T = thermal.current_T @ (thermal.T_left + thermal.T_right)

        X_delta_1 = X_T / DELTA_X ** 2
        Y_delta_1 = Y_T / DELTA_Y ** 2

        # Uconv = h * (thermal.current_T - Ta) / DELTA_Z
        Uconv = 0
        Us_input = (heat_matrix - Uconv) / Kt

        T_next_1 = (X_delta_1 + Y_delta_1 + Us_input) * ALPHA * (t / time_) + thermal.current_T

        thermal.current_T = T_next_1

        temperature.append(thermal.current_T[100, 39])

thermal.Display(thermal.current_T)
time2 = time.time()

# test_point = T_next_1[2, 2]
# print('temp', X_delta_1[2, 2], Y_delta_1[2, 2], Z_delta_1[2][2])
# y.append(test_point)
# x.append(i)

x = np.arange(time_ * cell)
y = temperature

plt.plot(x, y)
plt.title('折线图示例')
plt.show()

print('time', time2 - time1)
print(max(temperature))
