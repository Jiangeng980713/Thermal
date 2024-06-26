import time
from math import *
from parameter import *
import numpy as np
import matplotlib.pyplot as plt


class Thermal():
    def __init__(self):

        # init matrix
        self.current_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta

        # in-process location
        self.Actuator = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # requires update

        # boundary_ for the finished layers
        self.boundary_ = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.boundary_[0, :] = 1
        self.boundary_[CELL_SIZE_X - 1, :] = 1
        self.boundary_[:, 0] = 1
        self.boundary_[:, CELL_SIZE_Y - 1] = 1

        # heater information
        self.row, self.column = HEATER_ROW, HEATER_COLUMN
        self.heater = self.create_2d_gaussian(self.row, self.column)

        # build-in matrix
        self.diag_matrix_X = np.diag(np.ones(CELL_SIZE_X))
        self.zeros_X = np.zeros((1, CELL_SIZE_X))
        self.diag_matrix_Y = np.diag(np.ones(CELL_SIZE_Y))
        self.zeros_Y = np.zeros((1, CELL_SIZE_Y))

        # add "-" in the transaction matrix
        # upper transactions matrix # C @ (A-B) @ T = ^T   # checked
        self.A_upper = np.vstack((self.diag_matrix_X, self.zeros_X))
        self.B_upper = np.vstack((self.zeros_X, self.diag_matrix_X))
        self.B_upper[0][0] = 1
        self.C_upper = np.hstack((self.diag_matrix_X, self.zeros_X.T))
        self.T_upper = - self.C_upper @ (self.A_upper - self.B_upper)

        # lower transactions matrix # C @ (A-B) @ T = ^T   # checked
        self.A_lower = np.vstack((self.zeros_X, self.diag_matrix_X))
        self.B_lower = np.vstack((self.diag_matrix_X, self.zeros_X))
        self.B_lower[CELL_SIZE_X][CELL_SIZE_X - 1] = 1
        self.C_lower = np.hstack((self.zeros_X.T, self.diag_matrix_X))
        self.T_lower = - self.C_lower @ (self.A_lower - self.B_lower)

        # right transactions matrix  # T @ (A-B) @ C = ^T  # checked
        self.A_right = np.hstack((self.zeros_Y.T, self.diag_matrix_Y))
        self.B_right = np.hstack((self.diag_matrix_Y, self.zeros_Y.T))
        self.B_right[CELL_SIZE_Y - 1][CELL_SIZE_Y] = 1
        self.C_right = np.vstack((self.zeros_Y, self.diag_matrix_Y))
        self.T_right = - (self.A_right - self.B_right) @ self.C_right

        # left transactions matrix  # T @ (A-B) @ C = ^T   # checked
        self.A_left = np.hstack((self.diag_matrix_Y, self.zeros_Y.T))
        self.B_left = np.hstack((self.zeros_Y.T, self.diag_matrix_Y))
        self.B_left[0][0] = 1
        self.C_left = np.vstack((self.diag_matrix_Y, self.zeros_Y))
        self.T_left = - (self.A_left - self.B_left) @ self.C_left

        # layer-wise velocity
        self.Vs = VS

    def Display(self, matrix):
        plt.imshow(matrix)
        plt.show()

    # def save_FIG(self, matrix, name):
    #     plt.imshow(matrix)
    #     plt.savefig("figure" + name)

    def transform_heat(self):
        return 0

    # "load one temperature distribution from exist situation"
    # def Load(self, load_Temperature, load_Previous, load_body):
    #     self.current_T = load_Temperature
    #     self.previous_T = load_Previous
    #     self.body = load_body
    #     self.Actuator = self.transform_heat()

    def Reset(self):
        self.current_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.Actuator = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

    def Check_boundary(self, loc):

        temp = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

        # in the middle
        if 0 < loc[1] < CELL_SIZE_Y - 1:
            temp[:, 0] = 1  # left
            temp[0, :loc[1] + 1] = 1  # upper
            temp[CELL_SIZE_X - 1, :loc[1] + 1] = 1  # lower
            temp[:, loc[1]] = 1  # right
            temp[loc[0]:, loc[1]] = 0  # remove right
            temp[loc[0]:, loc[1] - 1] = 1  # add secondary right

        # the left boundary
        if loc[1] == 0:
            temp[:loc[0], 0] = 1

        # the right boundary
        if loc[1] == CELL_SIZE_Y - 1:
            temp[:, 0] = 1  # left
            temp[0, :loc[1] + 1] = 1  # upper
            temp[CELL_SIZE_X - 1, :CELL_SIZE_Y - 2] = 1  # down
            temp[:loc[0], loc[1]] = 1  # right-1
            temp[loc[0]:, loc[1] - 1] = 1  # right-2

        return temp

    def create_2d_gaussian(self, rows, cols):
        x, y = np.meshgrid(np.arange(cols) - (cols - 1) / 2, np.arange(rows) - (rows - 1) / 2)
        gauss_kernel = np.exp(-2 * (x ** 2 + y ** 2) / ((HEATER_ROW // 2) ** 2))
        return gauss_kernel / np.max(gauss_kernel)

    def Heat_matrix(self, P, loc):

        Q = 2 * LAMDA * P / (pi * Rb ** 2)
        Q = Q / (SIMU_H / LAYER_HEIGHT)    # volume heater

        heat_matrix_temp = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        heat_matrix_current = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # activate cell

        # the boundary location for heater on the plate
        boundary_x, boundary_y = loc[0] - self.row // 2, loc[1] - self.column // 2

        for i in range(len(self.heater[0])):
            for j in range(len(self.heater)):
                temp_x, temp_y = boundary_x + i, boundary_y + j
                # outer boundary skip the value
                if temp_x < 0 or temp_x >= CELL_SIZE_X or temp_y < 0 or temp_y >= CELL_SIZE_Y:
                    continue
                else:
                    heat_matrix_temp[temp_x][temp_y] = self.heater[i][j]
                    heat_matrix_current[temp_x][temp_y] = 1  # activate cell

        heat_matrix_ = heat_matrix_temp * Q
        self.Actuator = self.Actuator + heat_matrix_current
        self.Actuator[self.Actuator != 0] = 1
        return heat_matrix_

    def Step(self, P, V, loc):

        Time_rate = V / self.Vs
        # time1 = time.time()

        """ Time_rate donot cause divergence, TIME_SCALE harm!"""
        # 因为 t / TIME_SCALE 是一种时间划分方式，TIME_RATE是重复多少次传递，并不改变传导等物理过程的时间差分跨度
        # 相当于 for i 传递差分， for j 重复 i 的循环
        for i in range(int(TIME_SCALE * Time_rate)):

            # first layer - convention to air
            Uc_temp = h * (self.current_T - Ta) / DELTA_Z
            Uconv_now = - Uc_temp * self.Actuator

            # second layer - convention from the uncovered
            Uc_temp_ = h * (self.previous_T - Ta) / DELTA_Z
            Uconv_previous = - Uc_temp_ * (np.ones((CELL_SIZE_X, CELL_SIZE_Y)) - self.Actuator)

            # heater input
            Heat_matrix = self.Heat_matrix(P, loc)
            Us_now = Heat_matrix + Uconv_now

            """ boundary condition """
            # boundary = self.Check_boundary(loc)

            # first layer - boundary convention
            # Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

            # second layer - boundary convention
            # Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

            """ zeros the boundary matrix"""
            # Uc_boundary = Uc_boundary_ = 0

            # temperature diffusion - first layer
            X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
            Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
            Z_delta_1 = ((self.previous_T - self.current_T) * self.Actuator) / DELTA_Z ** 2
            T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_now / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

            # temperature diffusion - second layer
            X_delta_2 = ((self.T_upper + self.T_lower) @ self.previous_T) / DELTA_X ** 2
            Y_delta_2 = (self.previous_T @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            Z_delta_2 = ((self.current_T - self.previous_T) * self.Actuator + (self.body - self.previous_T)) / DELTA_Z ** 2
            T_next_2 = (X_delta_2 + Y_delta_2 + Z_delta_2 + Uconv_previous / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T

            # temperature diffusion - body
            """Problem: 1. Body的温度应该如何计算（通过实验/仿真进行标定吧）"""
            T_nest_body = T_next_2.copy()

            # update the temperature in one small cell
            self.current_T = T_next_1.copy()
            self.previous_T = T_next_2.copy()
            self.body = T_nest_body.copy()
            self.Actuator = self.Actuator

        # time2 = time.time()

        return T_next_1, T_next_2, T_nest_body

    # Layer-wise Temperature Update
    def reset(self):
        self.body = np.average(self.previous_T.copy())
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * np.average(self.current_T.copy())
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

    " 更新一下目标函数"
    def Cost_function(self, loc):

        current_T = self.current_T * self.Actuator

        if loc[1] >= GRADIENT_LENGTH * INTERVAL_Y:
            heat_gradient = current_T[:, loc[0] - GRADIENT_LENGTH * INTERVAL_Y:loc[0]]
        else:
            heat_gradient = current_T[:, :loc[0]]

        Average_T = np.average(self.current_T)
        cost = np.sum((heat_gradient - Average_T) / Tm ** 2)

        return cost
