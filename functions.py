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

        """ PT """
        # heater information
        self.row, self.column = HEATER_ROW, HEATER_COLUMN
        self.heater = self.create_2d_gaussian(self.row, self.column)

        """ Diffusion Transaction Matrix """
        # build matrix
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

    def transform_heat(self):
        return 0

    "load one temperature distribution from exist situation"

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

    """ requires for double check"""

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

    """ 如何标定热源是当前的一个问题 """

    def Heat_matrix(self, P, loc):

        Q = 2 * LAMDA * P / (pi * Rb ** 2)
        # volume heater
        Q = Q / (SIMU_H / LAYER_HEIGHT)

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
        return heat_matrix_, heat_matrix_current

    """Problem: 1. Body的温度应该如何计算（通过实验/仿真进行标定吧）"""

    def Diffusion(self, P, V, loc):

        Time_rate = V / self.Vs

        for i in range(TIME_SCALE):

            # first layer - convention to air
            Uc_temp = h * (self.current_T - Ta) / DELTA_Z
            Uconv_now = - Uc_temp * self.Actuator

            # second layer - convention from the uncovered
            Uc_temp_ = h * (self.previous_T - Ta) / DELTA_Z
            Uconv_previous = - Uc_temp_ * (np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * self.Actuator)

            # heater input
            Heat_matrix, Heat_matrix_current = self.Heat_matrix(P, loc)
            Us_now = Heat_matrix + Uconv_now

            """ 边界条件 """
            boundary = self.Check_boundary(loc)

            # first layer - boundary convention
            # Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

            # second layer - boundary convention
            # Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

            """ zeros the boundary matrix"""
            Uc_boundary = Uc_boundary_ = 0

            # temperature diffusion
            if True:  # layer one
                X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
                Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
                # Z_delta_1 = ((self.current_T - self.previous_T) * self.Actuator) / DELTA_Z ** 2
                Z_delta_1 = 0
                T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Uc_boundary / Kt + Us_now / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

                """ 这个其中的内容和上面相同，完全没有变化 """
                X_delta_2 = (self.T_upper + self.T_lower) @ self.current_T
                Y_delta_2 = self.current_T @ (self.T_left + self.T_right)
                Z_delta_2 = (self.current_T - self.previous_T) * self.Actuator + (self.previous_T - self.body)  # test
                T_next_2 = (X_delta_2 / DELTA_X ** 2 + Y_delta_2 / DELTA_Y ** 2 + Z_delta_2 / DELTA_Z ** 2 + Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test

                T_nest_body = T_next_2

            # layer number higher than 1
            """ X-delta-1 和 Y-delta-1 没有加上 actuators """
            # if loc[2] >= 1:  # second layer and higher
            #     # upper layer
            #     X_delta_1 = self.current_T @ (self.T_upper + self.T_lower)
            #     Y_delta_1 = (self.T_left + self.T_right) @ self.current_T
            #     Z_delta_1 = (self.current_T - self.previous_T) * self.Actuator  # test
            #     T_next_1 = (- X_delta_1 / DELTA_X ** 2 - Y_delta_1 / DELTA_Y ** 2 - Z_delta_1 / DELTA_Z ** 2
            #                 - Uc_boundary / Kt + Us_now / Kt) * ALPHA * t * Time_rate + self.current_T  # test
            #
            #     # second upper layer  # test
            #     X_delta_2 = self.current_T @ (self.T_upper + self.T_lower)
            #     Y_delta_2 = (self.T_left + self.T_right) @ self.current_T
            #     Z_delta_2 = (self.current_T - self.previous_T) * self.Actuator + (self.previous_T - self.body)  # test
            #     T_next_2 = (- X_delta_2 / DELTA_X ** 2 - Y_delta_2 / DELTA_Y ** 2 - Z_delta_2 / DELTA_Z ** 2
            #                 - Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test
            #
            #     # diffuse to body - third upper layer
            #     T_nest_body = ((self.previous_T - self.body) / DELTA_Z ** 2 - Uc_boundary_ / Kt) * ALPHA * t * Time_rate

            # update the temperature in one small cell
            self.current_T = T_next_1.copy()
            self.previous_T = T_next_2.copy()
            self.body = T_nest_body.copy()

        return T_next_1, T_next_2, T_nest_body, Heat_matrix_current

    # Track-wise Temperature Update
    def Step(self, P, V, loc):

        # Update Temperature
        T_next_1, T_next_2, T_body, Heat_matrix_current = self.Diffusion(P, V, loc)

        # temp = self.current_T.copy()
        # temp[temp != 0] = 1
        # self.Display(temp)

        # update the active cells
        self.Actuator = self.Actuator + Heat_matrix_current
        self.Actuator[self.Actuator != 0] = 1
        # self.Display(self.Actuator)

    # Layer-wise Temperature Update
    def reset(self):

        self.body = np.average(self.previous_T.copy())
        self.previous_T = np.average(self.current_T.copy())
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

    # Calculate temperature in 3 stripes
    def Cost_function(self, loc):

        current_T = self.current_T * self.Actuator

        if loc[0] >= 3:
            heat_dis = current_T[:, loc[0] - 3 * INTERVAL_Y:loc[0]]
        else:
            heat_dis = current_T[:, :loc[0]]

        Average_T = np.average(self.current_T)
        cost = np.sum((heat_dis - Average_T) / Tm ** 2)

        return cost
