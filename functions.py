from math import *
import numpy as np
from parameter import *

import numpy as np
import matplotlib.pyplot as plt


class Thermal():
    def __init__(self):

        # init matrix
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.previous_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.body = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

        # in-process location
        self.current_exist = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # requires update

        # boundary_ for the finished layers
        self.boundary_ = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.boundary_[0, :] = 1
        self.boundary_[CELL_SIZE_X - 1, :] = 1
        self.boundary_[:, 0] = 1
        self.boundary_[:, CELL_SIZE_Y - 1] = 1

        """PT"""
        # heater information
        self.row, self.column, std = 15, 15, 2.5
        self.heater = self.create_2d_gaussian(self.row, self.column, std=std)

        """ Diffusion Transaction Matrix """
        # build matrix
        self.diag_matrix_X = np.diag(np.ones(CELL_SIZE_X))
        self.zeros_X = np.zeros((1, CELL_SIZE_X))
        self.diag_matrix_Y = np.diag(np.ones(CELL_SIZE_Y))
        self.zeros_Y = np.zeros((1, CELL_SIZE_Y))

        # upper transactions matrix # C @ (A-B) @ T = ^T   # checked
        self.A_upper = np.vstack((self.diag_matrix_X, self.zeros_X))
        self.B_upper = np.vstack((self.zeros_X, self.diag_matrix_X))
        self.B_upper[0][0] = 1
        self.C_upper = np.hstack((self.diag_matrix_X, self.zeros_X.T))
        self.T_upper = self.C_upper @ (self.A_upper - self.B_upper)

        # lower transactions matrix # C @ (A-B) @ T = ^T   # checked
        self.A_lower = np.vstack((self.zeros_X, self.diag_matrix_X))
        self.B_lower = np.vstack((self.diag_matrix_X, self.zeros_X))
        self.B_lower[CELL_SIZE_X][CELL_SIZE_X - 1] = 1
        self.C_lower = np.hstack((self.zeros_X.T, self.diag_matrix_X))
        self.T_lower = self.C_lower @ (self.A_lower - self.B_lower)

        # right transactions matrix  # T @ (A-B) @ C = ^T  # checked
        self.A_right = np.hstack((self.zeros_Y.T, self.diag_matrix_Y))
        self.B_right = np.hstack((self.diag_matrix_Y, self.zeros_Y.T))
        self.B_right[CELL_SIZE_Y - 1][CELL_SIZE_Y] = 1
        self.C_right = np.vstack((self.zeros_Y, self.diag_matrix_Y))
        self.T_right = (self.A_right - self.B_right) @ self.C_right

        # left transactions matrix  # T @ (A-B) @ C = ^T   # checked
        self.A_left = np.hstack((self.diag_matrix_Y, self.zeros_Y.T))
        self.B_left = np.hstack((self.zeros_Y.T, self.diag_matrix_Y))
        self.B_left[0][0] = 1
        self.C_left = np.vstack((self.diag_matrix_Y, self.zeros_Y))
        self.T_left = (self.A_left - self.B_left) @ self.C_left

        # layer-wise velocity
        """ test for determine"""
        self.Vs = VS

    def Display(self, matrix):
        plt.imshow(matrix)
        plt.show()

    def Reset(self):
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.current_exist = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # requires update

    """ requires for double check"""

    def check_boundary(self, loc):

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

    """ Require further discussion"""

    def create_2d_gaussian(self, rows, cols, std):
        x, y = np.meshgrid(np.arange(cols) - (cols - 1) / 2, np.arange(rows) - (rows - 1) / 2)
        gauss_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
        return gauss_kernel / np.max(gauss_kernel)

    # Transform Heater into Matrix
    def Heat_matrix(self, P, loc):
        """ PT-Rb"""
        Q = 2 * LAMDA * P / (pi * Rb ** 2) * 5000000

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
        self.Display(heat_matrix_)
        return heat_matrix_, heat_matrix_current

    def Convention_matrix(self, loc):

        # check boundary
        boundary = self.check_boundary(loc)

        # convention to air
        Uc_temp = h * (self.current_T - Ta) / DELTA_Z
        Uc_matrix = - Uc_temp * self.current_exist  # 是否应该有"-"号

        # convention from the uncovered second layer
        Uc_temp_ = h * (self.previous_T - Ta) / DELTA_Z
        Uc_matrix_ = - Uc_temp_ * (np.ones((CELL_SIZE_X, CELL_SIZE_Y)) - self.current_exist)  # 是否应该有"-"号

        # boundary convention in the first layer
        Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

        # boundary convention in the second layer
        Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

        return Uc_matrix, Uc_matrix_, Uc_boundary, Uc_boundary_

    """Problem: 1. V 没有统一 2. Body的温度应该如何计算"""

    def Diffusion(self, P, V, loc):

        Uconv_now, Uconv_previous, Uc_boundary, Uc_boundary_ = self.Convention_matrix(loc)

        # zeros the convention
        Uconv_now = Uconv_previous = Uc_boundary = Uc_boundary_ = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        Heat_matrix, Heat_matrix_current = self.Heat_matrix(P, loc)

        Us_now = Uconv_now + Heat_matrix
        Time_rate = V / self.Vs

        if loc[2] == 0:

            X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T - 2 * self.current_T) / DELTA_X ** 2
            Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) - 2 * self.current_T) / DELTA_Y ** 2
            Z_delta_1 = ((self.current_T - self.previous_T) * self.current_exist) / DELTA_Z ** 2   # test
            input = Us_now / Kt
            total = X_delta_1 + Y_delta_1 + Z_delta_1 + Uc_boundary / Kt + input
            T = total
            T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Uc_boundary / Kt + input) * ALPHA * t + self.current_T

            self.Display(T_next_1)

            X_delta_2 = (self.T_upper + self.T_lower) @ self.current_T - 2 * self.current_T
            Y_delta_2 = self.current_T @ (self.T_left + self.T_right) - 2 * self.current_T
            Z_delta_2 = (self.current_T - self.previous_T) * self.current_exist + (self.previous_T - self.body)  # test
            T_next_2 = (X_delta_2 / DELTA_X ** 2 + Y_delta_2 / DELTA_Y ** 2 + Z_delta_2 / DELTA_Z ** 2 + Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test

            T_nest_body = T_next_2

        # layer number higher than 1
        if loc[2] >= 1:
            # upper layer
            X_delta_1 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_1 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_1 = (self.current_T - self.previous_T) * self.current_exist  # test
            T_next_1 = (- X_delta_1 / DELTA_X ** 2 - Y_delta_1 / DELTA_Y ** 2 - Z_delta_1 / DELTA_Z ** 2
                        - Uc_boundary / Kt + Us_now / Kt) * ALPHA * t * Time_rate + self.current_T  # test

            # second upper layer  # test
            X_delta_2 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_2 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_2 = (self.current_T - self.previous_T) * self.current_exist + (self.previous_T - self.body)  # test
            T_next_2 = (- X_delta_2 / DELTA_X ** 2 - Y_delta_2 / DELTA_Y ** 2 - Z_delta_2 / DELTA_Z ** 2
                        - Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test

            # diffuse to body - third upper layer
            T_nest_body = ((self.previous_T - self.body) / DELTA_Z ** 2 - Uc_boundary_ / Kt) * ALPHA * t * Time_rate

        return T_next_1, T_next_2, T_nest_body, Heat_matrix_current

    # Track-wise Temperature Update
    def Step(self, P, V, loc):
        # Update Temperature
        T_next_1, T_next_2, T_body, Heat_matrix_current = self.Diffusion(P, V, loc)
        self.current_T = T_next_1.copy()
        self.previous_T = T_next_2.copy()
        self.body = T_body.copy()

        # update the active cells
        self.current_exist = self.current_exist + Heat_matrix_current
        self.current_exist[self.current_exist != 0] = 1

    # Layer-wise Temperature Update
    def reset(self):
        self.body = np.average(self.previous_T.copy())
        """ requires to check - 如何进行迭代"""
        self.previous_T = np.average(self.current_T.copy())
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

    # Calculate temperature in 3 stripes
    def Cost_function(self, loc):

        current_T = self.current_T * self.current_exist
        if loc[0] >= 3:
            heat_dis = current_T[:, loc[0] - 3 * INTERVAL_Y:loc[0]]
        else:
            heat_dis = current_T[:, :loc[0]]
        Average_T = np.average(self.current_T)
        cost = np.sum((heat_dis - Average_T) / Tm ** 2)

        return cost
