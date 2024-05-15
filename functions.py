from math import *
import numpy as np
from parameter import *


class Thermal():
    def __init__(self):

        # init matrix
        self.current_T = np.zeros((CELL_SIZE, CELL_SIZE))
        self.previous_T = np.zeros((CELL_SIZE, CELL_SIZE))
        self.body = np.zeros((CELL_SIZE, CELL_SIZE))

        # in-process location
        self.current_exist = np.zeros((CELL_SIZE, CELL_SIZE))  # requires update

        # boundary_ for the finished layers
        self.boundary_ = np.zeros((CELL_SIZE, CELL_SIZE))
        self.boundary_[0, :] = 1
        self.boundary_[CELL_SIZE - 1, :] = 1
        self.boundary_[:, 0] = 1
        self.boundary_[:, CELL_SIZE - 1] = 1

        """ Heater Info """
        self.rate = 3
        self.heater_radius = 1
        self.diffusion_radius = self.rate * self.heater_radius  # d = 3a
        self.heater_shape = np.zeros((self.heater_radius, self.heater_radius))
        self.Gaussian_heat()

        """ Diffusion Transaction Matrix """
        # build matrix
        self.diag_matrix = np.diag(np.ones(CELL_SIZE))
        self.ones = np.ones((1, CELL_SIZE))
        self.zeros = np.zeros((1, CELL_SIZE))

        # upper transactions matrix # C @ (A-B) @ T = ^T  # checked
        self.A_upper = np.vstack((self.diag_matrix, self.zeros))
        self.B_upper = np.vstack((self.zeros, self.diag_matrix))
        self.B_upper[0][0] = 1
        self.C_upper = np.hstack((self.diag_matrix, self.zeros.T))
        self.T_upper = self.C_upper @ (self.A_upper - self.B_upper)

        # lower transactions matrix # C @ (A-B) @ T = ^T  # checked
        self.A_lower = np.vstack((self.zeros, self.diag_matrix))
        self.B_lower = np.vstack((self.diag_matrix, self.zeros))
        self.B_lower[CELL_SIZE][CELL_SIZE - 1] = 1
        self.C_lower = np.hstack((self.zeros.T, self.diag_matrix))
        self.T_lower = self.C_lower @ (self.A_lower - self.B_lower)

        # right transactions matrix  # T @ (A-B) @ C = ^T
        self.A_right = np.hstack((self.zeros.T, self.diag_matrix))
        self.B_right = np.hstack((self.diag_matrix, self.zeros.T))
        self.B_right[CELL_SIZE - 1][CELL_SIZE] = 1  # 测试一下
        self.C_right = np.vstack((self.zeros, self.diag_matrix))
        self.T_right = (self.A_right - self.B_right) @ self.C_right

        # left transactions matrix  # T @ (A-B) @ C = ^T
        self.A_left = np.hstack((self.diag_matrix, self.zeros.T))
        self.B_left = np.hstack((self.zeros.T, self.diag_matrix))
        self.B_left[0][0] = 1  # 测试
        self.C_left = np.vstack((self.diag_matrix, self.zeros))
        self.T_left = (self.A_left - self.B_left) @ self.C_left

        # layer-wise velocity
        """ test for determine"""
        self.Vs = 200

    def reset(self):
        self.current_T = np.zeros((CELL_SIZE, CELL_SIZE))
        self.previous_T = np.ones((CELL_SIZE, CELL_SIZE)) * Ta
        self.body = np.ones((CELL_SIZE, CELL_SIZE)) * Ta
        self.current_exist = np.zeros((CELL_SIZE, CELL_SIZE))  # requires update

    def check_boundary(self, loc):

        temp = np.zeros((CELL_SIZE, CELL_SIZE))

        # in the middle
        if 0 < loc[1] < CELL_SIZE - 1:
            temp[:, 0] = 1  # left
            temp[0, :loc[1] + 1] = 1  # upper
            temp[CELL_SIZE - 1, :loc[1] + 1] = 1  # lower
            temp[:, loc[1]] = 1  # right
            temp[loc[0]:, loc[1]] = 0  # remove right
            temp[loc[0]:, loc[1] - 1] = 1  # add secondary right

        # the left boundary
        if loc[1] == 0:
            temp[:loc[0], 0] = 1

        # the right boundary
        if loc[1] == CELL_SIZE - 1:
            temp[:, 0] = 1  # left
            temp[0, :loc[1] + 1] = 1  # upper
            temp[CELL_SIZE - 1, :CELL_SIZE - 2] = 1  # down
            temp[:loc[0], loc[1]] = 1  # right-1
            temp[loc[0]:, loc[1] - 1] = 1  # right-2

        return temp

    " 采用什么热源进行加热可以有效反应温度"
    def Gaussian_heat(self):
        # heat_matrix = np.zeros((self.reaction_radius*2-1,self.reaction_radius*2-1))
        # heat_matrix[self.reaction_radius][self.reaction_radius] = 1
        self.heater_shape = np.array([[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]])

    def Heat_matrix(self, P, loc):
        Q = 2 * LAMDA * P * exp(-2 * rb ** 2 / Rb ** 2) / (pi * Rb ** 2)
        heat_matrix_temp = np.zeros((CELL_SIZE, CELL_SIZE))
        # 等效热源，3x3加到loc周围的点上，如果3x3内没有cell，则认为没有被加热
        for i in range(len(self.heater_shape[0])):
            for j in range(len(self.heater_shape)):
                temp_x, temp_y = loc[0] - 1, loc[1] - 1
                if temp_x < 0 or temp_x > CELL_SIZE or temp_y < 0 or temp_y > CELL_SIZE:
                    continue
                else:
                    heat_matrix_temp[temp_x][temp_y] = self.heater_shape[i][j]
        heat_matrix_ = heat_matrix_temp * Q
        return heat_matrix_

    def Convention_matrix(self, loc):

        # check boundary
        boundary = self.check_boundary(loc)

        # convention from the first layer
        Uc_temp = h * (self.current_T - Ta) / DELTA_Z
        Uc_matrix = - Uc_temp * self.current_exist  # 是否应该有"-"号

        # convention from the uncovered second layer
        Uc_temp_ = h * (self.previous_T - Ta) / DELTA_Z
        Uc_matrix_ = - Uc_temp_ * (np.ones((CELL_SIZE, CELL_SIZE)) - self.current_exist)  # 是否应该有"-"号

        # boundary convention in the first layer
        Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

        # boundary convention in the second layer
        Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

        return Uc_matrix, Uc_matrix_, Uc_boundary, Uc_boundary_

    """Problem: 1. V 没有统一 3. Body的温度应该如何计算"""
    def Diffusion(self, P, V, loc):

        Uconv_now, Uconv_previous, Uc_boundary, Uc_boundary_ = self.Convention_matrix(loc)
        Us_now = Uconv_now + self.Heat_matrix(P, loc)
        Time_rate = V / self.Vs

        # layer number = 1
        if loc[2]==0:
            X_delta_1 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_1 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_1 = (self.current_T - self.previous_T) * self.current_exist  # test
            T_next_1 = (- X_delta_1 / DELTA_X ** 2 - Y_delta_1 / DELTA_Y ** 2 - Z_delta_1 / DELTA_Z ** 2
                        - Uc_boundary / Kt + Us_now / Kt) * ALPHA * t * self.Vs + self.current_T  # test

            X_delta_2 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_2 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_2 = (self.current_T - self.previous_T) * self.current_exist + (self.previous_T - self.body)  # test
            T_next_2 = (- X_delta_2 / DELTA_X ** 2 - Y_delta_2 / DELTA_Y ** 2 - Z_delta_2 / DELTA_Z ** 2
                        - Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test

            T_nest_body = T_next_2

        # layer number higher than 1
        if loc[2]>=1:
            # upper layer
            X_delta_1 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_1 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_1 = (self.current_T - self.previous_T) * self.current_exist  # test
            T_next_1 = (- X_delta_1 / DELTA_X ** 2 - Y_delta_1 / DELTA_Y ** 2 - Z_delta_1 / DELTA_Z ** 2
                        - Uc_boundary / Kt + Us_now / Kt) * ALPHA * t * self.Vs + self.current_T  # test

            # second upper layer  # test
            X_delta_2 = self.current_T @ (self.T_upper + self.T_lower) - 2 * self.current_T
            Y_delta_2 = (self.T_left + self.T_right) @ self.current_T - 2 * self.current_T
            Z_delta_2 = (self.current_T - self.previous_T) * self.current_exist + (self.previous_T - self.body)  # test
            T_next_2 = (- X_delta_2 / DELTA_X ** 2 - Y_delta_2 / DELTA_Y ** 2 - Z_delta_2 / DELTA_Z ** 2
                        - Uc_boundary / Kt) * ALPHA * t * Time_rate + self.current_T  # test

            # diffuse to body - third upper layer
            T_nest_body = ((self.previous_T - self.body) / DELTA_Z ** 2 - Uc_boundary_ / Kt) * ALPHA * t * Time_rate

        return T_next_1, T_next_2, T_nest_body

    # Track-wise Temperature Update
    def Step(self, P, V, loc):

        # Update Temperature
        T_next_1, T_next_2, T_body = self.Diffusion(P, V, loc)
        self.current_T = T_next_1.copy()
        self.previous_T = T_next_2.copy()
        self.body = T_body.copy()

    # Layer-wise Temperature Update
    def Episode_Update(self):
        self.body = np.average(self.previous_T.copy())
        self.previous_T = np.average(self.current_T.copy())
        self.current_T = np.zeros((CELL_SIZE, CELL_SIZE))
