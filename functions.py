from math import *
import numpy as np
from parameter import *


class Thermal():
    def __init__(self, scan_pattern):
        # reset
        self.heat_loc = (0, 0, 0)
        self.previous_T = np.zeros((CELL_SIZE, CELL_SIZE))
        self.current_T = np.zeros((CELL_SIZE, CELL_SIZE))
        self.current_exist = np.zeros((CELL_SIZE, CELL_SIZE))
        self.body = np.zeros((CELL_SIZE, CELL_SIZE))
        # self.boundary = np.zeros((CELL_SIZE, CELL_SIZE))

        # parameter
        self.rate = 3
        # heat-matrix range
        self.heater_radius = 1
        # diffusion-matrix range
        self.diffusion_radius = self.rate * self.heater_radius  # d = 3a
        self.heater_shape = np.zeros((self.heater_radius, self.heater_radius))
        self.Gaussian_heat()

        # layer
        self.pointer = [0, 0]
        self.scan_pattern = scan_pattern

        # build matrix
        self.diag_matrix = np.diag(np.ones(CELL_SIZE))
        self.ones = np.ones((1,CELL_SIZE))
        self.zeros = np.zeros((1, CELL_SIZE))

        # upper transactions matrix # C @ (A-B ) @ T = ^T
        self.A_upper = np.vstack((self.diag_matrix,self.zeros))
        self.B_upper = np.vstack((self.zeros,self.diag_matrix))
        self.B_upper[0][0] = 1 # 测试
        self.C_upper = np.hstack((self.diag_matrix,self.zeros.T))
        self.T_upper = self.C_upper@(self.A_upper-self.B_upper)

        # lower transactions matrix # C @ (A-B ) @ T = ^T
        self.A_lower = np.vstack((self.zeros,self.diag_matrix))
        self.B_lower = np.vstack((self.diag_matrix,self.zeros))
        self.B_lower[CELL_SIZE][CELL_SIZE-1] = 1 # 测试一下
        self.C_lower = np.hstack((self.zeros.T, self.diag_matrix))
        self.T_lower = self.C_lower@(self.A_lower - self.B_lower)

        # right transactions matrix  # T @ (A-B) @ C = ^T
        self.A_right = np.hstack((self.zeros.T, self.diag_matrix))
        self.B_right = np.hstack((self.diag_matrix, self.zeros.T))
        self.B_right[CELL_SIZE-1][CELL_SIZE] = 1  # 测试一下
        self.C_right = np.vstack((self.zeros, self.diag_matrix))
        self.T_right = (self.A_right - self.B_right)@self.C_right

        # left transactions matrix  # T @ (A-B) @ C = ^T
        self.A_left = np.hstack((self.diag_matrix,self.zeros.T))
        self.B_left = np.hstack((self.zeros.T,self.diag_matrix))
        self.B_left[0][0] = 1 # 测试
        self.C_left = np.hstack((self.diag_matrix,self.zeros))
        self.T_left = (self.A_left-self.B_left)@self.C_left

    def Gaussian_heat(self):
        # heat_matrix = np.zeros((self.reaction_radius*2-1,self.reaction_radius*2-1))
        # heat_matrix[self.reaction_radius][self.reaction_radius] = 1
        self.heater_shape = np.array([[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]])

    # 转化成为矩阵形式，推导一个矩阵形势出来
    def Heat_matrix(self, P, loc):
        Q = 2 * LAMDA * P * exp(-2 * rb ^ 2 / Rb ^ 2) / (pi * Rb ^ 2)
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

    def Convention_matrix(self):

        # 当前温度挥发
        U_conv_temp = h * (self.current_T - Ta) / DELTA_Z
        Uconv_matrix = - U_conv_temp * self.current_exist

        # 前一层未被覆盖部分的温度挥发
        U_conv_temp_ = h * (self.previous_T - Ta) / DELTA_Z
        Uconv_matrix_ = - U_conv_temp_ * (np.ones((CELL_SIZE, CELL_SIZE)) - self.current_exist)

        # 向空气中传导的边界条件（当前没有考虑在内）
        # U_conv_temp_boundary = h * (self.current_T - Ta) / DELTA_Z
        # Uconv_matrix = U_conv_temp * self.boundary

        return Uconv_matrix, Uconv_matrix_

    def Diffusion(self, P, loc):

        Uconv_now, Uconv_previous = self.Convention_matrix()
        Us_now = Uconv_now + self.Heat_matrix(P, loc)


        T_next = [[2 * self.current_T - self.current_T *


                   ] / DELTA_X ^ 2 + \
                  [2 * self.current_T - self.current_T * (front + behind)] / DELTA_Y ^ 2 + \
                  [self.current_T * (1 - down)] + Uconv_now / Kt] * ALPHA * t




        # second layer
        T_second = self.previous_T

    def Update_step(self, loc):
        self.current_exist[loc[0]][loc[1]] = 1
        # loc = loc # 完善loc的移动

    def average_T(self):
        # 无法确定average_T的温度，那么应该如何确定变化
        # 1。 通过实验进行验证，看一下体温度是如何进行上升的，不是不太受温度的影响，需要进行一定的后期标定
        # 2。 如果不行的话就采用经验常数，通过一个拟合项来表达当前的数值变化，和温度一样，基于经验的标定数量
        self.body = np.average(self.current_T)
        return np.average(self.current_T)
