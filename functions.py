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

    def Gaussian_heat(self):
        # heat_matrix = np.zeros((self.reaction_radius*2-1,self.reaction_radius*2-1))
        # heat_matrix[self.reaction_radius][self.reaction_radius] = 1
        self.heater_shape = np.array([[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]])

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

    # upper layer convent to air
    def Convention_matrix(self):
        U_conv_temp = h * (self.current_T - Ta) / DELTA_Z
        Uconv_matrix = - U_conv_temp * self.current_exist

        # 向空气中传导的边界条件（当前没有考虑在内）
        # U_conv_temp_boundary = h * (self.current_T - Ta) / DELTA_Z
        # Uconv_matrix = U_conv_temp * self.boundary

        return Uconv_matrix

    # diffusion into material
    def Diffusion(self, P, loc):

        # diffuse to material
        U_total = self.Heat_matrix(P, loc) + self.Convention_matrix()






    def Update_step(self, loc):
        self.current_exist[loc[0]][loc[1]] = 1
        # loc = loc # 完善loc的移动

    def average_T(self):
        # 无法确定average_T的温度，那么应该如何确定变化
        # 1。 通过实验进行验证，看一下体温度是如何进行上升的，不是不太受温度的影响，需要进行一定的后期标定
        # 2。 如果不行的话就采用经验常数，通过一个拟合项来表达当前的数值变化，和温度一样，基于经验的标定数量
        self.body = np.average(self.current_T)
        return np.average(self.current_T)
