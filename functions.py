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

    # input heat
    def Heat_matrix(self, P, loc):
        Q = 2 * LAMDA * P * exp(-2 * rb ^ 2 / Rb ^ 2) / (pi * Rb ^ 2)
        heat_matrix_temp = np.zeros((CELL_SIZE, CELL_SIZE))
        for i in range(len(self.heater_shape[0])):
            for j in range(len(self.heater_shape)):
                temp_x, temp_y = loc[0] - 1, loc[1] - 1
                if temp_x < 0 or temp_x > CELL_SIZE or temp_y < 0 or temp_y > CELL_SIZE:
                    break  # break is good?
                else:
                    heat_matrix_temp[temp_x][temp_y] = self.heater_shape[i][j]
        heat_matrix_ = heat_matrix_temp * Q
        Us_matrix = heat_matrix_ / Kt
        return Us_matrix

    # upper layer convent to air
    def Convention_matrix(self):
        T_temp = self.current_T * self.current_exist
        U_conv_temp = h * (T_temp - Ta) / DELTA_Z
        Uconv_matrix = U_conv_temp * self.current_exist
        return Uconv_matrix

    # diffusion into material
    def Diffusion(self):
        # diffuse to air

        # diffuse to material

        return 0

    def Update_step(self, loc):
        self.current_exist[loc[0]][loc[1]] = 1

    def average_T(self):
        return np.average(self.current_T)
