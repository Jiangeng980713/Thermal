from math import *
import numpy as np
from parameter import *


class Thermal():
    def __init__(self, scan_pattern):
        # reset
        self.heat_loc = (0, 0, 0)
        self.previous_layer = np.zeros((CELL_SIZE, CELL_SIZE))
        self.current_layer = np.zeros((CELL_SIZE, CELL_SIZE))
        self.current_exist = np.zeros((CELL_SIZE, CELL_SIZE))
        # parameter
        self.rate = 3
        self.heater_radius = 1
        self.diffusion_radius = self.rate * self.heater_radius  # d = 3a
        self.heat_matrix = np.zeros((self.heater_radius, self.heater_radius))
        self.Gaussian_heat()
        # layer
        self.pointer = [0, 0]
        self.scan_pattern = scan_pattern

    def Gaussian_heat(self):
        # heat_matrix = np.zeros((self.reaction_radius*2-1,self.reaction_radius*2-1))
        # heat_matrix[self.reaction_radius][self.reaction_radius] = 1
        self.heat_matrix = np.array([[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]])

    # input heat
    def Heat_matrix(self, P, pointer):
        Q = 2 * LAMDA * P * exp(-2 * rb ^ 2 / Rb ^ 2) / (pi * Rb ^ 2)
        heat_matrix_ = self.heat_matrix * Q
        Us = Q / Kt
        return Us

    # convention to air
    # 只考虑当前 layer，没有上一层的散热
    def Convention_matrix(self):
        T_temp = self.current_layer * self.current_exist
        U_conv_temp = h * (T_temp - Ta) / DELTA_Z
        U_conv_temp = U_conv_temp * self.current_exist
        return U_conv_temp

    # diffusion into material
    def Diffusion(self):
        self.current_layer =




        return 0

