from math import *
from Parameter import *
import numpy as np
import matplotlib.pyplot as plt


class Thermal():
    def __init__(self):

        # init matrix
        self.current_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T_2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta

        # middle layer
        self.temp1 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.temp2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta

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
        self.heater_2d = self.create_2d_gaussian(self.row, self.column)

        self.depth = MIDDLE_LAYER
        self.heater_3d = self.create_3d_gaussian(self.row, self.column, self.depth)

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

    def save_FIG(self, matrix, name):
        plt.imshow(matrix)
        plt.savefig("figure" + name)

    def transform_heat(self):
        return 0

    def Reset(self):
        self.current_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.previous_T_2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.Actuator = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

        # middle layer
        self.temp1 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        self.temp2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta

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
        Q = Q / (SIMU_H / LAYER_HEIGHT)  # volume heater

        heat_matrix_temp = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        heat_matrix_current = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # activate cell

        # the boundary location for heater on the plate
        boundary_x, boundary_y = loc[0] - self.row // 2, loc[1] - self.column // 2

        for i in range(len(self.heater_2d[0])):
            for j in range(len(self.heater_2d)):
                temp_x, temp_y = boundary_x + i, boundary_y + j
                # outer boundary skip the value
                if temp_x < 0 or temp_x >= CELL_SIZE_X or temp_y < 0 or temp_y >= CELL_SIZE_Y:
                    continue
                else:
                    heat_matrix_temp[temp_x][temp_y] = self.heater_2d[i][j]
                    heat_matrix_current[temp_x][temp_y] = 1  # activate cell

        heat_matrix_ = heat_matrix_temp * Q
        self.Actuator = self.Actuator + heat_matrix_current
        self.Actuator[self.Actuator != 0] = 1
        return heat_matrix_

    ###############################################  加入了三维高斯分布 #############################################
    def gaussian_3d(self, x, y, z, mu, sigma):
        diff = np.array([x, y, z]) - mu
        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)
        denominator = np.sqrt((2 * np.pi) ** 3 * np.linalg.det(sigma))
        return np.exp(exponent) / denominator

    def create_3d_gaussian(self, rows, cols, depth):

        # 均值向量和协方差矩阵
        mu = np.array([0, 0, 0])  # 均值向量
        sigma = np.array([[SIGMA_1 ** 2, 0, 0],
                          [0, SIGMA_1 ** 2, 0],
                          [0, 0, SIGMA_2 ** 2]])  # 协方差矩阵

        x = np.arange(rows) - (rows - 1) / 2
        y = np.arange(cols) - (cols - 1) / 2
        z = np.linspace(0, depth - 1, depth)
        X, Y, Z = np.meshgrid(x, y, z)

        # 计算每个网格点的概率密度值
        values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    values[i, j, k] = self.gaussian_3d(X[i, j, k], Y[i, j, k], Z[i, j, k], mu, sigma)

        # TODO: 对熔池进行尺寸一个重塑，大于多少的和小于多少的要弄一下
        values = values / np.max(values)

        # 最底层和最上层换一个位置，最大范围最热范围在最上面，最下面的凉和小
        values[:, :, [0, -1]] = values[:, :, [-1, 0]]

        return values

    def Heat_matrix_3D(self, P, loc):

        # the first layer heater
        Q = 2 * LAMDA * P / (pi * Rb ** 2)
        Q = Q / (SIMU_H / LAYER_HEIGHT)  # volume heater

        # activate cell on the top
        heat_matrix_temp = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        heat_matrix_current_1 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        heat_matrix_temp_2 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        heat_matrix_temp_3 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

        # the boundary location for heater on the plate
        boundary_x, boundary_y = loc[0] - self.row // 2, loc[1] - self.column // 2

        # 遍历所有单元格 x-y 方向
        for i in range(len(self.heater_3d[0])):  # 11
            for j in range(len(self.heater_3d[1])):  # 11

                temp_x, temp_y = boundary_x + i, boundary_y + j

                # outer boundary skip the value
                if temp_x < 0 or temp_x >= CELL_SIZE_X or temp_y < 0 or temp_y >= CELL_SIZE_Y:
                    continue
                else:
                    # first layer
                    heat_matrix_temp[temp_x][temp_y] = self.heater_3d[i][j][2]
                    heat_matrix_current_1[temp_x][temp_y] = 1  # activate cell

                    # second layer
                    heat_matrix_temp_2[temp_x][temp_y] = self.heater_3d[i][j][1]

                    # third layer
                    heat_matrix_temp_3[temp_x][temp_y] = self.heater_3d[i][j][0]

        # update the temperature in heater matrix (layer 1-3)
        heat_matrix_layer_1 = heat_matrix_temp * Q
        heat_matrix_layer_2 = heat_matrix_temp_2 * Q
        heat_matrix_layer_3 = heat_matrix_temp_3 * Q

        # update the actuator
        self.Actuator = self.Actuator + heat_matrix_current_1
        self.Actuator[self.Actuator != 0] = 1

        return heat_matrix_layer_1, heat_matrix_layer_2, heat_matrix_layer_3

    def Step(self, P, V, loc, heater_actuated):

        Time_rate = V / self.Vs

        """ Time_rate don't cause divergence, TIME_SCALE harm!"""
        # 收敛不收敛看时间跨度，不看迭代多少次
        # 因为 t / TIME_SCALE 是一种时间划分方式，TIME_RATE 是重复多少次传递，并不改变传导等物理过程的时间差分跨度
        # 相当于 for i 传递差分， for j 重复 i 的循环
        for i in range(int(TIME_SCALE * Time_rate)):

            # # 热-空气传导模型
            # first layer - convention to air
            Uc_temp = h * (self.current_T - Ta) / DELTA_Z
            Uconv_now = - Uc_temp * self.Actuator

            # second layer - convention from the uncovered
            Uc_temp_ = h * (self.previous_T - Ta) / DELTA_Z
            Uconv_previous = - Uc_temp_ * (np.ones((CELL_SIZE_X, CELL_SIZE_Y)) - self.Actuator)

            # # 二维高斯熔池模型
            # # heater is actuated or wait for the heater transform into next layer
            # if heater_actuated:
            #     Heat_matrix = self.Heat_matrix(P, loc)
            #     Us_now = Heat_matrix + Uconv_now
            # else:
            #     Us_now = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

            # 三维高斯熔池模型
            # heater is actuated or wait for the heater transform into next layer
            if heater_actuated:
                Heat_matrix_1, Us_now_2, Us_now_3 = self.Heat_matrix_3D(P, loc)
                Us_now_1 = Heat_matrix_1 + Uconv_now
            else:
                Us_now_1 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
                Us_now_2 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
                Us_now_3 = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))

            """ boundary condition """
            # boundary = self.Check_boundary(loc)

            # first layer - boundary convention
            # Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

            # second layer - boundary convention
            # Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

            """ zeros the boundary matrix"""
            # Uc_boundary = Uc_boundary_ = 0

            # # 两个半层
            # # temperature diffusion - first layer
            # X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
            # Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
            # Z_delta_1 = ((self.temp1 - self.current_T) * self.Actuator) / (DELTA_Z / 2) ** 2    # 仅仅有对中间层的交互
            # T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_now / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

            # # temp middle layer
            # Z_delta_temp_1 = (((self.previous_T - self.temp1) + (self.current_T - self.temp1)) * self.Actuator) / (DELTA_Z / 2) ** 2
            # T_next_temp_1 = Z_delta_temp_1 * ALPHA * (t / TIME_SCALE) + self.temp1

            # # temperature diffusion - second layer
            # X_delta_2 = ((self.T_upper + self.T_lower) @ self.previous_T) / DELTA_X ** 2
            # Y_delta_2 = (self.previous_T @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            # Z_delta_2 = ((self.temp1 - self.previous_T) * self.Actuator + (self.temp2 - self.previous_T)) / (DELTA_Z / 2) ** 2
            # T_next_2 = (X_delta_2 + Y_delta_2 + Z_delta_2 + Uconv_previous / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T

            # # temp middle layer
            # Z_delta_temp_2 = (((self.previous_T - self.temp2) + (self.body - self.temp2)) * self.Actuator) / (DELTA_Z / 2) ** 2
            # T_next_temp_2 = Z_delta_temp_2 * ALPHA * (t / TIME_SCALE) + self.temp2
            #
            # # update the temperature in one small cell
            # self.current_T = T_next_1.copy()
            # self.previous_T = T_next_2.copy()

            # # update the temp temperature
            # self.temp1 = T_next_temp_1.copy()
            # self.temp2 = T_next_temp_2.copy()

            #  三层模型
            # temperature diffusion - first layer
            X_delta_1 = X_TRANS * ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
            Y_delta_1 = Y_TRANS * (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
            Z_delta_1 = Z_TRANS * ((self.previous_T - self.current_T) * self.Actuator) / (DELTA_Z) ** 2
            T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_now_1 / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

            # temperature diffusion - second layer
            X_delta_2 = X_TRANS * ((self.T_upper + self.T_lower) @ self.previous_T) / DELTA_X ** 2
            Y_delta_2 = Y_TRANS * (self.previous_T @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            Z_delta_2 = Z_TRANS * ((self.current_T - self.previous_T) * self.Actuator + (self.previous_T_2 - self.previous_T)) / (DELTA_Z) ** 2
            T_next_2 = (X_delta_2 + Y_delta_2 + Z_delta_2 + (Uconv_previous + Us_now_2) / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T

            # add a third layer
            X_delta_3 = X_TRANS * ((self.T_upper + self.T_lower) @ self.previous_T_2) / DELTA_X ** 2
            Y_delta_3 = Y_TRANS * (self.previous_T_2 @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            Z_delta_3 = Z_TRANS * ((self.previous_T - self.previous_T_2) + (self.body - self.previous_T_2)) / (DELTA_Z) ** 2
            T_next_3 = (X_delta_3 + Y_delta_3 + Z_delta_3 + Us_now_3 / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T_2

            # update the temperature in one small cell / body 温度不变
            self.current_T = T_next_1.copy()
            self.previous_T = T_next_2.copy()
            self.previous_T_2 = T_next_3.copy()

            # # 原始 - body 固定参数
            # # temperature diffusion - first layer
            # X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
            # Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
            # Z_delta_1 = ((self.previous_T - self.current_T) * self.Actuator) / (DELTA_Z) ** 2
            # T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_now / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

            # # temperature diffusion - second layer
            # X_delta_2 = ((self.T_upper + self.T_lower) @ self.previous_T) / DELTA_X ** 2
            # Y_delta_2 = (self.previous_T @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            # Z_delta_2 = ((self.current_T - self.previous_T) * self.Actuator + (self.body - self.previous_T)) / (DELTA_Z) ** 2
            # T_next_2 = (X_delta_2 + Y_delta_2 + Z_delta_2 + Uconv_previous / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T

            # # update the temperature in one small cell
            # self.current_T = T_next_1.copy()
            # self.previous_T = T_next_2.copy()

            # ## body 与 previous相同
            # # temperature diffusion - first layer
            # X_delta_1 = ((self.T_upper + self.T_lower) @ self.current_T * self.Actuator) / DELTA_X ** 2
            # Y_delta_1 = (self.current_T @ (self.T_left + self.T_right) * self.Actuator) / DELTA_Y ** 2
            # Z_delta_1 = ((self.previous_T - self.current_T) * self.Actuator) / (DELTA_Z) ** 2
            # T_next_1 = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_now / Kt) * ALPHA * (t / TIME_SCALE) + self.current_T

            # # temperature diffusion - second layer
            # X_delta_2 = ((self.T_upper + self.T_lower) @ self.previous_T) / DELTA_X ** 2
            # Y_delta_2 = (self.previous_T @ (self.T_left + self.T_right)) / DELTA_Y ** 2
            # Z_delta_2 = ((self.current_T - self.previous_T) * self.Actuator + (self.body - self.previous_T)) / (DELTA_Z) ** 2
            # T_next_2 = (X_delta_2 + Y_delta_2 + Z_delta_2 + Uconv_previous / Kt) * ALPHA * (t / TIME_SCALE) + self.previous_T

            # # update the temperature in one small cell
            # self.current_T = T_next_1.copy()
            # self.previous_T = T_next_2.copy()
            # self.body = T_next_2.copy()

    # Layer-wise Temperature Update
    def reset(self):
        self.body = np.average(self.previous_T.copy())
        # 如果存在三维输入且三层传热
        self.previous_T_2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * np.average(self.previous_T.copy())
        self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * np.average(self.current_T.copy())
        self.current_T = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.Actuator = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        self.body = self.body + BODY_OFFSET