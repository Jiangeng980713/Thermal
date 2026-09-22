from math import *
from Parameter import *
import numpy as np
import matplotlib.pyplot as plt


class Thermal():
    def __init__(self):

        # # init matrix
        # self.current_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        # self.previous_T = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        # self.body = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        # self.previous_T_2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        #
        # # middle layer
        # self.temp1 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta
        # self.temp2 = np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta

        # init matrix
        self.Total_layer_num = Virtual_Layer * Available_LAYER_Num
        self.Depth_per_layer = Virtual_Layer
        self.T_slice = [np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta for _ in range(self.Total_layer_num)]    # inti artificial layer
        self.T_slice_next = [np.ones((CELL_SIZE_X, CELL_SIZE_Y)) * Ta for _ in range(self.Total_layer_num)]    # inti artificial layer
        self.body = None

        # in-process location
        self.Actuator = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))  # requires update

        # boundary_ for the finished layers
        # self.boundary_ = np.zeros((CELL_SIZE_X, CELL_SIZE_Y))
        # self.boundary_[0, :] = 1
        # self.boundary_[CELL_SIZE_X - 1, :] = 1
        # self.boundary_[:, 0] = 1
        # self.boundary_[:, CELL_SIZE_Y - 1] = 1

        # heater information
        self.row, self.column = HEATER_ROW, HEATER_COLUMN
        self.heater_depth = Heater_DEPTH  # TODO：
        self.heater_3d = self.create_3d_gaussian(self.row, self.column, self.heater_depth)

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

    def Reset_start(self):  # TODO

        for T in self.T_slice:
            T.fill(Ta)

        for T in self.T_slice_next:
            T.fill(Ta)

        self.body = Ta
        self.Actuator.fill(0)

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

    ###############################################  加入了三维高斯分布 #############################################
    def gaussian_3d(self, x, y, z, mu, sigma):
        diff = np.array([x, y, z]) - mu
        exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)
        denominator = np.sqrt((2 * np.pi) ** 3 * np.linalg.det(sigma))
        return np.exp(exponent) / denominator

    def create_3d_gaussian(self, rows, cols, depth):    # TODO: REQUIRE A TEST

        # 均值向量和协方差矩阵
        mu = np.array([0, 0, 0])  # 均值向量
        sigma = np.array([[SIGMA_1 ** 2, 0, 0],
                          [0, SIGMA_1 ** 2, 0],
                          [0, 0, SIGMA_2 ** 2]])  # 协方差矩阵

        x = np.arange(rows) - (rows - 1) / 2
        y = np.arange(cols) - (cols - 1) / 2
        z = np.linspace(0, depth - 1, depth)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        values = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    values[i, j, k] = self.gaussian_3d(X[i, j, k], Y[i, j, k], Z[i, j, k], mu, sigma)

        values = values / np.max(values)
        values = values[:, :, ::-1]  # TODO: A TEST HERE 测试一下 index， 看看index的顺序对不对

        return values

    def Heat_matrix_3D(self, P, loc):   # TODO: REQUIRE A TEST

        Q = 2 * LAMDA * P / (np.pi * Rb ** 2)
        Q = Q / (SIMU_H / LAYER_HEIGHT)

        heat_matrix_layers = [np.zeros((CELL_SIZE_X, CELL_SIZE_Y)) for _ in range(self.heater_depth)]

        x0 = loc[0] - self.row // 2
        y0 = loc[1] - self.column // 2

        x1 = x0 + self.row
        y1 = y0 + self.column

        if x0 >= 0 and x1 <= CELL_SIZE_X and y0 >= 0 and y1 <= CELL_SIZE_Y:
            for k in range(self.heater_depth):
                heat_matrix_layers[k][x0:x1, y0:y1] = self.heater_3d[:, :, self.heater_depth - 1 - k] * Q     # TODO: A TEST HERE 测试一下 index，看看index的顺序对不对

        return heat_matrix_layers

    def Step(self, P, V, loc, heater_actuated, right_bound):

        Time_rate = V / self.Vs

        """ Time_rate don't cause divergence, TIME_SCALE harm!"""
        # 收敛不收敛看时间跨度，不看迭代多少次
        # 因为 t / TIME_SCALE 是一种时间划分方式，TIME_RATE 是重复多少次传递，并不改变传导等物理过程的时间差分跨度
        # 相当于 for i 传递差分， for j 重复 i 的循环
        for i in range(int(TIME_SCALE * Time_rate)):

            # heat convection and radiation to air
            T_top_1 = self.T_slice[0]
            T_top_2 = self.T_slice[self.Depth_per_layer]

            # first physical layer - exposed fabricated region
            U_conv_1 = -h * (T_top_1 - Ta) / DELTA_Z * self.Actuator
            U_rad_1 = -EPSILON * SIGMA * (T_top_1 ** 4 - Ta ** 4) / DELTA_Z * self.Actuator

            # second physical layer - uncovered region
            U_conv_2 = -h * (T_top_2 - Ta) / DELTA_Z * (1 - self.Actuator)
            U_rad_2 = -EPSILON * SIGMA * (T_top_2 ** 4 - Ta ** 4) / DELTA_Z * (1 - self.Actuator)

            # 三维高斯熔池模型
            # heater is actuated or wait for the heater transform into next layer
            if heater_actuated:
                Us_input_now = self.Heat_matrix_3D(P, loc)
            else:
                Us_input_now = [np.zeros((CELL_SIZE_X, CELL_SIZE_Y)) for _ in range(self.heater_depth)]

            """ boundary condition """    # TODO: 重新引入边界条件
            # boundary = self.Check_boundary(loc)

            # first layer - boundary convention
            # Uc_boundary = h * (self.current_T * boundary - Ta) / DELTA_X

            # second layer - boundary convention
            # Uc_boundary_ = h * (self.previous_T * self.boundary_ - Ta) / DELTA_X

            """ zeros the boundary matrix"""
            # Uc_boundary = Uc_boundary_ = 0

            # mute the thermal conduction between material and air - an artificial boundary   # TODO：没有进行索引
            A = self.Actuator
            A_up = np.vstack((np.zeros((1, CELL_SIZE_Y)), A[:-1, :]))
            A_down = np.vstack((A[1:, :], np.zeros((1, CELL_SIZE_Y))))
            A_left = np.hstack((np.zeros((CELL_SIZE_X, 1)), A[:, :-1]))
            A_right = np.hstack((A[:, 1:], np.zeros((CELL_SIZE_X, 1))))

            # 第一层进行温度传导
            for k in range(self.Depth_per_layer):

                T_1 = self.T_slice[k]    # 索引当前 layer 编号下的温度场

                # X-direction diffusion
                X_delta_1 = X_TRANS * ((self.T_upper @ T_1) * self.Actuator * A_up + (self.T_lower @ T_1) * self.Actuator * A_down) / DELTA_X ** 2
                # Y-direction diffusion
                Y_delta_1 = Y_TRANS * ((T_1 @ self.T_left) * self.Actuator * A_left + (T_1 @ self.T_right) * self.Actuator * A_right) / DELTA_Y ** 2
                # Z-direction diffusion
                if k == 0:     # 最上面的一层，考虑到温度的耗散
                    Z_delta_1 = Z_TRANS * ((self.T_slice[k + 1] - T_1) * self.Actuator) / DELTA_Z ** 2
                else:
                    Z_delta_1 = Z_TRANS * (((self.T_slice[k - 1] - T_1) + (self.T_slice[k + 1] - T_1)) * self.Actuator) / DELTA_Z ** 2

                # temperature update
                self.T_slice_next[k] = (X_delta_1 + Y_delta_1 + Z_delta_1 + Us_input_now[k] / Kt) * ALPHA_T * (t / TIME_SCALE) + T_1

            # 第二物理层进行温度传导
            heat_depth_previous = max(0, self.heater_depth - self.Depth_per_layer)
            for k in range(self.Depth_per_layer):

                Layer_2_id = self.Depth_per_layer + k
                T_2 = self.T_slice[Layer_2_id]

                # X-direction diffusion
                X_delta_2 = X_TRANS * ((self.T_upper @ T_2) + (self.T_lower @ T_2)) / DELTA_X ** 2
                # Y-direction diffusion
                Y_delta_2 = Y_TRANS * ((T_2 @ self.T_left) + (T_2 @ self.T_right)) / DELTA_Y ** 2
                # Z-direction diffusion
                if k == self.Depth_per_layer - 1:        # 到达最底下的一层，考虑与底边的传导
                    Z_delta_2 = Z_TRANS * ((self.T_slice[Layer_2_id - 1] - T_2) + (self.body - T_2)) / DELTA_Z ** 2
                else:
                    Z_delta_2 = Z_TRANS * ((self.T_slice[Layer_2_id - 1] - T_2) + (self.T_slice[Layer_2_id + 1] - T_2)) / DELTA_Z ** 2

                # heat input: only the upper part within melt-pool penetration depth receives direct heat
                if k < heat_depth_previous:
                    Us_2 = Us_input_now[self.Depth_per_layer + k]
                else:
                    Us_2 = 0

                # temperature update
                self.T_slice_next[Layer_2_id] = (X_delta_2 + Y_delta_2 + Z_delta_2 + Us_2 / Kt) * ALPHA_T * (t / TIME_SCALE) + T_2

            # update the temperature matrix
            for k in range(self.Total_layer_num):
                self.T_slice[k][:] = self.T_slice_next[k]

    # Layer-wise Temperature Update
    def reset(self):

        # update body temperature
        self.body = np.average(self.T_slice[2 * self.Depth_per_layer - 1])
        self.body = self.body + BODY_OFFSET

        # current physical layer -> previous physical layer
        for k in range(self.Depth_per_layer):
            self.T_slice[k + self.Depth_per_layer][:] = self.T_slice[k]

        # create a new current physical layer
        for k in range(self.Depth_per_layer):
            self.T_slice[k].fill(Ta)

        # reset T_slice_next matrix
        for T in self.T_slice_next:
            T.fill(Ta)

        # reset actuator
        self.Actuator.fill(0)