import numpy as np
import random
import matplotlib.pyplot as plt


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.Location = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.Velocity = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    # # ---------------------目标函数Sphere函数-----------------------------
    # def function(self, x):
    #     sum = 0
    #     length = len(x)
    #     x = x ** 2
    #     for i in range(length):
    #         sum += x[i]
    #     return sum

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.Location[i][j] = random.uniform(0, 1)
                self.Velocity[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.Location[i]
            tmp = self.function(self.Location[i])   # 加和 cost-function 输出
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.Location[i]

    # ----------------------更新粒子位置----------------------------------

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.Location[i])      # 进行并行运算的位置
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp    # 更新 cost function
                    self.pbest[i] = self.Location[i]  # 把 i 粒子的内容给到 fit
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.fit = self.p_fit[i]  # 把 i 粒子 cost function 给 fit
                        self.gbest = self.Location[i]    # 把 i 粒子的内容给到 fit

            # update agent i Velocity and Location
            for i in range(self.pN):
                self.Velocity[i] = self.w * self.Velocity[i] + self.c1 * self.r1 * (self.pbest[i] - self.Location[i]) + self.c2 * self.r2 * (self.gbest - self.Location[i])
                self.Location[i] = self.Location[i] + self.Velocity[i]
            fitness.append(self.fit)        # 一个step中最优的损失函数
            print(float(self.fit))          # 输出最优值
        return fitness

    # ----------------------程序执行-----------------------


my_pso = PSO(pN=30, dim=5, max_iter=100)
my_pso.init_Population()
fitness = my_pso.iterator()

# # -------------------画图--------------------
# plt.figure(1)
# plt.title("Figure1")
# plt.xlabel("iterators", size=14)
# plt.ylabel("fitness", size=14)
# t = np.array([t for t in range(0, 100)])
# fitness = np.array(fitness)
# plt.plot(t, fitness, color='b', linewidth=3)
# plt.show()