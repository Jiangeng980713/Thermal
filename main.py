import random
from worker import *


def start():

    # parameter
    w = 0.8
    c1 = 2
    c2 = 2
    r1 = 0.6
    r2 = 0.3
    pN = 12  # 粒子数量
    DIM = 30  # Dimension of states
    max_iter = 100  # 迭代次数

    # current value
    Location = np.zeros((pN, DIM))  # 所有粒子的位置和速度
    Velocity = np.zeros((pN, DIM))

    # best value
    pbest = np.zeros((pN, DIM))  # 个体经历的最佳位置和全局最佳位置
    gbest = np.zeros((1, DIM))

    # cost function
    p_fit = np.zeros(pN)  # 每个个体的历史最佳适应值
    fit = 1e10  # 全局最佳适应值

    # init Location & Velocity
    for p_i in range(pN):
        for dim in range(DIM):
            Location[p_i][dim] = random.uniform(0, 1)
            Velocity[p_i][dim] = random.uniform(0, 1)
        pbest[p_i] = Location[p_i]
        tmp = 1e10  # 加和 cost-function 输出
        p_fit[p_i] = tmp
        if tmp < fit:
            fit = tmp
            gbest = Location[p_i]

    fitness = []

    for episode in range(max_iter):
        for i in range(pN):  # 更新gbest\pbest

            temp = (Location[i])  # 进行并行运算的位置

            if temp < p_fit[i]:  # 更新个体最优
                p_fit[i] = temp  # 更新 cost function
                pbest[i] = Location[i]  # 把 i 粒子的内容给到 fit
                if p_fit[i] < fit:  # 更新全局最优
                    fit = p_fit[i]  # 把 i 粒子 cost function 给 fit
                    gbest = Location[i]  # 把 i 粒子的内容给到 fit

        # update pixel i Velocity and Location
        for i in range(pN):
            Velocity[i] = w * Velocity[i] + c1 * r1 * (pbest[i] - Location[i]) + c2 * r2 * (gbest - Location[i])
            Location[i] = Location[i] + Velocity[i]

        # record cost function
        fitness.append(fit)  # 一个step中最优的损失函数
        print(float(fit))  # 输出最优值

    return gbest, fitness


if __name__ == '__main__':
    Final_result, Cost_function = start()
    print(Final_result)
    np.save('result.npy', np.array(Final_result))
    np.save('cost_function.npy', np.array(Cost_function))
