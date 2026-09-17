import numpy as np
import multiprocessing
import time


class Particle:
    def __init__(self, particle_id, x_bound, v_bound):
        self.id = particle_id
        self.position = np.random.uniform(x_bound[0], x_bound[1])
        self.velocity = np.random.uniform(v_bound[0], v_bound[1])
        self.best_position = self.position
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand()
        r2 = np.random.rand()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, x_bound):
        self.position += self.velocity
        if self.position < x_bound[0]:
            self.position = x_bound[0]
        elif self.position > x_bound[1]:
            self.position = x_bound[1]


def fitness_function(x):
    time.sleep(0.1)  # 模拟耗时计算
    return x ** 2  # 简单的二次函数，目标是求其最小值


def evaluate_particle(particle):
    fitness = fitness_function(particle.position)
    return fitness, particle.id


def pso(x_bound, v_bound, num_particles, max_iter):
    particles = [Particle(i, x_bound, v_bound) for i in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    w = 0.5  # 惯性权重
    c1 = 1.5  # 认知参数
    c2 = 1.5  # 社会参数

    for episode in range(max_iter):
        with multiprocessing.Pool(processes=12) as pool:  # 12个进程的进程池
            results = pool.map(evaluate_particle, particles)  # 并行评估粒子
            print('results', results)

        for fitness, particle_id in results:
            particle = particles[particle_id]
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(x_bound)
        print('episode', episode)
    return global_best_position, global_best_fitness


if __name__ == "__main__":
    x_bound = [-10, 10]  # 粒子位置的范围
    v_bound = [-1, 1]  # 粒子速度的范围
    num_particles = 30  # 粒子数量
    max_iter = 100  # 最大迭代次数

    best_position, best_fitness = pso(x_bound, v_bound, num_particles, max_iter)
    print(f'Best position: {best_position}')
    print(f'Best fitness: {best_fitness}')
