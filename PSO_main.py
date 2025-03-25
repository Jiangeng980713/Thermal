import psutil
from worker import *
import multiprocessing


class Particle:
    def __init__(self, particle_id, dim, x_bound, v_bound, load, input_vector):

        self.id = particle_id

        if not load:
            # self.position = np.random.uniform(x_bound[0], x_bound[1], dim)  # 生成随机数 Location
            self.position = np.ones(dim) * P_START  # 针对当前的 P 进行优化
        else:
            self.position = input_vector
            assert len(self.position) == LAYER_HEIGHT * STRIPE_NUM, " V num do not match stripe num " + str(self.id)
        self.velocity = np.random.uniform(v_bound[0], v_bound[1], dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.v_bound = v_bound
        self.x_bound = x_bound

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

        """clip the velocity"""
        self.velocity = np.clip(self.velocity, self.v_bound[0], self.v_bound[1])

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.x_bound[0], self.x_bound[1])


# REACT WITH SIMULATED MODEL -> Worker_agent
def evaluate_particle(particle):
    fitness = worker_agent(particle.position)
    return fitness, particle.id


def pso(x_bound, v_bound, num_particles, max_iter):
    RAMDOM_START = True
    load = False

    # 是否是随机生成位置开局，是的话会在 particle 中生成随机 vector
    if RAMDOM_START:
        input_vector = [np.random.uniform(V_MIN, V_MAX, LAYER_HEIGHT * STRIPE_NUM)]
    else:
        input_vector = [np.random.uniform(V_MIN, V_MAX, LAYER_HEIGHT * STRIPE_NUM)]

    dim = LAYER_HEIGHT * STRIPE_NUM  # Position 对应的维度，就是优化项目的维度，V 有多少维度
    particles = [Particle(i, dim, x_bound, v_bound, load, input_vector) for i in range(num_particles)]
    global_best_position = np.random.uniform(x_bound[0], x_bound[1], dim)
    global_best_fitness = float('inf')

    # dynamic w parameter
    w_max = 0.9  # 初始惯性权重
    w_min = 0.4  # 最小惯性权重
    c1 = 1.5  # 认知参数
    c2 = 1.5  # 社会参数

    parent_process = psutil.Process()  # 获取当前进程

    # record
    global_costs = []

    for episode in range(max_iter):

        print("episode", episode)

        temp_state_x = []
        temp_state_v = []

        # dynamic w
        w = w_max - (w_max - w_min) * (episode / max_iter)

        # dynamic c1 & c2
        # c1 = 2.5 - (2.5 - 0.5) * (episode / max_iter)  # c1 从 2.5 线性减小到 0.5
        # c2 = 0.5 + (2.5 - 0.5) * (episode / max_iter)  # c2 从 0.5 线性增大到 2.5

        with multiprocessing.Pool(processes=THREAD_NUM) as pool:  # number of cpu threads
            results = pool.map(evaluate_particle, particles)  # parallel cost function

        current_children = parent_process.children()
        assert len(current_children) == 0, 'children process is not done'

        for fitness, particle_id in results:
            particle = particles[particle_id]
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

            # TODO：这个位置的 id 不一定是顺序的，就是说 particles 中的 particle 可能不是顺序
            # record the state
            x, v = particle.position, particle.velocity
            temp_state_x.append(x)
            temp_state_v.append(v)

        # record global fitness
        global_costs.append(global_best_fitness)

        # save the states
        np.save('temp_state_x', temp_state_x)
        np.save('temp_state_v', temp_state_v)

    return global_best_position, global_best_fitness, global_costs


if __name__ == "__main__":
    # 调节制造功率以实现热梯度的最低
    x_bound = [P_Min, P_Max]

    # 基于位置范围的比例确定速度范围
    alpha = 0.5  # 速度范围比例因子
    v_min = -alpha * (x_bound[1] - x_bound[0])
    v_max = alpha * (x_bound[1] - x_bound[0])
    v_bound = [v_min, v_max]

    # 粒子数量
    num_particles = PARTICLE_NUM
    # 最大迭代次数
    max_iter = EPISODE_NUM

    best_position, best_fitness, global_costs = pso(x_bound, v_bound, num_particles, max_iter)
    print(f'Best position: {best_position}')
    print(f'Best fitness: {best_fitness}')
    np.save('cost', global_costs)