from worker import *
import multiprocessing
import time


class Particle:
    def __init__(self, particle_id, dim, x_bound, v_bound, load, input_vector):

        self.id = particle_id

        if not load:
            self.position = np.ones(dim) * P_START                     # 针对当前的 P 进行优化
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
    # print("particle" + str(particle.id) + "is done")
    return fitness, particle.id


def pso(x_bound, v_bound, num_particles, max_iter, save_path):

    load = False

    # 是否是随机生成位置开局，是的话会在 particle 中生成随机 vector
    if load:
        input_vector = np.load('input.npy')   # 换成需要导入的 input tensor
    else:
        input_vector = [np.random.uniform(P_Min, P_Max, LAYER_HEIGHT * STRIPE_NUM)]

    dim = LAYER_HEIGHT * STRIPE_NUM  # Position 对应的维度，就是优化项目的维度，V 有多少维度
    particles = [Particle(i, dim, x_bound, v_bound, load, input_vector) for i in range(num_particles)]

    # 生成一个字典 {particle_id: particle}，确保 ID 和粒子映射正确
    particle_dict = {particle.id: particle for particle in particles}

    global_best_position = np.random.uniform(x_bound[0], x_bound[1], dim)
    global_best_fitness = float('inf')

    # fixed w parameter
    w_max = W_MAX     # (惯性权重)
    w_min = W_MIN     # (惯性权重)
    c1 = C1     # 认知参数
    c2 = C2     # 社会参数

    global_costs = []

    for episode in range(max_iter):

        temp_state_x = []
        temp_state_v = []

        # dynamic w (惯性权重)
        w = w_max - (w_max - w_min) * (episode / max_iter)

        # dynamic c1 & c2
        # c1 = 2.5 - (2.5 - 0.5) * (episode / max_iter)  # c1 从 2.5 线性减小到 0.5
        # c2 = 0.5 + (2.5 - 0.5) * (episode / max_iter)  # c2 从 0.5 线性增大到 2.5

        time1 = time.time()
        with multiprocessing.Pool(processes=THREAD_NUM) as pool:  # start n threads for calculation
            results = pool.map(evaluate_particle, particles)  # parallel cost function
        time2 = time.time()

        # start optimization of the particle parameter
        for fitness, particle_id in results:
            particle = particle_dict[particle_id]
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

            temp_state_x.append(particle.position)
            temp_state_v.append(particle.velocity)

        # record global fitness
        global_costs.append(global_best_fitness)

        # name_ = str(global_count) + str(physical_data)
        name_ = 'temp_state_x' + str(episode) + '.npy'
        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
        np.save(file_path, temp_state_x)  # 保存数组

        # record data into files
        name_ = 'temp_state_v' + str(episode) + '.npy'
        file_path = os.path.join(save_path, name_)  # 创建每个数组的保存路径
        np.save(file_path, temp_state_v)  # 保存数组

        print("episode", episode)
        print('本次循环的推理时间为：', time2-time1)

    return global_best_position, global_best_fitness, global_costs


if __name__ == "__main__":

    x_bound = [P_Min, P_Max]

    # 基于位置范围的比例确定速度范围
    alpha = ALPHA    # 速度范围比例因子
    v_min = -alpha * (x_bound[1] - x_bound[0])
    v_max = alpha * (x_bound[1] - x_bound[0])
    v_bound = [v_min, v_max]

    # 粒子数量
    num_particles = PARTICLE_NUM
    # 最大迭代次数
    max_iter = EPISODE_NUM

    # 记录文件夹
    current_folder = '.'
    folder_name = str(ALPHA) + "_" + str(W_MAX) + '_' + str(W_MIN) + '_' + str(C1) + '_' + str(C2)
    save_path = os.path.join(current_folder, folder_name)
    os.makedirs(save_path, exist_ok=True)

    best_position, best_fitness, global_costs = pso(x_bound, v_bound, num_particles, max_iter, save_path)
    print(f'Best position: {best_position}')
    print(f'Best fitness: {best_fitness}')
    np.save('cost', global_costs)