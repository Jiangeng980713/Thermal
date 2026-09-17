from worker import *
import multiprocessing
import time
import os
import datetime

class Particle:
    def __init__(self, particle_id, dim, x_bound, v_bound, resume, input_vector):

        self.id = particle_id

        if not resume:
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
    fitness, _, _ = worker_agent(particle.position)
    return fitness, particle.id


def save_checkpoint(particles, global_best_position, global_best_fitness, episode, save_path):

    particle_states = []

    for p in particles:
        particle_states.append({
            'id': p.id,
            'position': p.position,
            'velocity': p.velocity,
            'best_position': p.best_position,
            'best_fitness': p.best_fitness
        })

    Checkpoint_PSO = {
        'particles': particle_states,
        'global_best_position': global_best_position,
        'global_best_fitness': global_best_fitness,
        'episode': episode
    }

    file_path = os.path.join(save_path, 'checkpoint.npy')

    np.save(file_path, Checkpoint_PSO)


def load_checkpoint(dim, x_bound, v_bound, save_path):

    file_path = os.path.join(save_path, 'checkpoint.npy')

    checkpoint = np.load(file_path, allow_pickle=True).item()

    particles = []
    for state in checkpoint['particles']:
        p = Particle(state['id'], dim, x_bound, v_bound, resume=True, input_vector=state['position'])
        p.velocity = state['velocity']
        p.best_position = state['best_position']
        p.best_fitness = state['best_fitness']
        particles.append(p)

    global_best_position = checkpoint['global_best_position']
    global_best_fitness = checkpoint['global_best_fitness']
    start_episode = checkpoint['episode'] + 1

    return particles, global_best_position, global_best_fitness, start_episode


def pso(x_bound, v_bound, num_particles, max_iter, save_path, load_path):

    resume = False

    dim = LAYER_HEIGHT * STRIPE_NUM    # dimension for particle

    input_power = 800
    input_vector = np.full((dim,), input_power)

    if resume:
        particles, global_best_position, global_best_fitness, start_episode = load_checkpoint(dim, x_bound, v_bound, load_path)
    else:
        particles = [Particle(i, dim, x_bound, v_bound, resume, input_vector) for i in range(num_particles)]
        global_best_position = np.random.uniform(x_bound[0], x_bound[1], dim)
        global_best_fitness = float('inf')
        start_episode = 0

    # 生成一个字典 {particle_id: particle}，确保 ID 和粒子映射正确
    particle_dict = {particle.id: particle for particle in particles}

    # fixed w parameter
    w_max = W_MAX  # (惯性权重)
    w_min = W_MIN  # (惯性权重)

    global_costs = []

    for episode in range(start_episode, max_iter):

        temp_state_x = []
        temp_state_v = []

        # dynamic w (惯性权重)
        w = w_max - (w_max - w_min) * (episode / max_iter)

        # dynamic c1 & c2
        c1 = C1_MAX - (C1_MAX - C1_MIN) * (episode / max_iter)  # c1 从 2.5 线性减小到 0.5
        c2 = C2_MIN + (C2_MAX - C2_MIN) * (episode / max_iter)  # c2 从 0.5 线性增大到 2.5

        time1 = time.time()

        with multiprocessing.Pool(processes=THREAD_NUM) as pool:  # start n threads for calculation
            results = pool.map(evaluate_particle, particles)  # parallel cost function
        time2 = time.time()

        # estimate particle position
        for fitness, particle_id in results:
            particle = particle_dict[particle_id]
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        # update particle position
        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

            temp_state_x.append(particle.position)
            temp_state_v.append(particle.velocity)

        # save the optimization information
        save_checkpoint(particles, global_best_position, global_best_fitness, episode, save_path)

        # record global fitness
        global_costs.append(global_best_fitness)

        name_ = 'global_costs'
        file_path = os.path.join(save_path, name_)
        np.save(file_path, global_costs)

        print("episode", episode)
        print('本次循环的推理时间为：', time2 - time1)
        print('本次损失函数为：', global_best_fitness)

    return global_best_position, global_best_fitness, global_costs


if __name__ == "__main__":

    x_bound = [P_Min, P_Max]

    # 基于位置范围的比例确定速度范围
    alpha = ALPHA  # 速度范围比例因子
    v_min = -alpha * (x_bound[1] - x_bound[0])
    v_max = alpha * (x_bound[1] - x_bound[0])
    v_bound = [v_min, v_max]

    # 粒子数量
    num_particles = PARTICLE_NUM

    # 最大迭代次数
    max_iter = EPISODE_NUM

    # save file
    current_folder = '.'
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    folder_name = 'ALPHA' + str(ALPHA) + '_' + "w" + str(W_MAX) + str(W_MIN) + '_' + 'c1' + str(C1_MAX) + str(C1_MIN) + '_' + 'c2' + str(C2_MAX) + str(C2_MIN) + '_' + today_date
    save_path = os.path.join(current_folder, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # load file
    load_path = None

    best_position, best_fitness, global_costs = pso(x_bound, v_bound, num_particles, max_iter, save_path, load_path)

    print(f'Best position: {best_position}')
    print(f'Best fitness: {best_fitness}')

    name_ = 'cost_function'
    file_path = os.path.join(save_path, name_)
    np.save(file_path, global_costs)

    name_ = 'best_position'
    file_path = os.path.join(save_path, name_)
    np.save(file_path, best_position)