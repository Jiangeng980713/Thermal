import psutil
from worker import *
import multiprocessing
import time


def evaluate_particle(a):

    result = a ** 2

    return result

if __name__ == "__main__":

    a = [5.0]

    for i in range(2):
        parent_process = psutil.Process()  # 获取当前进程

        time1 = time.time()
        with multiprocessing.Pool(processes=THREAD_NUM) as pool:  # start n threads for calculation
            results = pool.map(evaluate_particle, a)  # parallel cost function

        print('pool', pool)
        pool.close()  # 关闭进程池，禁止新任务
        pool.join()   # 等待所有任务完成

        time2 = time.time()

        # check whether the children parallel is done
        # current_children = parent_process.children()
        # time.sleep(1)
        # print("Current running child processes:", current_children)
        # assert len(current_children) == 0, 'children process is not done'

        max_wait_time = 5  # 设置最大等待时间
        wait_time = 0
        start_time = time.time()  # 记录开始等待的时间

        while parent_process.children() and wait_time < max_wait_time:
            time.sleep(0.5)  # 每 0.5 秒检查一次
            wait_time += 0.5

        end_time = time.time()  # 记录结束等待的时间
        elapsed_time = end_time - start_time  # 计算子进程释放所需时间

        if not parent_process.children():
            print(f"子进程在 {elapsed_time:.2f} 秒内成功释放")
        else:
            print(f"子进程未完全释放，已等待 {elapsed_time:.2f} 秒，可能存在卡住的进程！")

        current_children = parent_process.children()
        for child in current_children:
            print(f"PID {child.pid} | Status: {child.status()} | Name: {child.name()}")

        import psutil

        # 查找所有正在运行的进程
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                # 如果进程名是 conhost.exe，则打印相关信息
                if proc.info['name'].lower() == 'conhost.exe':
                    print(f"PID: {proc.info['pid']}, Status: {proc.info['status']}, Name: {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # 处理进程可能已经终止或没有访问权限的情况
                pass
