import multiprocessing
import time

# 定义要并行计算的函数
def dummy_function(number):
    return number

if __name__ == "__main__":
    pool_sizes = [2, 4, 8, 16, 32]  # 不同大小的进程池
    for size in pool_sizes:
        start_time = time.time()
        pool = multiprocessing.Pool(processes=size)
        pool_creation_time = time.time() - start_time
        pool.close()
        pool.join()
        print(f'Pool size {size}: Creation time = {pool_creation_time:.6f} seconds')
