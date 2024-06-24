import random
import time
import multiprocessing
from worker import *


# def worker(name, V, q):
#     t = 0
#     for i in range(10):
#         print(name + " " + str(i))
#         x = int(name)
#         t += x
#         # time.sleep(x * 0.1)
#     q.put({name: t})


if __name__ == '__main__':
    # outer
    q = multiprocessing.Queue()
    # inner
    jobs = []

    for id in range(10):
        p = multiprocessing.Process(target=worker, args=(str(id), q))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [q.get() for a in jobs]

    print('1', results)

    jobs = []

    for id in range(10):
        p = multiprocessing.Process(target=worker, args=(str(id), q))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [q.get() for j in jobs]

    print('2,', results)
