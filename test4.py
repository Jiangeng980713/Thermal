"""
简单示例：创建一个子进程
"""
import os
import time
from multiprocessing import Process



def func(s, s1):
    # 输出传入的参数，当前子进程的进程ID，当前进程的父进程ID
    return 100

def func1():
    time.sleep(5)
    return 2


# 注意：此处的if __name__ == '__main__'语句不能少
if __name__ == '__main__':
    # 打印当前进程的进程ID
    print(os.getpid())
    print('main process start...')
    # 创建进程对象
    p = []
    for i in range(5):
        p.append(Process(target=func, args=('hello', 'hello1', )))
    p.append(Process(target=func1, args=()))
    # 生成一个进程，并开始运行新的进程
    for i in range(6):
        p[i].start()
    for i in range(6):
        p[i].join()
        a = p[i].exitcode
        print(str(i) + ' ' + str(a) + ' is done')
    # 等待子进程运行完毕
    print('main process end!')
