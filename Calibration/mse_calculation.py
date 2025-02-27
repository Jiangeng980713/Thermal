import matplotlib.pyplot as plt
import numpy as np

with open('1_1_0.1_50_None\mses.txt', 'r') as f:
    data = f.read()  # 读取整个文件内容

# 将字符串拆分为列表
loaded_list_1 = list(map(float, data.split(',')))

with open('global_counts.txt', 'r') as f:
    data = f.read()  # 读取整个文件内容

# 将字符串拆分为列表
global_count = list(map(int, data.split(',')))

# 示例数据
y = loaded_list_1
print(len(y))
v = np.array(y)
print(np.average(v))

# 根据 y 的长度自动生成自然数序列
x = global_count

# 绘制折线图
plt.plot(x, y, marker='H', color='b', linestyle='-')

# 显示图表
plt.show()
