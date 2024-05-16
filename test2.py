import matplotlib.pyplot as plt
import numpy as np


def create_2d_gaussian(rows, cols, std):
    x, y = np.meshgrid(np.arange(cols) - (cols - 1) / 2, np.arange(rows) - (rows - 1) / 2)
    gauss_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
    return gauss_kernel / np.max(gauss_kernel)


# 示例：创建一个5x5的二维高斯分布矩阵，标准差为1.5
std = 1.5
rows, cols = 10, 10
gauss_matrix = create_2d_gaussian(rows, cols, std)
print(gauss_matrix)

# 设置图表标题和轴标签
plt.title('2D Gaussian Distribution')
plt.xlabel('X axis')
plt.ylabel('Y axis')

# 显示图表
plt.show()