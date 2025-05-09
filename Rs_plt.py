import matplotlib.pyplot as plt
import numpy as np


def data_treat(name, scale):
    percentage = np.load(name) * scale
    percentage[:6] = -percentage[:6]

    offset = np.average(percentage)
    percentage = percentage - offset

    indices = np.arange(len(percentage))
    target_indices_4 = indices[indices % 7 == 3]
    target_indices_5 = indices[indices % 7 == 4]
    target_indices_6 = indices[indices % 7 == 5]

    slope_45 = (percentage[target_indices_5] - percentage[target_indices_4]) / (target_indices_5 - target_indices_4)
    slope_56 = (percentage[target_indices_6] - percentage[target_indices_5]) / (target_indices_6 - target_indices_5)

    slope_67 = (slope_45 + slope_56)/2

    target_indices_7 = indices[indices % 7 == 6]

    percentage[target_indices_7] = percentage[target_indices_6] + slope_67

    variance = np.var(np.array(percentage))
    SD = np.sqrt(variance)
    print(name, SD)

    return percentage, SD


percentages, SD = data_treat('percentages-400.npy', scale=1.5)
percentages_1, SD_1 = data_treat('percentages-500.npy', scale=1.45)
percentages_2, SD_2 = data_treat('percentages-600.npy', scale=1.4)
percentages_3, SD_3 = data_treat('percentages-700.npy', scale=1.3)
percentages_4, SD_4 = data_treat('percentages-800.npy', scale=1.3)


# x 坐标索引
x = np.arange(len(percentages_1))

plt.figure(figsize=(12, 6))  # 先设置图像大小

# 绘制每一组折线图
plt.plot(x, percentages, marker='x', label='Input 0')
plt.plot(x, percentages_1, marker='o', label='Input 1')
plt.plot(x, percentages_2, marker='s', label='Input 2')
plt.plot(x, percentages_3, marker='^', label='Input 3')
plt.plot(x, percentages_4, marker='d', label='Input 4')
plt.show()


# 柱状图 plt

# 数据
labels = ['400', '500', '600', '700', '800']
sd_values = [SD, SD_1, SD_2, SD_3, SD_4]

# 绘图
plt.bar(labels, sd_values, color='skyblue')

# 添加标签和标题
plt.xlabel('Input Power (W)')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Input Power')

# 显示图形
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()


# 800W
# 0.009939602453890764
# 0.09969755490427418

# 700W
# 0.010781055494859412
# 0.1038318616555603

# 600W
# 0.012383529835504161
# 0.11128130946167088

# 500W
# 0.015386671993206986
# 0.12404302476643733

# 400W
# 0.021145899766218272
# 0.1454162981450782