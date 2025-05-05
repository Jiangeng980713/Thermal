import matplotlib.pyplot as plt
import numpy as np


def data_treat(name, scale):
    percentage = np.load(name) * scale
    percentage[:6] = -percentage[:6]
    offset = np.average(percentage)
    percentages = percentage - offset

    variance = np.var(np.array(percentages))
    SD = np.sqrt(variance)
    print(name, SD)
    return percentages


percentages = data_treat('percentages-400.npy', scale=1.4)
percentages_1 = data_treat('percentages-500.npy', scale=1.35)
percentages_2 = data_treat('percentages-600.npy', scale=1.25)
percentages_3 = data_treat('percentages-700.npy', scale=1.2)
percentages_4 = data_treat('percentages-800.npy', scale=1.1)


# import matplotlib.pyplot as plt
# import numpy as np


# data = [percentages, percentages_1, percentages_2, percentages_3, percentages_4]
# labels = ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5']
# colors = ['skyblue', 'orange', 'green', 'red', 'purple']
#
# # 横轴位置
# x = np.arange(len(percentages_1))  # 假设每组有5个值
# bar_width = 0.15  # 柱子的宽度
#
# # 绘图
# for i, (group, color) in enumerate(zip(data, colors)):
#     plt.bar(x + i * bar_width, group, width=bar_width, label=labels[i], color=color)
#
# # 设置 x 轴刻度居中
# plt.xticks(x + bar_width * 2, [str(i) for i in x])  # 中心偏移量：bar_width * (num_groups / 2)
#
# # 添加标签
# plt.xlabel('Index (unit: 1)')
# plt.ylabel('Percentage (unit: 1)')
# plt.legend()
# plt.title('Grouped Bar Chart for 5 Inputs')
#
# # 显示图形
# plt.show()


# x 坐标索引
x = np.arange(len(percentages_1))

# 绘制每一组折线图
plt.plot(x, percentages_1, marker='o', label='Input 1')
plt.plot(x, percentages_2, marker='s', label='Input 2')
plt.plot(x, percentages_3, marker='^', label='Input 3')
plt.plot(x, percentages_4, marker='d', label='Input 4')
plt.plot(x, percentages, marker='x', label='Input 5')

# 设置坐标轴和图例
plt.xticks(ticks=x, labels=[str(i) for i in x])
plt.xlabel('Index (unit: 1)')
plt.ylabel('Percentage (unit: 1)')
# plt.ylim(-0.35, 0.35)  # 设置 y 轴范围
plt.title('Trend of 5 Input Curves')
plt.legend()

# 显示图形
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