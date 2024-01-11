import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 一个包含字符串的列表
with open('word_counts.csv', 'r') as file:
    # 创建CSV读取器
    items = []
    csv_reader = csv.reader(file)
    # 遍历CSV文件中的每一行
    for row in csv_reader:
        # 在这里，row是一个包含CSV文件中一行数据的列表
        # 你可以将row添加到数组中，从而得到一个包含所有行的数组
        items.append(row)

word_counts = items[1:]
# shape = np.array(word_counts).shape
# print(shape)
# print(word_counts)
x = [int(row[0]) for row in word_counts]
y = [float(row[3].rstrip('%')) for row in word_counts]

x = x[:100]
print(x)
y = y[:100]
print(y)
# 多项式拟合，这里选择3次多项式
fit = np.polyfit(x, y, 15)
fit_fn = np.poly1d(fit)

# 创建一个新的 x 值范围，用于绘制拟合曲线
x_fit = np.linspace(min(x), max(x), 100)

# 绘制原始数据点
plt.scatter(x, y, s=10, label='Data', color='orange')

# 绘制拟合曲线
plt.plot(x_fit, fit_fn(x_fit), label='Fit', color='green')

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel('word occurrence probability')
plt.ylabel('word list by times')
plt.title("Fitting curve of Zipf's law")

# 显示图表
plt.show()
