import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def process_strings(strings):
    total_size = 0  # 用于跟踪已读取的字符串的总大小
    batch = []  # 用于存储将传递给count函数的字符串批次
    print('start processing batch0...')
    word_counts = None
    j = 1
    batch_turn = 1
    for i in range(0, len(strings)):
        string_size = len(strings[i])
        total_size += string_size
        if total_size >= 2 * 1024 * 1024:  # 2MB in bytes
            word_counts, num = count(j, i, strings, word_counts)
            j = i
            batch.append(num)
            total_size = 0  # 重置总占用空间大小

            batch_turn += 1
            print('start processing batch' + str(batch_turn) + '...')

    # 处理剩余的字符串，如果有的话
    if total_size:
        word_counts, num = count(j, len(strings), strings, word_counts)
        batch.append(num)
    return batch


def count(m, n, ListOfItems, word_counts):
    if word_counts is None:
        # print('wordcounts is none')
        word_counts = Counter(ListOfItems[m:n])
    else:
        # print('wordcounts is no empty')
        word_counts.update(ListOfItems[m:n])
    total_elements = n
    # 使用列表解析计算出现次数的百分比
    word_percentage = {word: float(count1 / total_elements) for word, count1 in word_counts.items()}

    values_view = word_percentage.values()
    # 将视图对象转换为列表
    word_percentage2 = list(values_view)
    # word_sum = sum(word_percentage2)
    # print(word_sum)
    result = 0
    for xnum in word_percentage2:
        result -= math.log2(xnum) * xnum
    return word_counts, result


if __name__ == '__main__':
    # 一个包含字符串的列表
    with open('data/origin/rebuild_wordlist.txt', 'r') as file:
        # 创建CSV读取器
        items = []
        csv_reader = csv.reader(file)
        # 遍历CSV文件中的每一行
        print("reading wordlist...")
        for row in csv_reader:
            # 在这里，row是一个包含CSV文件中一行数据的列表
            # 你可以将row添加到数组中，从而得到一个包含所有行的数组
            items.append(row[0])

    letter_list = []
    # 遍历单词数组并拆分每个单词为字母，并添加到letter_list中
    for word in items:
        for letter in word:
            letter_list.append(letter)
    print('letter_list has been created')
    batch_list = process_strings(letter_list)
    print(batch_list)

    x = [x for x in range(1, len(batch_list) + 1)]
    y = [row for row in batch_list]

    # x = x[:100]
    # print(x)
    # y = y[:100]
    # print(y)
    # 多项式拟合，这里选择3次多项式
    fit = np.polyfit(x, y, 15)
    fit_fn = np.poly1d(fit)

    # 创建一个新的 x 值范围，用于绘制拟合曲线
    x_fit = np.linspace(min(x), max(x), 100)

    # 绘制原始数据点
    plt.scatter(x, y, s=10, label='Data', color='red')

    # 绘制拟合曲线
    plt.plot(x_fit, fit_fn(x_fit), label='Fit', color='black')

    # 添加图例
    plt.legend()

    # 添加标签和标题
    plt.ylabel('Entropy')
    plt.xlabel('Increasing Text Content Continuously (2MB at a Time)')
    plt.title("Entropy Change Curve with Increasing Text Content")

    # 显示图表
    plt.show()
