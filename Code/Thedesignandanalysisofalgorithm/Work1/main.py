# # 最大距离问题描述：
#
# # 最大距离题。给定n个数x₁, x₂, ..., xₙ，求n个数在数轴上相邻两数之间的最大差值。
#
# 简单直接的解法：将n个数排序后计算相邻两数之间的距离，再找出最大值。这个解法的时间复杂度为O(n * log(n))。
#
# 设计解决最大周距离问题的高效算法：
#
# 算法输入：
# 输入数据由文件名input.txt的文本提供。文件的第一行包含一个正整数n，接下来的n行包含n个整数。
#
# 算法输出：
# 将找到的最大周距离输出到文件output.txt中。

import numpy as np
import time

def input_generate(n):
    # 生成n个实数的随机数列
    list1 = {}
    for i in range(1, n):
        random_float_range = np.random.uniform(0.0, 1000.0)
        list1[i] = random_float_range
    # print("指定范围内的随机浮点数:", random_float_range)
    return list1


def write_Txt(list1, n):
    # 打开一个文本文件以写入模式
    file_path = "input.txt"
    # 使用 "w" 模式打开文件，如果文件不存在将创建一个新文件，如果文件已经存在则会覆盖原有内容
    with open(file_path, "w") as file:
        file.write(str(n))
        file.write("\n")
        for item in list1:
            file.write(str(list1))
    print("intput已创建并写入内容。")


def write_result(result):
    # 打开一个文本文件以写入模式
    file_path = "output.txt"
    # 使用 "w" 模式打开文件，如果文件不存在将创建一个新文件，如果文件已经存在则会覆盖原有内容
    with open(file_path, "w") as file:
        file.write("最大间距离为：")
        file.write(str(result))
        file.write("\n")
    print("output已创建并写入内容。")
    print("最大间距离为：", result)


def auto_sum(list11, na):
    # 解析输入为整数列表
    gap = list11[2] - list11[1]
    if list11[2] > list11[1]:
        maxoflist = list11[2]
        minoflist = list11[1]
    else:
        maxoflist = list11[1]
        minoflist = list11[2]
    for i in range(1, na - 2):
        if list11[i + 2] > maxoflist:
            maxoflist = list11[i + 2]
            gap = abs(maxoflist - minoflist)
        elif list11[i + 2] < minoflist:
            minoflist = list11[i + 2]
            gap = abs(maxoflist - minoflist)
    return gap
    print("最大间隙为:", gap)


if __name__ == "__main__":
    n = input("键入输入个数n:")
    start = time.perf_counter()
    n = int(n)
    list1 = input_generate(n)
    write_Txt(list1, n)
    result = auto_sum(list1, n)
    write_result(result)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))