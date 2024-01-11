import random

'''
生成一个随机整数数列，长度为n；
使用动态规划的思想求其和最大的子序列，输出子序列和其和
'''
import random

# Generate a list of 100 random numbers between -100 and 100
num_list = [random.randint(-100, 100) for _ in range(10)]


def max_subsequence_sum(arr):
    n = len(arr)
    max_sum = arr[0]
    current_sum = arr[0]

    for i in range(1, n):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)

    return max_sum


# 应用于随机数列
max_sum = max_subsequence_sum(num_list)
print("随机整数数列:", num_list)
print("最大子序列和:", max_sum)
