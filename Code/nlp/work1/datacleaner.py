import csv
from pathlib import Path

import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
nltk.download('wordnet')


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def TXTreader():
    folder_path = Path('FileofBook')  # 替换为你的文件夹路径
    # 获取文件夹中的所有文件名
    file_names = [file.name for file in folder_path.iterdir() if file.is_file()]
    print(file_names)
    for i in file_names:
        file_name = "data/origin/origin_text.txt"
        with open('FileofBook/' + i, 'r', encoding='utf-8') as file:
            a = file.read()
            cleaned_string = re.sub(r'[^a-zA-Z]', ' ', a)
            cleaned_string = cleaned_string.lower()
            # print(a)
        # 打开文件并写入文本内容
        with open(file_name, "a", encoding='utf-8') as file:
            file.write('\n')
            file.write(cleaned_string)
            print(f"文件 '{i}' 创建并写入成功！")
    print(f"文件 '{file_name}' 创建并写入成功！")


def dataprocess():
    """
    在本函数中，我们想实现如下功能
    1，将单词按照空格分开，形成一个字符串列表
    2.我们无法忽视is，are等词在其中的连写形式，并且对于单词的时态，单复数形式如何管理的问题
        具体的执行方法为
        1.由于我们在上面执行的预处理中已经将所有的符号转换为空格，所以我们将所有的s按照is统计，re按照are等统计
        2.对于单词的单复数和时态问题的处理
    """
    # split the string
    with open("data/origin/origin_text.txt", "r", encoding='utf-8') as file:
        data = file.read()
        data = data.split()
    print('data has been loaded')
    # # 进行data processing # #
    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in data]
    print('legitimatized has benn accomplished')
    # print(lemmatized_words)
    file_name = "data/origin/rebuild_wordlist.txt"
    # 打开文件并将字符串写入文件
    with open(file_name, 'w') as file:
        for i in lemmatized_words:
            file.write(i+'\n')

    word_counts = Counter(lemmatized_words)
    total_elements = len(lemmatized_words)
    # 使用列表解析计算出现次数的百分比
    word_percentage = {word: count / total_elements * 100 for word, count in word_counts.items()}
    print(word_percentage)
    # 按照出现次数从大到小排序
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    print(sorted_word_counts)

    data = [("序号", "单词字符串", "出现次数", "出现次数百分比")]

    for index, (word, count) in enumerate(sorted_word_counts.items(), start=1):
        percentage = word_percentage[word]
        data.append((index, word, count, f"{percentage:.2f}%"))

    # 将数据保存到CSV文件
    with open("word_counts.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

    print("CSV文件已生成：word_counts.csv")


#
if __name__ == '__main__':
    # mkdir('data/origin')
    # TXTreader()
    # dataprocess()

