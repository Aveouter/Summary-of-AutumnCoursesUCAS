from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math, os, torch
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import Truncate, PadTransform
from torchtext.vocab import Vectors, build_vocab_from_iterator
import numpy as np
import random
import time
import spacy
import sys
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import csv


def data_divide(vocab_dict, output_path='data/Vec/DeletedSegDone.txt'):
    with open(output_path, 'r', encoding='utf-8') as file:
        data_train = []
        data_target = []
        for i, line in enumerate(file):
            # 提取汉字词汇（假设词汇间使用空格分隔）
            # print(i)
            enpty_word = [1]*20
            chinese_words = [vocab_dict[word] for word in line.strip().split() if
                             any('\u4e00' <= char <= '\u9fff' for char in word)]
            # for random_integer in range(6, len(chinese_words)):
            #     data.append((chinese_words[random_integer - 6:random_integer - 1], chinese_words[random_integer]))
            if len(chinese_words) > 5:
                chinese_words = enpty_word+chinese_words
                random_integer = random.randint(20, len(chinese_words) - 1)
                data_train.append(chinese_words[random_integer - 20:random_integer])
                data_target.append(chinese_words[random_integer - 19:random_integer + 1])
    return data_train, data_target


def data_Process(output_path='data/Vec/DeletedSegDone.txt'):
    with open(output_path, 'r', encoding='utf-8') as file:
        data = []
        for i, line in enumerate(file):
            # 提取汉字词汇（假设词汇间使用空格分隔）
            # enpty_word = ['#填充', '#填充', '#填充', '#填充', '#填充']
            chinese_words = [word for word in line.strip().split()]
            chinese_words = chinese_words
            data.append(chinese_words)
            # for random_integer in range(6, len(chinese_words)):
            #     data.append((chinese_words[random_integer - 6:random_integer - 1], chinese_words[random_integer]))
    return data


def load_data():
    # 通过迭代器构建词汇表
    vocab = build_vocab_from_iterator(data_Process(), min_freq=5, specials=['<unk>', '<pad>'])
    vocab_dict = vocab.get_stoi()
    # 将默认索引设置为'<unk>'
    vocab.set_default_index(vocab['<unk>'])
    ntokens = len(vocab)  # 词汇表的大小
    print("------" * 30)

    # data loading
    data_train, data_target = data_divide(vocab_dict)
    # print(data_train)
    #
    # # 指定保存到本地的文件路径
    # file_path = "data/update/my_data_train.txt"
    # file_path2 = "data/update/my_data_target.txt"
    # # 将列表写入文件
    # with open(file_path, 'w') as file1:
    #     for item in data_train:
    #         file1.write(str(item) + '\n')
    # print(f"List has been saved to {file_path}")
    #
    # with open(file_path2, 'w') as file2:
    #     for item in data_target:
    #         file2.write(str(item) + '\n')
    #
    #
    # print(f"List has been saved to {file_path2}")
    return data_train, data_target
