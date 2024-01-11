import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import Words2Vec


class FNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(FNNLanguageModel, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * context_size, 128)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(128, 64)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(64, vocab_size)

    def forward(self, inputs):
        # inputs 的 shape 为 (batch_size, context_size)
        out = self.tanh1(self.linear1(inputs))
        out = self.tanh2(self.linear2(out))
        probs = self.linear3(out)
        return probs


# 训练模型的函数
def train_fnn_language_model(model, train_loader, epochs=10, lr=0.001):
    # def CrossEntropyLoss(probs,target)
    # # 在模型定义时使用 CrossEntropyLoss
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    i = 0
    average_loss_qu = 0
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        for context, target in train_loader:
            model.zero_grad()
            probs = model(context)
            probs = probs.squeeze(dim=1)
            target0 = target.t().type(torch.long)
            loss = loss_function(probs, target0.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            total_batches += 1
        average_loss = total_loss / total_batches
        if abs(average_loss_qu - average_loss) < 0.01:
            i = i + 1
        else:
            i = 0
        average_loss_qu = average_loss
        if i >= 5:
            break
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')


'''
dataprocessing
'''


def vocab_init(num):
    # 打开文件并读取第一列元素
    first_column_elements = []
    out_vector = 'data/Vec/corpusSegDone.vector'
    with open(out_vector, 'r', encoding='utf-8') as file:
        next(file)
        for i, line in enumerate(file):
            # 使用适当的分隔符提取第一列元素
            if num == 0:
                first_column_element = line.strip().split()[num]  # 假设使用空格分隔
            if num == 1:
                first_column_element = line.strip().split()[1:]  # 假设使用空格分隔
                first_column_element = [float(element) for element in first_column_element]
            first_column_elements.append(first_column_element)
    return first_column_elements


def dataprocess():
    output_path = 'data/Vec/DeletedSegDone.txt'
    with open(output_path, 'r', encoding='utf-8') as file:
        data = []
        for i, line in enumerate(file):
            # 提取汉字词汇（假设词汇间使用空格分隔）
            enpty_word = ['#填充', '#填充', '#填充', '#填充', '#填充']
            chinese_words = [word for word in line.strip().split() if
                             any('\u4e00' <= char <= '\u9fff' for char in word)]
            chinese_words = enpty_word + chinese_words
            # for random_integer in range(6, len(chinese_words)):
            #     data.append((chinese_words[random_integer - 6:random_integer - 1], chinese_words[random_integer]))
            if len(chinese_words) > 6:
                random_integer = random.randint(6, len(chinese_words) - 1)
                data.append((chinese_words[random_integer - 6:random_integer - 1], chinese_words[random_integer]))
    return data


def one_hot_vector_torch(n, length):
    one_hot = torch.zeros(length)
    one_hot[n] = 1
    return one_hot


class LanguageModelDataset(Dataset):
    def __init__(self, data, word_to_index):
        self.data = data
        self.word_to_index = word_to_index

    def __len__(self):
        return len(self.data)

    def index_array_to_one_hot(self, index_array, num_classes):
        one_hot_array = np.zeros(num_classes)  # 初始化全零的二维数组
        one_hot_array[index_array - 1] = 1  # 将对应索引的位置设为1
        return torch.from_numpy(one_hot_array).long()

    def __getitem__(self, idx):
        context, target = self.data[idx]
        # context_indices = [self.word_to_index[word] for word in context]
        context_0 = torch.Tensor(context)
        context_0 = context_0.view(1, -1)
        target = torch.Tensor([target])
        return context_0.to(device), target.to(device)


'''
data processing
'''
device = torch.device("cuda:0")

# load dataL
print('Loading data...')
model_WORD = Words2Vec.load_word2vec_model("word2vec.model")  # 加载词模型
data = dataprocess()  # 词模型

vocab = vocab_init(0)  # 词汇表

Ltable = vocab_init(1)  # 词汇向量对应表

word_to_index = {word: Ltable[i] for i, word in enumerate(vocab)}  # 向量表
word_to_onehot = {word: i for i, word in enumerate(vocab)}  # 词汇索引
word_to_index.update({'#填充': [0] * 100})
# word_to_onehot.update({'#填充': 128898})  # 填充
indexed_data = [[[word_to_index[word] for word in context], word_to_onehot[target]] for context, target in data]

# 划分训练集和测试集
train_size = int(len(indexed_data) - 1000)
test_size = 1000
train_data, test_data = indexed_data[:train_size], indexed_data[train_size:]

vocab_size = len(vocab)
embedding_dim = 100

# 创建数据加载器
train_dataset = LanguageModelDataset(train_data, word_to_index)
test_dataset = LanguageModelDataset(test_data, word_to_index)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
print('Training beginning...')
# 创建和训练FNN语言模型

fnn_model = FNNLanguageModel(vocab_size, embedding_dim, 5)
fnn_model = fnn_model.to(device)
# train_fnn_language_model(fnn_model, train_loader, epochs=1000)
# torch.save(fnn_model.state_dict(), 'ffn_language_model.pth')
#

#

def calculate_perplexity(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for context, target in dataloader:
            model.zero_grad()
            probs = model(context)
            probs = probs.squeeze(dim=1)
            target0 = target.t().type(torch.long)
            total_loss += criterion(probs, target0.squeeze())
            total_count = (len(dataloader) - 1)
    average_loss = total_loss / total_count
    print(average_loss)
    perplexity = torch.exp(torch.tensor([average_loss]))
    return perplexity.item()


# 如果你使用 NLL Loss
criterion = nn.CrossEntropyLoss()
test_perplexity = calculate_perplexity(fnn_model, test_loader, criterion)
print(f'Test Perplexity: {test_perplexity}')
