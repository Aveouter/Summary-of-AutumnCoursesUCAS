import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import Truncate, PadTransform
from torchtext.vocab import Vectors, build_vocab_from_iterator
import numpy as np
import random
import spacy
import sys
from datapocessingfornlp import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, nlayers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, nlayers, dropout=dropout)
            # getattr(nn, rnn_type) 相当于 nn.rnn_type
            # nlayers代表纵向有多少层。还有个参数是bidirectional: 是否是双向LSTM，默认false
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_size, hidden_size, nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)  # (1000, 50002)

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        # input: seq_len * batch = [50, 32]，可以在LSTM定义batch_first = True
        # hidden = (nlayers * b * hidden_size)
        # hidden是个元组，输入有两个参数，一个是刚开始的隐藏层h的维度，一个是刚开始的用于记忆的c的维度，
        embed = self.drop(self.embedding(input))  # seq_len * b * embedding_size
        output, hidden = self.rnn(embed, hidden)
        # output.shape = seq_len * b * hidden_size
        # hidden元组 = (h层：nlayers * 32 * hidden_size, c层：nlayers * 32 * hidden_size)
        output = self.drop(output)
        linear = self.linear(output.view(-1, output.size(2)))
        # [seq_len*batch, hidden_size] -> [seq_len*batch, vocab_size]

        return linear.view(output.size(0), output.size(1), linear.size(1)), hidden
        # 输出恢复维度 :[seq_len, b, vocab_size]
        # hidden = (h层维度：nlayers * b * hidden_size, c层维度：nlayers * b * hidden_size)

    def init_hidden(self, batch_size, requires_grad=True):
        # 最初隐藏层参数的初始化
        weight = next(self.parameters())
        # weight = torch.Size([50002, 500])是所有参数的第一个参数
        # 所有参数self.parameters()，是个生成器

        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, batch_size, self.hidden_size),
                                     requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, batch_size, self.hidden_size),
                                     requires_grad=requires_grad))
            # return = (2 * 32 * 1000, 2 * 32 * 1000)
            # 这里不明白为什么需要weight.new_zeros，我估计是想整个计算图能链接起来
            # 这里特别注意hidden的输入不是model的参数，不参与更新，就跟输入数据x一样
        else:
            return weight.new_zeros((self.nlayers, batch_size, self.hidden_size),
                                    requires_grad=requires_grad)
            # GRU神经网络把h层和c层合并了，所以这里只有一层。


def data_divide(output_path='data/Vec/DeletedSegDone.txt'):
    with open(output_path, 'r', encoding='utf-8') as file:
        data_train = []
        data_target = []
        for i, line in enumerate(file):
            # 提取汉字词汇（假设词汇间使用空格分隔）
            enpty_word = ['<pad>'] * 5
            chinese_words = [word for word in line.strip().split() if
                             any('\u4e00' <= char <= '\u9fff' for char in word)]
            chinese_words = enpty_word + chinese_words
            # for random_integer in range(6, len(chinese_words)):
            #     data.append((chinese_words[random_integer - 6:random_integer - 1], chinese_words[random_integer]))
            if len(chinese_words) > 7:
                random_integer = random.randint(6, len(chinese_words) - 2)
                data_train.append(chinese_words[:random_integer - 1])
                data_target.append(chinese_words[1:random_integer])
                data_train.append(chinese_words[1:random_integer])
                data_target.append(chinese_words[2:random_integer + 1])
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


# ######################################################################################################################
class DataFrameDataset(Dataset):
    def __init__(self, content, label):
        self.content = content
        self.label = label

    def __getitem__(self, index):
        contentTensor = torch.tensor(self.content[index], dtype=torch.int64)
        labelTensor = torch.tensor(self.label[index], dtype=torch.int64)
        return contentTensor.to(device), labelTensor.to(device)

    def __len__(self):
        return len(self.label)


def label_pipeline(_label):
    return vocab.get_stoi()[_label]


def text_pipeline(_text):
    text = []
    for text1 in _text:
        text = text + [vocab.get_stoi()[text1]]
    return text


def collate_batch(batch):
    label_list, text_list = [], []
    truncate = Truncate(max_seq_len=20)  # 截断
    pad = PadTransform(max_length=20, pad_value=vocab['<pad>'])
    for (_text, _label) in batch:
        label = text_pipeline(_label)
        label = truncate(label)
        label = torch.tensor(label, dtype=torch.int64)
        label = pad(label)
        label_list.append(label)
        text = text_pipeline(_text)
        text = truncate(text)
        text = torch.tensor(text, dtype=torch.int64)
        text = pad(text)
        text_list.append(text)

    label_list = torch.vstack(label_list)
    text_list = torch.vstack(text_list)
    return text_list, label_list


# - 我们首先定义评估模型的代码。
# - 模型的评估和模型的训练逻辑基本相同，唯一的区别是我们只需要forward pass，不需要backward pass
def evaluate(model, dev_iter):
    model.eval()  # 预测模式
    total_loss = 0.
    total_count = 0.

    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        # 这里不管是训练模式还是预测模式，h层的输入都是初始化为0，hidden的输入不是model的参数
        # 这里model里的model.parameters()已经是训练过的参数。
        for i, batch in enumerate(dev_iter):
            data, target = batch[0], batch[1]
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)  # 截断计算图
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += 1  # 得到该batch的单词总数，*data.szie进行拆包，得到seq_len * batch = 50*32
            total_loss += loss.item()   # 一次batch总的损失

    loss = total_loss / 1000  # 整个验证集总的损失除以总的单词数
    model.train()
    return loss


USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = 20  # 20
EMBEDDING_SIZE = 500
MAX_VOCAB_SIZE = 50000

vocab = build_vocab_from_iterator(data_Process(), min_freq=5, specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab["<unk>"])
print(vocab.get_itos()[:50])
print(vocab.get_stoi()["<unk>"], vocab.get_stoi()["<pad>"], vocab.get_stoi()["我"])
# sys.exit()
VOCAB_SIZE = len(vocab)
# build_vocab可以根据我们提供的训练数据集来创建最高频单词的单词表，max_size帮助我们限定单词总量。
print("vocabulary size: {}".format(len(vocab)))
# 这里越靠前越常见，增加x了两个特殊的token，<unk>表示未知的单词，<pad>表示padding。
print("------" * 10)

data_train, data_target = load_data()
train_iter = DataFrameDataset(data_train[:-2000], data_target[:-2000])
val_iter = DataFrameDataset(data_train[-2000:-1000], data_target[-2000:-1000])
test_iter = DataFrameDataset(data_train[-1000:], data_target[-1000:])

train_loader = DataLoader(train_iter, batch_size=16, shuffle=True)
val_loader = DataLoader(val_iter, batch_size=16, shuffle=True)
test_loader = DataLoader(test_iter, batch_size=8, shuffle=True)

hidden_size = 400
model = RNNModel("GRU", VOCAB_SIZE, EMBEDDING_SIZE, hidden_size, 2, dropout=0.5)
if USE_CUDA:
    model = model.cuda()


# #### 定义一个function，把一个hidden state和计算图之前的历史分离。
# 将当前隐藏层hidden与之前进行截断
def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        # 这个是GRU的截断，因为只有一个隐藏层
        return hidden.detach()  # 截断计算图，h是全的计算图的开始，只是保留了h的值
    else:  # 这个是LSTM的截断，有两个隐藏层，格式是元组
        return tuple(repackage_hidden(v) for v in hidden)


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
# 每调用一次这个函数，lenrning_rate就降一半，0.5是一半

GRAD_CLIP = 1.
NUM_EPOCHS = 2
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)  # 隐藏层初始化,得到hidden初始化后的维度

    for i, batch in enumerate(train_loader):
        data, target = batch[0], batch[1]
        hidden = repackage_hidden(hidden)
        # 语言模型每个batch的隐藏层的输出值是要继续作为下一个batch的隐藏层的输入的,
        # 因为batch数量很多，如果一直往后传，会造成整个计算图很庞大，反向传播会内存崩溃。
        # 所有每次一个batch的计算图迭代完成后，需要把计算图截断，只保留隐藏层的输出值。
        # 不过只有语言模型才这么干，其他比如翻译模型不需要这么做。
        # repackage_hidden自定义函数用来截断计算图的。

        optimizer.zero_grad()
        output, hidden = model(data, hidden)  # output = (50,32,50002)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # loss_fn((num, class), num)
        # output.view(-1, VOCAB_SIZE) = (1600,50002)
        # target.view(-1) =(1600),关于pytorch中交叉熵的计算公式请看下面链接。
        loss.backward()
        # 防止梯度爆炸，设定阈值，当梯度大于阈值时，更新的梯度为阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i % 1000 == 0 and i != 0:
            print("epoch", epoch, "/iter", i, "loss", loss.item())

        if i % 10000 == 0 and i != 0:
            val_loss = evaluate(model, val_loader)  # 在val_iter上进行验证

            if len(val_losses) == 0 or val_loss < min(val_losses):
                print("best model, val_loss: ", val_loss)
                torch.save(model.state_dict(), "lm-best.th")
            else:  # 否则loss没有降下来，需要优化
                print("not best model, val_loss: ", val_loss)
                scheduler.step()  # 自动调整学习率
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                # 学习率调整后需要更新optimizer，下次训练就用更新后的
            val_losses.append(val_loss)  # 保存每10000次迭代后的验证集损失损失

# 加载保存好的模型参数
best_model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, hidden_size, 2, dropout=0.5)
if USE_CUDA:
    best_model = best_model.cuda()
# 把训练好的模型参数load到best_model里
torch.save(best_model.state_dict(), 'gnn_language_model.pth')

# ### 使用最好的模型在valid数据上计算perplexity
val_loss = evaluate(best_model, val_loader)
print("perplexity: ", np.exp(val_loss))
# 这里不清楚语言模型的评估指标perplexity = np.exp(val_loss)


# ### 使用最好的模型在测试数据上计算perplexity
test_loss = evaluate(best_model, test_loader)
print("perplexity: ", np.exp(test_loss))
# # 使用训练好的模型生成一些句子。
# hidden = best_model.init_hidden(1)  # batch_size = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
# # (1,1)表示（seq_len, batch_size）输出格式是1行1列的2维tensor，VOCAB_SIZE表示随机取的值小于VOCAB_SIZE=50002
# # 我们input相当于取的是一个单词
# words = []
# for i in range(100):
#     output, hidden = best_model(input, hidden)
#     # output.shape = 1 * 1 * 50002
#     # hidden = (2 * 1 * 1000, 2 * 1 * 1000)
#     word_weights = output.squeeze().exp().cpu()
#     # .exp()的两个作用：一是把概率更大的变得更大，二是把负数经过e后变成正数，下面.multinomial参数需要正数
#     word_idx = torch.multinomial(word_weights, 1)[0]  # 得到的是概率最大的单词的索引,[0]相当于拿到数
#     # 按照word_weights里面的概率随机的取值，概率大的取到的机会大。
#     # torch.multinomial看这个博客理解：https://blog.csdn.net/monchin/article/details/79787621
#     # 这里如果选择概率最大的，会每次生成重复的句子。
#     input.fill_(word_idx)  # 预测的单词index是word_idx，然后把word_idx作为下一个循环预测的input输入
#     word = TEXT.vocab.itos[word_idx]  # 根据word_idx取出对应的单词
#     words.append(word)
# print(" ".join(words))
