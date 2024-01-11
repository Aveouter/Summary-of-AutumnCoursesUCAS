import pycrfsuite
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 加载训练数据
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = line.split()
                data.append(words)
    return data

def clearing_data(train_data,train_label):
    a=[]
    b=[]
    for i,j in zip(train_data,train_label):
        if len(i) != len(j):
            continue
        else:
            a.append(i)
            b.append(j)
    print('dataloaded')
    return a,b


# 特征提取函数
def extract_features(sentence, i):
    word = sentence[i][0]
    features = {
        'word': word,
        'prev': '' if i == 0 else sentence[i-1][0],
        'next': '' if i >= len(sentence) - 1 else sentence[i+1][0],
        'prev2': '' if i <= 1 else sentence[i-1][0],
        'next2': '' if i >= len(sentence) - 2 else sentence[i+2][0],
        'prev_prev_word': '' if i <= 1 else sentence[i-1][0] + '' if i == 0 else sentence[i-1][0] + word ,
        'word_next_next': word + '' if i >= len(sentence) - 1 else sentence[i+1][0] + '' if i >= len(sentence) - 2 else sentence[i+2][0],
        'prev_word_next': '' if i == 0 else sentence[i-1][0] + word + '' if i >= len(sentence) - 1 else sentence[i+1][0],
        'prev_word': '' if i == 0 else sentence[i-1][0] + word,
        'word_next': word + '' if i >= len(sentence) - 1 else sentence[i+1][0],
    }
    return features

# 标签提取函数
def load_labels(file_path):
    load_data(file_path)

# 训练CRF模型
def train_crf_model(train_data,train_label, model_path):
    trainer = pycrfsuite.Trainer(verbose=False)
    for sentence, labels in tqdm(zip(train_data, train_label), total=len(train_data), desc="Training"):
        features = [extract_features(sentence, i) for i in range(len(sentence))]
        trainer.append(features, labels)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,   # coefficient for L2 penalty
        'max_iterations': 50,
        'feature.possible_transitions': True # include transitions that are possible, but not observed
    })
    trainer.train(model_path)

# 评估CRF模型性能
def evaluate_crf_model(test_data, test_label, model_path):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    y_true = []
    y_pred = []
    for sentence, labels in zip(test_data, test_label):
        features = [extract_features(sentence, i) for i in range(len(sentence))]
        y_true.extend(labels)
        y_pred.extend(tagger.tag(features))
    
    # 打印分类报告
    print(classification_report(y_true, y_pred))

    # 生成混淆矩阵   这个图有点没意义
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',xticklabels=['B', 'M', 'E', 'S'], yticklabels=['B', 'M', 'E', 'S'])
    plt.title("归一化混淆矩阵")
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.show()

if __name__ == "__main__":
    # 加载训练数据
    train_data = load_data('seq.txt')
    train_label = load_data('label_seq.txt')

    train_data,train_label = clearing_data(train_data,train_label)   
    # 划分训练集和测试集
    train_data, test_data ,train_label, test_label= train_test_split(train_data, train_label,test_size=0.2, random_state=42)

    # 训练CRF模型
    model_path = 'crf_model.crfsuite'
    train_crf_model(train_data, train_label,model_path)

    # 评估CRF模型性能
    # 设置matplotlib支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
    evaluate_crf_model(test_data, test_label, model_path)

    # 测试
    seq = '我爱北京天安门'
    seq = list(seq)
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    features = [extract_features(seq, i) for i in range(len(seq))]
    print(seq)
    print(tagger.tag(features))