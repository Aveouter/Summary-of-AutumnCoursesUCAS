import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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

class Hmm:
    def __init__(self):
        self.trans_p = {'S': {}, 'B': {}, 'I': {}, 'E': {}}
        self.emit_p = {'S': {}, 'B': {}, 'I': {}, 'E': {}}
        self.start_p = {'S': 0, 'B': 0, 'I': 0, 'E': 0}
        self.state_num = {'S': 0, 'B': 0, 'I': 0, 'E': 0}
        self.state_list = ['S', 'B', 'I', 'E']
        self.line_num = 0
        self.smooth = 1e-6

    def train(self, train_data, train_label, save_model=False):
        print("正在训练模型……")
        start_time = time.thread_time()
        for sentence, labels in tqdm(zip(train_data, train_label), total=len(train_data), desc="Training"):
            for i, s in enumerate(labels):
                self.state_num[s] = self.state_num.get(s, 0) + 1.0
                self.emit_p[s][sentence[i]] = self.emit_p[s].get(
                    sentence[i], 0) + 1.0
                if i == 0:
                    self.start_p[s] += 1.0
                else:
                    last_s = labels[i - 1]
                    self.trans_p[last_s][s] = self.trans_p[last_s].get(
                        s, 0) + 1.0
    
        # 归一化：
        self.start_p = {
            k: (v + 1.0) / (self.line_num + 4)
            for k, v in self.start_p.items()
        }
        self.emit_p = {
            k: {w: num / self.state_num[k]
                for w, num in dic.items()}
            for k, dic in self.emit_p.items()
        }
        self.trans_p = {
            k1: {k2: num / self.state_num[k1]
                 for k2, num in dic.items()}
            for k1, dic in self.trans_p.items()
        }
        end_time = time.thread_time()
        print("训练完成，耗时 {:.3f}s".format(end_time - start_time))
        # 保存参数
        if save_model:
            parameters = {
                'start_p': self.start_p,
                'trans_p': self.trans_p,
                'emit_p': self.emit_p,
                'state_list': self.state_list,
                'line_num': self.line_num,
                'smooth': self.smooth,
                'state_num': self.state_num
            }
            jsonstr = json.dumps(parameters, ensure_ascii=False, indent=4)
            param_filepath = "./HmmParam_Token.json"
            with open(param_filepath, 'w', encoding='utf8') as jsonfile:
                jsonfile.write(jsonstr)

    def viterbi(self,sentence):
        '''
        :param sentence:  输入的句子
        :param tag_list:  所有的tag
        :return: prob预测的最大的概率 bestpath 预测的tag序列
        '''
        tag_list = ['S', 'B', 'I', 'E']
        V = [{}] #tabular
        path = {}
        backpointers = []
        for y in tag_list: #init
            V[0][y] = self.start_p[y] * (self.emit_p[y].get(sentence[0],0.00000001))
            path[y]=y
        backpointers.append(path)
        for t in range(1,len(sentence)):
            V.append({})
            newpath = {}
            path = {}
            for y in tag_list:
                (prob,state ) = max([(V[t-1][y0] * self.trans_p[y0].get(y,0.00000001) * self.emit_p[y].get(sentence[t],0.00000001) ,y0) for y0 in tag_list])
                V[t][y] =prob
                path[y]=state
            backpointers.append(path)
        (prob, state) = max([(V[len(sentence) - 1][y], y) for y in tag_list])
        best_path=[]
        best_path.append(state)
        for pathi in reversed(backpointers):
            state = pathi[state]
            best_path.append(state)
        best_path.pop()
        # Pop off the start tag (we dont want to return that to the caller)
        best_path.reverse()
        return best_path
            
    def cut(self, text):
        """根据 viterbi 算法获得状态，根据状态切分句子
        Args:
            text (string): 待分词的句子
        Returns:
            list: 分词列表
        """
        state = self.viterbi(text)
        cut_res = []
        begin = 0
        for i, ch in enumerate(text):
            if state[i] == 'B':
                begin = i
            elif state[i] == 'E':
                cut_res.append(text[begin:i + 1])
            elif state[i] == 'S':
                cut_res.append(text[i])
        return cut_res

    # 评估CRF模型性能
def evaluate_crf_model(test_data, test_label):
    # Load the model from JSON file
    with open('HmmParam_Token.json', 'r', encoding='utf-8') as json_file:
        model_json = json_file.read()
        hmm_loaded = Hmm()
        hmm_loaded.__dict__ = json.loads(model_json)
        print("模型加载成功！")
    y_true = []
    y_pred = []
    for sentence, labels in zip(test_data, test_label):
        y_true.extend(labels)
        sentence = ''.join(sentence)
        y_pred.extend(hmm_loaded.viterbi(sentence))
    
    # 打印分类报告
    print(classification_report(y_true, y_pred))

    # 生成混淆矩阵   这个图有点没意义
    # 设置matplotlib支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
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
    print(len(train_data),len(train_label))
    train_data,train_label = clearing_data(train_data,train_label)   
    # 划分训练集和测试集
    train_data, test_data ,train_label, test_label= train_test_split(train_data, train_label,test_size=0.2, random_state=42)

    hmm = Hmm()
    hmm.train(train_data, train_label, save_model=True)

    # 评估模型性能
    # evaluate
    evaluate_crf_model(test_data, test_label)

    # Test the loaded model
    cutres = hmm.cut('我爱北京天安门')
    viterbi_res = hmm.viterbi('我爱北京天安门')
    print(cutres)
    print(viterbi_res)
