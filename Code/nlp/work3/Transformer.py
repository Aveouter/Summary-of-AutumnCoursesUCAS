# Function: 用transformer模型进行中文分词
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from CRF import load_data,clearing_data
from evaluate import load
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, BertTokenizerFast
import seaborn as sns

# 读取数据      
def read_data(file_path1,file_path2):
    # 加载训练数据
    train_data = load_data(file_path1)
    train_label = load_data(file_path2)
    # 数据清洗
    train_data,train_label = clearing_data(train_data,train_label)   
    # 划分训练集和测试集
    train_data, test_data ,train_label, test_label= train_test_split(train_data, train_label,test_size=0.2, random_state=42)
    return train_data, train_label, test_data, test_label

label_list = ['S', 'B', 'I', 'E']
id2tag = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}
tag2id = {'S': 0, 'B': 1, 'I': 2, 'E': 3}
plt.ioff()

def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # 创建全由-100组成的矩阵
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        if len(doc_labels) >= 510:  # 防止异常
            doc_labels = doc_labels[:510]
        # 设置第一个偏移位置为0，第二个偏移位置不为0的标签（offset-mapping中 [0,0] 表示不在原文中出现的内容）
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels 
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels

class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
 
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
 
    def __len__(self):
        return len(self.labels)

def compute_metrics(p):
    metric = load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
 
    # 不要管-100那些，剔除掉
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
 
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def dataset_bulit():
    # 加载数据集    
    train_data, train_label, test_data, test_label = read_data('seq.txt','label_seq.txt')
    # train_data = train_data[:1000]
    # train_label = train_label[:1000]
    # test_data  = test_data[:1000]
    # test_label = test_label[:1000]
    # 文本预处理
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    train_encodings = tokenizer(train_data, is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=512)  # is_split_into_words表示已经分词好了
    val_encodings = tokenizer(test_data, is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=512)
    train_labels = encode_tags(train_label, train_encodings, tag2id)
    val_labels = encode_tags(test_label, val_encodings, tag2id)
    # print(train_labels[0])
    train_encodings.pop("offset_mapping")  # 训练不需要这个
    val_encodings.pop("offset_mapping")
    train_dataset = NerDataset(train_encodings, train_labels)
    val_dataset = NerDataset(val_encodings, val_labels)  ## 这两个是喂给模型的训练和评估的，后面的confusionmatrix和classificationreport要单独用原始数据
    return train_dataset,val_dataset, test_data, test_label

def load_AND_train_model(train_dataset,val_dataset):
    model_dir = 'hfl/rbt3'
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=4,  # 4分类
                                                            ignore_mismatched_sizes=True,  # 不加载权重
                                                            id2label=id2tag,
                                                            label2id=tag2id
                                                            ) 
    training_args = TrainingArguments(
    output_dir='./output',  # 模型输出路径
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.01,  # 权重衰减
    logging_steps=10,  # 日志记录的步长(loss,学习率)
    evaluation_strategy="epoch",  # 评估策略为训练完一个epoch之后进行评估
    save_strategy="epoch",  # 保存策略同上
    save_total_limit=3,  # 最多保存数量
    load_best_model_at_end=True,  # 设置训练完成后加载最优模型
    metric_for_best_model="f1",  # 指定最优模型的评估指标为f1
    fp16=True  # 半精度训练（提高训练速度）
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
    )
    # 训练与评估
    print('开始训练')
    trainer.train()
    print('开始评估')
    result = trainer.evaluate()
    print(result)

    
def ws_predict(input_str, tokenizer, model):
    input_char = list(input_str.replace(' ', ''))  # 文本去空格
    input_tensor = tokenizer(input_char, is_split_into_words=True, padding=True, truncation=True,
                                return_offsets_mapping=True, max_length=512, return_tensors="pt")
    offsets = input_tensor["offset_mapping"]
    ignore_mask = offsets[0, :, 1] == 0
    input_tensor.pop("offset_mapping")  # 不剔除的话会报错
    outputs = model(**input_tensor)
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    res = ''
    idx = 0
    while idx < len(predictions):
        if ignore_mask[idx]:  # 跳过分隔符
            idx += 1
            continue
        while idx < len(predictions) - 1 and model.config.id2label[predictions[idx]] == f"I":  # 如果下一个是'i'
            res += input_char[idx - 1]
            idx += 1
        if idx < len(predictions) - 1 and model.config.id2label[predictions[idx]] == f"B":
            res += '  '
            res += input_char[idx - 1]
            idx += 1
        elif idx < len(predictions) - 1 and model.config.id2label[predictions[idx]] == f"E":
            res += input_char[idx - 1]
            res += '  '
            idx += 1
        elif idx < len(predictions) - 1 and model.config.id2label[predictions[idx]] == f"S":
            res += '  '
            res += input_char[idx - 1]
            res += '  '
            idx += 1
    return res

# 评估TRANSFORMER模型性能
def evaluate_Transformer_model(test_data, test_label, tokenizer, model):
    y_true = []
    y_pred = []
    for sentence, labels in zip(test_data, test_label):
        if len(sentence) != len(labels) or len(sentence) == 0 or len(labels) == 0 :
            continue
        input_char = sentence # 文本去空格
        input_tensor = tokenizer(input_char, is_split_into_words=True, padding=True, truncation=True,
                                    return_offsets_mapping=True, max_length=512, return_tensors="pt")
        offsets = input_tensor["offset_mapping"]
        input_tensor.pop("offset_mapping")  # 不剔除的话会报错
        outputs = model(**input_tensor)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()
        label_list = ['S', 'B', 'I', 'E']
        y_true.extend(labels)
        tab =['S']*len(labels)
        for idx,i in enumerate(predictions[1:-1]):
            tab[idx] = label_list[i]
        if len(tab)!= len(labels):
            print('error')
            exit()
        y_pred.extend(tab)


    # 打印分类报告
    print(set(y_pred))
    print(set(y_true))
    print(classification_report(y_true, y_pred))

    # 设置matplotlib支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',xticklabels=['B', 'I', 'E', 'S'], yticklabels=['B', 'I', 'E', 'S'])
    plt.title("归一化混淆矩阵")
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.show()


if __name__ == "__main__":
    train_dataset,val_dataset,test_data, test_label = dataset_bulit()
    # load_AND_train_model(train_dataset,val_dataset)

    # 加载模型和评估
    model_dir = './output/checkpoint-2435'
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    evaluate_Transformer_model(test_data, test_label, tokenizer, model)

    # 测试
    input_str = '我爱北京天安门' 
    res = ws_predict(input_str, tokenizer, model)
    print(res)

