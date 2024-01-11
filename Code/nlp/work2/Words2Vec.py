import logging
import sys
import gensim.models as word2vec
import smart_open
from gensim.models.word2vec import LineSentence, logger
from tqdm import tqdm


def train_word2vec(dataset_path, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # sentences = LineSentence(smart_open.open(dataset_path, encoding='utf-8'))  # 或者用smart_open打开
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频）
    model = word2vec.Word2Vec(sentences, vector_size=100, sg=1, window=5, min_count=5, workers=4, epochs=10)
    # (iter随机梯度下降法中迭代的最大次数，sg为1是Skip-Gram模型)
    # 保存word2vec模型
    model.save("word2vec.model")
    model.wv.save_word2vec_format(out_vector, binary=False)


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语最相似的词
def calculate_most_similar(self, word):
    similar_words = self.wv.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


# 计算两个词相似度
def calculate_words_similar(model, word1, word2):
    print(model.similarity(word1, word2))


# 找出不合群的词
def find_word_dismatch(self, list):
    print(self.wv.doesnt_match(list))


def vocab_init():
    # 打开文件并读取第一列元素
    first_column_elements = []
    out_vector = 'data/Vec/corpusSegDone.vector'
    with open(out_vector, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用适当的分隔符提取第一列元素
            first_column_element = line.strip().split()[0]  # 假设使用空格分隔
            first_column_elements.append(first_column_element)
    return first_column_elements[1:]


if __name__ == '__main__':
    dataset_path = "data/processed_output.txt"
    out_vector = 'data/Vec/corpusSegDone.vector'
    train_word2vec(dataset_path, out_vector)
    model = load_word2vec_model("word2vec.model")  # 加载模型
    calculate_most_similar(model, "国家")  # 找相近词
    print(model.wv.__getitem__('国家'))  # 词向量

    ## get word to vector and L
    # 表 x 中的汉字词汇
    table_x = vocab_init()  # 请替换为实际的汉字词汇集合

    output_path = 'data/Vec/DeletedSegDone.txt'  # 替换为你的文件路径
    output_line = ''  # 初始化空字符串

    with open(dataset_path, 'r', encoding='utf-8') as input_file:
        for i, line in enumerate(input_file):
            # 提取汉字词汇（假设词汇间使用空格分隔）
            chinese_words = [word for word in line.strip().split() if
                             any('\u4e00' <= char <= '\u9fff' for char in word)]
            # 仅保留存在于表 x 中的汉字词汇
            valid_chinese_words = [word for word in chinese_words if word in table_x]
            # 将有效的汉字词汇拼接成一行
            output_line += ' '.join(valid_chinese_words) + '\n'
            print('completed processing line', i,len(input_file[0]))

    # 在第一个 with 块结束后，output_line 中已经包含了所有有效的汉字词汇
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(output_line)
        print('completed writing to output file')
