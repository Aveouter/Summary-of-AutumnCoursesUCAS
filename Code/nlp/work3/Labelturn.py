'''
使用其将文本语料转化为标签语料
'''

import os

filepath = 'ChineseCorpus199801.txt'


def to_label(line):
    def label(words):
        label = []
        for word in words:
            if len(word) == 1:
                label.append('S')
            else:
                label.append('B')
                if len(word)-2 != 0:
                    for i in range(len(word)-2):
                        label.append('I')
                label.append('E')

        return label
    if len(line):
        words = line.split()
        rewords= []
        for word in words[1:]:
            word = word.replace('[', '')
            word = word.replace(']', '')
            word = word.split('/')[0]  # Remove the part-of-speech tag
            rewords.append(word)  # Append the modified word to the words list
        return rewords,label(rewords)
        
            


def read_file(filepath, n=0):
    with open(filepath, 'r', encoding='GB2312', errors='ignore') as file:
        i = 0
        for line in file:
            line = line.strip()
            if not len(line):
                continue
            seq, label_seq = to_label(line)
            
            # Save seq in a new file
            with open('seq.txt', 'a', encoding='UTF-8') as seq_file:
                # print(seq)
                seq = list( "".join(seq))
                # print(seq)
                seq_file.write(' '.join(seq) + '\n')
            
            # Save label_seq in a new file
            with open('label_seq.txt', 'a', encoding='UTF-8') as label_seq_file:
                label_seq_file.write(' '.join(label_seq) + '\n')
            
            i = i + 1
            if n == 1 and i >= 2:
                break

def delete_file_content(filepath):
    with open(filepath, 'w') as file:
        file.truncate(0)


if __name__ == "__main__":
    delete_file_content('seq.txt')
    delete_file_content('label_seq.txt')
    read_file(filepath,0)