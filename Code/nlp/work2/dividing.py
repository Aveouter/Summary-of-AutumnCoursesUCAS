import jieba
import os

# 载入自定义词典（可选）
# jieba.load_userdict("your_custom_dict.txt")
output_file = "data/merged_output.txt"
divide_file = "data/merged_divide_file.txt"


def merge_files(output_file):
    folder_path = "data/data"
    # 逐个读取文件并合并内容
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 列出文件夹中的所有文件
        file_list = os.listdir(folder_path)
        # 遍历文件列表
        i = 0
        for file_name in file_list:  # change it to read the txt
            print('NO.' + str(i) + ' is  processing...')
            i += 1
            file_path = os.path.join(folder_path, file_name)
            # 检查是否为文件
            if os.path.isfile(file_path):
                # 读取当前文件内容
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                # 将当前文件内容写入合并文件
                outfile.write(content)
    # 输出合并完成的文件路径
    print(f"合并完成，文件保存在: {output_file}")


def merge_divide_file(output_file):
    folder_path = "data/data"
    # 逐个读取文件并合并内容
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 列出文件夹中的所有文件
        file_list = os.listdir(folder_path)
        # 遍历文件列表
        i = 1
        for file_name in file_list[8:9]:  # change it to read the txt
            print('Dividing... NO.' + str(i) + ' is  processing...')
            i += 1
            file_path = os.path.join(folder_path, file_name)
            # 检查是否为文件
            if os.path.isfile(file_path):
                # 读取当前文件内容
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                # 将当前文件内容写入合并文件
                seg_list = jieba.cut(content, cut_all=False)
                result_str = " ".join(seg_list)
                outfile.write(result_str)
    # 输出合并完成的文件路径
    print("dividing completed ")


# merge_files(output_file)

merge_divide_file(divide_file)
