import re

input_file = "data/merged_divide_file.txt"
output_file = "data/processed_output.txt"

# 逐行读取文件，删除非中文字符，然后写入新文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 使用正则表达式匹配中文字符
        cleaned_line = re.sub('[^a-zA-Z0-9 \u4e00-\u9fa5]+', ' ', line)
        # 将数字替换为'Num数'
        cleaned_line = re.sub(r'\d+', '#数字', cleaned_line)
        # 将英文字符替换为'N名词'
        cleaned_line = re.sub(r'[a-zA-Z]+', '#名词', cleaned_line)
        # 将处理后的行写入新文件
        outfile.write(cleaned_line)
        outfile.write('\n')

print(f"处理完成，文件保存在: {output_file}")
