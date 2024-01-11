# we want to clawer the data from the following website;
# http://novel.tingroom.com/shuangyu
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import numpy as np
import random
import string
import csv
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import pdfplumber


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_paper_info_list(number=20):
    print("start scrapying……")
    # USE 181个网页
    paper_title_list = []
    paper_pdf_list = []
    i = [x for x in range(1, 180)]
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
    for num in i:
        url = 'http://novel.tingroom.com/jingdian/list_1_{}.html'.format(num)  # redefined
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        for x in soup.find_all('h6', {"class": ['yuyu']}):  ##找到所有div(块),其中id为detail
            for y in x.find_all('a'):
                paper_title_list.append(y.text)  ##得到文本, strip()去除空格
                paper_pdf_list.append('https://novel.tingroom.com' + y['href'])
        print("现在执行的页数", num)
    items = []

    for _, paper_info in enumerate(zip(paper_pdf_list, paper_title_list)):
        print(paper_info, _)
        items.append([paper_info[0], paper_info[1].replace("\n", ""),
                      # paper_info[2].text.split(':')[1].strip(" ").replace("\n", ""), paper_info[3][0]
                      ])
    df = pd.DataFrame(index=list(range(1, len(items) + 1)),
                      columns=['文章下载链接', '文章标题'], data=items)
    df.to_csv('ListofBook/' + 'EnglishArticle' + '.csv')
    print("Generate summary Excel successfully!")
    return items


def get_text(url):
    try:
        news = ''
        r = requests.get(url)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.title.text.strip()  # 获得标题
        title += "!!!! "  # 区分标记
        news += title
        for x in soup.find_all('div', {'id': ['detail']}):  # 找到所有div(块),其中id为detail
            for y in x.find_all('p'):
                text = y.text.strip()  # 得到文本, strip()去除空格
                news += text
        return news
    except:
        print("爬取失败")


def download(target_paper_list):
    """
    下载目标文章到本地
    """
    print("Downloading new paper ing....")
    temp_paper_list = []
    for i in target_paper_list:
        i[2] = re.sub(r'[\\/:"*?<>|]', '', i[2])
        if os.path.exists('FileofBook/{}.txt'.format(i[2])):
            pass
        else:
            a = re.findall(r'\d+', i[1])
            r = requests.get('http://novel.tingroom.com/novel_down.php?aid={}&dopost=txt'.format(a[0]))
            a = 'http://novel.tingroom.com/novel_down.php?aid={}&dopost=txt'.format(a[0])
            print(a)
            with open('FileofBook/{}.txt'.format(i[2]), 'wb') as f:
                f.write(r.content)
            print("finish downloading {}".format(i[2]))
            temp_paper_list.append(i[2])
            time.sleep(2)
    return temp_paper_list


def select_allpaper():
    """
    搜索相关主题并包含相应关键词的论文
    """
    # 打开文本文件并逐行读取
    with open('ListofBook/englishArticle.csv', 'r', encoding='utf-8') as file:
        # 创建CSV读取器
        items = []
        csv_reader = csv.reader(file)
        # 遍历CSV文件中的每一行
        for row in csv_reader:
            # 在这里，row是一个包含CSV文件中一行数据的列表
            # 你可以将row添加到数组中，从而得到一个包含所有行的数组
            items.append(row)
    char_to_check = ["?", "/"]
    target_paper_list = []
    for item in items:
        # print(_)
        # if (char_to_check[0] in item[1]) == 0 and (char_to_check[1] in item[1]) == 0:
        target_paper_list.append(item)
    return target_paper_list


#
if __name__ == '__main__':
    # mkdir('ListofBook/')
    # mkdir('FileofBook/')
    # ListOfItems = get_paper_info_list()

    target_paper_list = select_allpaper()
    temp_paper_list = download(target_paper_list[1:])
    print(temp_paper_list)
