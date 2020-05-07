import os
import jieba
import random
import numpy as np
import re
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
import keras.backend as K
from collections import Counter

## 设定超参数

## 设定的词向量维度
word_size = 128
## 窗口大小
window_size = 5
## 随机负采样的样本数
nb_negative = 50
## 设定频数少于10的抛弃掉
min_count = 10
## 循环次数
nb_epoch = 2

## 语料库的路径
## 先把下载下来的news_tensite_xml.full.zip放在linux服务器上解压，出来一个.dat文件,但是不能用，这个我踩了很多坑.
## 直接用 iconv命令转换成utf-8编码：cat news_tensite_xml.dat | iconv -f gbk -t utf-8 -c > 0.txt，然后再处理这个txt
file_path = 'E:\\nlp-dataset\\sougou_news\\'
file_output_path = file_path + "datasets\\"
source_file_path = file_path + '0.txt'
stopwords_file_path = 'E:\\nlp-dataset\\stopwords.dat'
## 每个文件多少个doc
doc_size = 6000


## 1. 将这个近 2.2G的0.txt文件切割成小文件,稍微计算一下，有130万个doc,拆成10M一个文件，每个文件大约6000个doc.下面开始拆分
def split_files():
    with open(source_file_path, 'r', encoding='utf-8') as f:
        doc_end = '</doc>'
        content = f.read()
        r = re.compile(doc_end)
        doc_list = r.split(content)
        size = len(doc_list)
        ## 这样分出来的第一个元素是空字符串,而且每个元素都没有</doc>,需要自己加上
        spilt_file = open(file_output_path + "0.txt", 'a', encoding='utf-8')
        for index in range(size):
            doc = doc_list[index]
            ## 去掉第一个，因为是空字符串
            if doc != '':
                doc += doc_end
                print(doc)
                spilt_file.writelines(doc)
                if (index + 1) % 6000 == 0:
                    spilt_file.close()
                    file_index = (index + 1) / 6000
                    spilt_file = open(file_output_path + str(file_index) + ".txt", "a", encoding='utf-8')
        spilt_file.close()


## 抽取预料内容
def squeeze_content():
    ## 语料库
    corpus = []
    ## 词库
    words = []
    with open(file_output_path + "0.txt", 'r', encoding='utf-8') as f:
        ## 提取content内容
        content = f.read()
        r = re.compile(r'<content>(.*)?</content>')
        doc_list = re.findall(r, content)
        for i in range(len(doc_list)):
            line = doc_list[i]
            if(line != ''):
                ## 去掉停用词，标点符号
                stopwords = read_file(stopwords_file_path)
                p = [word for word in jieba.cut(line) if word not in stopwords]
                corpus += p
                words += p
    words = dict(Counter(words))
    total = sum(words.values())
    print(total)


def read_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    #去除行末的换行符，否则会在停用词匹配的过程中产生干扰
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines


#抽取预料,统计词频
squeeze_content()


