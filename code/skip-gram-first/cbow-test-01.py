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

##语料库的路径
file_path = 'E:\\nlp\\houchang-nlp\\nlp-study\\dataset'


## 切割.dat文件
def split():
    p = re.compile('</doc>', re.S)
    end = '</doc>'
    fileContent = open(file_path + '\\news_tensite_xml.dat', 'r', encoding='ansi').read()  # 读文件内容

    paraList = p.split(fileContent)  # 根据</doc>对文本进行切片
    # print(len(paraList))

    ## 添加到末尾
    fileWriter = open(file_path + '\\0.txt', 'a', encoding='utf-8')  # 创建一个写文件的句柄
    # 遍历切片后的文本列表
    for paraIndex in range(len(paraList)):
        t = paraList[paraIndex]
        print(t)
        fileWriter.writelines(t + '\n' + end)  # 先将列表中第一个元素写入文件中
        if (paraIndex != len(paraList)):  # 不加if这两行的运行结果是所有的</doc>都没有了，除了最后分割的文本
            fileWriter.write(end)
        if ((paraIndex + 1) % 5000 == 0):  # 5000个切片合成一个.txt文本
            fileWriter.close()
            fileWriter = open(file_path + '\\' + str((paraIndex + 1) / 5000) + '.txt', 'a')  # 重新创建一个新的句柄，等待写入下一个切片元素。注意这里文件名的处理技巧。
    fileWriter.close()  # 关闭最后创建的那个写文件句柄
    print('finished')


split()
