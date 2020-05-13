import csv
import jieba
from collections import Counter

## 去掉没用的词，发现说的频率很高
REMOVE_WORDS = ['[', ']', '语音', '图片', '说']


## 分词,返回分完词后的list
def spilt(sentence):
    return jieba.lcut(sentence)


# 停用词文件的格式要求:每一行存储一个词
def load_stop_words(path):
    ## 去重存储
    dataset = set()
    with open(path, 'r', encoding='utf-8') as f:
        for word in f:
            ##去除首尾空格
            p = word.strip()
            if p is not None:
                dataset.add(p)
    return dataset


# 对于给定的词的列表, 过滤没有用的词(根据训练预料观察获取)
def filter_unused_words(target_list, unused_list):
    words_list = [word for word in target_list if word not in unused_list]
    return words_list


## 初步处理数据
def clean_data_draft(train_file_path, test_file_path):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    ## 处理训练集
    with open(train_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # 去掉第一行header,直接取数据
            if i == 0:
                continue
            ## 去掉每一行前三列没用的维度,如品牌，车型
            question = row[3]
            dialog = row[4]
            report = row[5].strip()
            ##过滤 report（y）是空的数据
            if report is None:
                continue
            ## quest和dialog拼接到一起组成 x
            if question is None:
                question = ''
            if dialog is None:
                dialog = ''
            x = question + str(dialog)
            y = report
            train_x.append(x)
            train_y.append(y)
    f.close()
    ## 处理测试集
    with open(test_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # 去掉第一行header,直接取数据
            if i == 0:
                continue
            ## 去掉每一行前三列没用的维度,如品牌，车型
            question = row[3]
            dialog = row[4]
            if question is None:
                question = ''
            if dialog is None:
                dialog = ''
            x = question + str(dialog)
            test_x.append(x)
    f.close()
    return train_x, train_y, test_x, test_y


## 对指定的训练数据集分词,保存数据文件
def clean_data_save_split_words(train_sentence_data, stop_words_path, target_file_path):
    data_list = []
    with open(target_file_path, 'w', encoding='utf-8') as f:
        i = 0
        for sentence in train_sentence_data:
            if not isinstance(sentence, str):
                sentence = str(sentence)
            segment = spilt(sentence)
            ## 去掉停用词
            list_without_stop_words = [word for word in segment if word not in load_stop_words(stop_words_path)]
            ## 过滤预料无关的词，如语音，图片等
            list_without_unused_words = filter_unused_words(list_without_stop_words, REMOVE_WORDS)
            ## 去掉空字符串
            target_list = [word for word in list_without_unused_words if word.strip() is not None and word != '']
            data_list += target_list
            ##过滤完以后，写入文件
            f.writelines(' '.join(target_list))
            i = i + 1
            print("完成第" + str(i) + "条数据写入.........")
    f.close()
    return data_list


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = clean_data_draft("E:/nlp-dataset/AutoMaster_TrainSet.csv",
                                                        "E:/nlp-dataset/AutoMaster_TestSet.csv")
    ## 生成分完词的文件，以一个空格隔开
    stop_words_file_path = 'E:/nlp-dataset/stopwords.dat'
    print("begin to save train-x..........")
    train_x_list = clean_data_save_split_words(train_x, stop_words_file_path, 'E:/nlp-dataset/gen/train-x.txt')
    print("end to save train-x..........")
    print("begin to save train-y..........")
    train_y_list = clean_data_save_split_words(train_y, stop_words_file_path, 'E:/nlp-dataset/gen/train-y.txt')
    print("end to save train-y..........")
    print("begin to save test-x..........")
    test_x_list = clean_data_save_split_words(test_x, stop_words_file_path, 'E:/nlp-dataset/gen/test-x.txt')
    print("end to save test-x..........")
    print("begin to save test-y..........")
    test_y_list = clean_data_save_split_words(test_y, stop_words_file_path, 'E:/nlp-dataset/gen/test-y.txt')
    print("end to save test-y..........")
    ## 合到一起
    train_x_list += train_y_list
    train_x_list += test_x_list
    ## 统计词频
    word_fre_dict = Counter(train_x_list)
    ## 按词频降序排序
    p = sorted(word_fre_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 写文件: word index格式
    with open('E:/nlp-dataset/gen/vocab.txt', 'w', encoding='utf-8') as f:
        i = 0
        for index, dic in enumerate(p):
            f.writelines(str(index) + "\r" + dic[0] + "\n")
            i += 1
            print('已写入第' + str(index) + "个词，词频是:" + str(dic[1]))
    f.close()
