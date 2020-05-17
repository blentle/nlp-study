def build_vocabs(train_x_cut_list, train_y_cut_list, test_x_cut_list):
    pass


## 读取切完词保存的路径，每个词按空格分隔，读取到的每个词放在list中返回:["word1","word2"]
def read_data(path):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ## 去掉空格
            line = line.strip()
            ## 去掉空字符串
            p = [w for w in line.split() if w.strip() != '']
            words += p
    return words
