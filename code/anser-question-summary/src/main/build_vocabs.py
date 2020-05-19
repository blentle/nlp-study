from collections import Counter


## 读取切完词保存的路径，每个词按空格分隔，读取到的每个词放在list中返回:["word1","word2"]
def read_words_to_list(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ## 去掉空格
            line = line.strip()
            ## 去掉空字符串
            p = [w for w in line.split() if w.strip() != '']
            words += p
    f.close()
    return words


## 构建正向、反向字典 word2id, id2word,并去掉低频词,频数低于 min_count的词会被去掉
def build_vocabs(train_x_cut_list, train_y_cut_list, test_x_cut_list, min_count):
    print("begin to build vocabs...")
    words_all = []
    words_all += train_x_cut_list
    words_all += train_y_cut_list
    words_all += test_x_cut_list
    ## 统计词频数,封装成字典
    words_dict = dict(Counter(words_all))
    ## 去掉低频次
    words_dict = {w: cnt for w, cnt in words_dict if cnt > min_count}
    print("去掉低频次以后还剩下 " + str(len(words_dict)) + " 单词")
    ## 按每个词的频数降序排列
    p = sorted(words_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ## 排列后的序号作为id，生成words2id的字典和id2words的字典
    words2id = {}
    id2words = {}
    for index, dict_data in enumerate(p):
        ## 软件强迫症，id 从1 开始
        words2id[index + 1] = dict_data[0]
        id2words[dict_data[0]] = index + 1
    ## 也返回词的频数字典
    print("build vocabs successfully...")
    words2id = sorted(words2id.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    id2words = sorted(id2words.items(), key=lambda kv: (kv[0], kv[1]), reverse=False)
    return words2id, id2words, p


##保存 words2id、id2words、p到磁盘，供后续使用
## 参数dict_data 注意这里是列表类型，不是字典类型
## 参数dist_path 保存的磁盘路径
def persist_dict_data(dict_data_list, dist_path):
    print("begin to persist dict_data to " + str(dist_path) + " ....")
    with open(dist_path, "w", encoding='utf-8') as f:
        for index, data in enumerate(dict_data_list):
            f.writelines(data[0] + " " + str(data[1]) + "\n")
    f.close()
    print("persist dict_data " + str(dist_path) + " successfully")


if __name__ == '__main__':
    train_x_cut_list = read_words_to_list("../../temp/train-x.txt")
    train_y_cut_list = read_words_to_list("../../temp/train-y.txt")
    test_x_cut_list = read_words_to_list("../../temp/test-x.txt")
    words2id, id2words, fre_p = build_vocabs(train_x_cut_list, train_y_cut_list, test_x_cut_list, 2)
    persist_dict_data(words2id, "../../temp/words_to_id.txt")
    persist_dict_data(id2words, "../../temp/id_to_words.txt")
    persist_dict_data(fre_p, "../../temp/words-fre.txt")
