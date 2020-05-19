from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


## 读取每一行数据放到list
def read_segmented_sentences_to_list(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ## 去掉空格
            line = line.strip()
            words.append(line)
    f.close()
    return words


##保存一行一行的文本(按空格分开[gensim LineSentence要求])到磁盘，供后续使用
## 参数line_data_list 一行一行的数据组成的list
## 参数dist_path 保存的磁盘路径
def persist_line_data(line_data_list, dist_path):
    print("begin to persist line_data to " + str(dist_path) + " ....")
    with open(dist_path, "w", encoding='utf-8') as f:
        for line in line_data_list:
            f.write(line.strip() + "\n")
    f.close()
    print("persist line_data " + str(dist_path) + " successfully")


def build_word2vector_model(line_data_list_file_path, model_persist_path):
    ##Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory in alphabetical order by filename.
    print("begin to train w2v model...")
    w2v = Word2Vec(sentences=PathLineSentences(line_data_list_file_path), size=256, window=5, sg=1, min_count=2, iter=8)
    print("train success....")
    print("begin to save model to : " + model_persist_path)
    w2v.wv.save_word2vec_format(model_persist_path, binary=True)
    print("save model w2v successfully....")


if __name__ == '__main__':
    ## todo:
    pass
