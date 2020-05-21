from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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
    return dist_path


def build_word2vector_model(line_data_list_file_path, model_persist_path):
    ##Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory in alphabetical order by filename.
    print("begin to train w2v model...")
    ## 这里我的机器不错，所以开了 12个线程, 5分钟不到就跑完了
    w2v = Word2Vec(sentences=PathLineSentences(line_data_list_file_path), size=256, window=5, sg=1, min_count=2,
                   iter=15,
                   workers=12)
    print("train success....")
    print("begin to persist model to : " + model_persist_path)
    w2v.wv.save_word2vec_format(model_persist_path, binary=True)
    print("persist model w2v successfully....")


if __name__ == '__main__':
    train_x_cut_lines = read_segmented_sentences_to_list("../../temp/train-x.txt")
    train_y_cut_lines = read_segmented_sentences_to_list("../../temp/train-y.txt")
    test_x_cut_lines = read_segmented_sentences_to_list("../../temp/test-x.txt")
    lines = []
    lines += train_x_cut_lines
    lines += train_y_cut_lines
    lines += test_x_cut_lines
    line_data_list_file_path = "../../temp/input-line-sentence.txt"
    model_persist_path = "../../temp/w2v-model-bin"
    persist_line_data(lines, line_data_list_file_path)
    build_word2vector_model(line_data_list_file_path, model_persist_path)
    model = KeyedVectors.load_word2vec_format(model_persist_path, binary=True)
    data_list = model.most_similar("汽车")
    print(data_list)
    # 输出结果
    # [('支持车', 0.6342968940734863), ('支持车子', 0.6292246580123901), ('支持汽车', 0.6093997955322266),
    #  ('大师车子', 0.5865792632102966), ('支持我', 0.5858114957809448), ('支持你好', 0.5830122232437134),
    #  ('支持今天', 0.581800103187561), ('支持有没有', 0.5750285387039185), ('大师请问', 0.5671405792236328),
    #  ('支持您好', 0.56513911485672)]
    ## 感觉效果没有想象的好
