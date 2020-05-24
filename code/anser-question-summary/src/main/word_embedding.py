from gensim.models import KeyedVectors


##从制定的路径加载模型,构建词向量，并保存
def build_word_emmbeding(w2v_model_bin_path, embedding_dict_path):
    model = KeyedVectors.load_word2vec_format(w2v_model_bin_path, binary=True)
    with open(embedding_dict_path, "w", encoding="utf-8") as f:
        ## python 有没有像java那样有stream的api ,充分利用多核 cpu 去循环遍历，这样串行遍历太慢了？
        for word in model.vocab:
            word_matrix = model.get_vector(word)
            ##因为得到的是numpy array类型,这里需要转换
            word_matrix_list = word_matrix.tolist()
            matrix_str = [str(vec) for vec in word_matrix_list]
            f.writelines(word + " " + " ".join(matrix_str) + "\n")
    f.close()


if __name__ == '__main__':
    model_persist_path = "../../temp/w2v-model-bin"
    ## 这个生成文件太大了，就不上传了.
    embedding_path = "../../temp/word_embedding.txt"
    build_word_emmbeding(model_persist_path, embedding_path)
