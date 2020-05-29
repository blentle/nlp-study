## todo: 理解整理中, 代码就几行，但是我理解起来好困难，要搜集各种论文，博文啃一遍.
import tensorflow as tf


class Encoder(tf.keras.Model):

    ##初始化 encoder_unit 编码器神经元的个数
    def __init__(self, vocab_size, embedding_dim_size, embedding_matrix, encoder_unit, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = encoder_unit
        ## 是否双向
        self.use_bi_gru = True
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim_size, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)
        initial_state = self.gru.get_initial_state(enc_input_embedded)
        if self.use_bi_gru:
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=initial_state * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        else:
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

        return output, enc_hidden