## 定义损失函数
import tensorflow as tf

## 使用稀疏分类交叉熵
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# 定义损失函数
def loss_function(real, pred, pad_index):
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    loss_ = loss_object(real, pred)
    ## 转换
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
