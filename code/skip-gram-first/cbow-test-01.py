import os
import jieba
import random
import numpy as np
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
import keras.backend as K
from collections import Counter
## 设定超参数
word_size = 128
