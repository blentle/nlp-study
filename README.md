# NLP Practice
## 1. anser-question-summary(问答摘要与推理)
### 背景
本工程是来源百度AI Studio的常规比赛.要求使用汽车大师提供的11万条 技师与用户的多轮对话与诊断建议报告 数据建立模型，模型需基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，综合考验模型的归纳总结与推断能力。汽车大师是一款通过在线咨询问答为车主解决用车问题的APP，致力于做车主身边靠谱的用车顾问，车主用语音、文字或图片发布汽车问题，系统为其匹配专业技师提供及时有效的咨询服务。由于平台用户基数众多，问题重合度较高，大部分问题在平台上曾得到过解答。重复回答和持续时间长的多轮问询不仅会花去汽修技师大量时间，也使用户获取解决方案的时间变长，对双方来说都存在资源浪费的情况。为了节省更多人工时间，提高用户获取回答和解决方案的效率，汽车大师希望通过利用机器学习对平台积累的大量历史问答数据进行模型训练，基于汽车大师平台提供的历史多轮问答文本，输出完整的建议报告和回答，让用户在线通过人工智能语义识别即时获得全套解决方案.
### 项目地址
[问答摘要与推理](https://aistudio.baidu.com/aistudio/competition/detail/3 "问答摘要与推理")

### 参考资料
+ [self attention 比较好的博客](http://jalammar.github.io/illustrated-transformer/)

## 2. multi-tag-classify(试题知识点多标签分类)

### 第一节知识点
#### NLP四大基本任务
+ 序列标注:分词、词性标注
+ 分类任务:文本分类、情感分析
+ 句子关系:问答系统、对话系统
+ 生成任务:机器翻译、文章摘要

#### 文本分类的应用场景
+ 有监督学习
+ 区分新闻类型
+ 作弊检测
+ 文章男女作者性别

#### 复习传统的机器学习分类
+ LR
+ SVM
+ NB
+ KNN

#### 深度学习文本分类
+ CNN、RNN (textCNN)
+ 深度学习预训练模型

#### 文本分类流程
文本获取 > 预处理 > 文本特征工程 > 算法选择 > 效果评估 > 持续调优 > 预测 > 模型上线

### 第三节知识点
### Transformer的过程
+ encoder-decoder
+ self-attention
+ multi-head attention
+ positional encoding
+ 解码器Self-Attention输入层与编码器区别：只允许关注输出序列中较前的位置（mask遮罩）

#### Transformer的优点
+ 不同数据的时间和空间关系做假设,可以处理一组对象
+ 层输出可以并行计算, 不像RNN那样需要序列计算
+ 可以学习长距离的依赖
+ 远距离项可以影响彼此输出，无需多步RNN或各种卷积层

#### Elmo的优点
+ 解决了一词多义
+ 一些任务上有提升

#### GPT

## 2. 使用 [rasa](https://github.com/RasaHQ/rasa) 写一个对话机器人

### 创建示例工程

#### a.安装依赖库
```shell script
    pip install rasa
```

#### b.安装rasa x 
```shell script
    pip install rasa-x --extra-index-url https://pypi.rasa.com/simple
```

#### c.创建一个工程目录, 执行下面的命令创建一个rasa 工程
```shell script
    rasa init
```

#### d.启动rasa x 
```shell script
    rasa x
```

### Action、Rule、Form、Story
