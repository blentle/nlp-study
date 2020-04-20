## 1.使用全模式
import jieba
seg_list = jieba.cut("我来到品友的第一天", cut_all=True)
print(" ".join(seg_list)) ## 得到: 我 来到 品 友 的 第一 第一天 一天

## 2.使用精确模式(默认的),去掉cut_all
p = jieba.cut("我来到品友的第一天")
print(" ".join(p)) ## 得到: 我 来到 品友 的 第一天

## 3.使用搜索引擎模式
q = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(" ".join(q)) ## 得到: 小明 硕士 毕业 于 中国 科学 学院 科学院 中国科学院 计算 计算所 ， 后 在 日本 京都 大学 日本京都大学 深造

## 4.使用一些新词
r = jieba.cut("新研发的产品名为一网搜索。")
print(" ".join(r)) ## 得到:  新 研发 的 产品 名为 一网 搜索 。
## 发现以往搜索分开了，分的不对，这时候想达到我们想要的结果: "新 研发 的 产品 名为 一网搜索 。" 就得创建我们自定义的词典
jieba.load_userdict(r'E:\后场nlp学习\code\skip-gram-first\user-jieba.txt')
s = jieba.cut("新研发的产品名为一网搜索。")
print("Default Mode: " + " ".join(s))  ## 得到了: 新 研发 的 产品 名为 一网搜索 。
