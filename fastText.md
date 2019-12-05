#### 任务安排

> Gensim（generate similarity）是一个简单高效的自然语言处理Python库，用于抽取文档的语义主题（semantic topics）
> FastText是Facebook开发的一款快速文本分类器，提供简单而高效的文本分类工具
>
> 阅读学习所有有关文档
> 遵照指引，使用gensim/fasttext 完成3个不同类别的任务
>
> 尝试分析gensim的文件结构，列举如果打算深入剖析这些系统，自己的知识结构和技能还有哪些欠缺，写在实验报告内



## Gensim 学习笔记

`Gensim`是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。支持包括`TF-IDF`, `LSA`, `LDA`, `Word2Vec`在内的多种主题模型算法，支持分布式训练，提供了相似度计算、信息检索等一些常用的API接口




#### 基本概念

- 语料（Corpus）：一组原始文本的集合，

  在Gensim中，Corpus通常是一个可迭代的对象（比如列表）。每一次迭代返回一个可用于表达文本对象的稀疏向量。



**我们直接看Gensim实际处理过程比较清晰:**



#### 1.训练预料的预处理

将文档中原始的字符文本转换成Gensim模型所能理解的稀疏向量的过程。

**分词, 清洗->特征列表**

原始文本:(每一行表示一个文档,实际情况中可能很长)

```
text_corpus = [ 
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
]
```

得到每一篇文档所谓的特征列表。例如，在词袋模型中，文档的特征就是其包含的word：

```
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
]
```

**稀疏向量化**

利用Gensim对上述特征建立索引词典, 然后利用索引词典把文本变成向量,如词袋模型下的稀疏向量

```bash
vectors = [[(0, 1), (1, 1), (2, 1)],
 [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
 [(2, 1), (5, 1), (7, 1), (8, 1)],
 ]
 # Gensim会把此处向量的处理转化成流式处理,即一次处理一个文档yield一个稀疏变量
```



#### 2. 选择模型,建立向量空间

利用上一步的稀疏向量作为输入,选择不同的模型,生成不同的向量空间. 比如这学期一直在讨论的TF-IDF模型

```python
tfidf = models.TfidfModel(corpus) #corpus是上步稀疏向量的迭代器
```

此处提高效率的方法: 还是转为流式输出(迭代器),而且需要多次使用的话,先序列化到磁盘上.

#### 3.计算文档相似度

已经得到了每个文档对应的向量了,计算相似度就比较简单了(比如直接用余弦值)



---

#### Gensim参考资料

[Gensim官方文档](https://pypi.org/project/gensim/)

[Gensim Core Concepts](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html)

[Gensim Tutorial](https://www.machinelearningplus.com/nlp/gensim-tutorial/)

[Gensim训练模型, 词向量的保存与调用](https://juejin.im/post/5d527ecf518825056144e33f)

[15分钟入门Gensim](https://zhuanlan.zhihu.com/p/37175253)

---

---





## fastText 学习笔记

fastText是Facebook于2016年开源的一个词向量计算和文本分类工具. [官方文档](https://fasttext.cc/docs/en/supervised-tutorial.html)

#### 安装

在linux命令行下进行安装编译.

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```



---

### 快速上手实践

> 选用官方文档的实例: we build a classifier which automatically classifies stackexchange **questions about cooking** into one of several possible tags, such as `pot`, `bowl` or `baking`!
>



**获取数据,**

> fastText要求数据的labels以 ____label____ 格式进行标注(各两个下划线)
>
> e.g. ____label____sauce  ____label____cheese How much does potato starch affect a cheese sauce recipe?   (多label, sauce & cheese)

共15404条语句; 切分为训练集(12404)和验证集(3000)

```bash
$ wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz 
$ tar xvzf cooking.stackexchange.tar.gz
$ wc cooking.stackexchange.txt
15404  169582 1401900 cooking.stackexchange.txt
$ head -n 12404 cooking.stackexchange.txt > cooking.train
$ tail -n 3000 cooking.stackexchange.txt > cooking.valid
```

**训练分类器**

调用有监督训练命令, 训练, 得到分类器`model_cooking`

```bash
$ ./fasttext supervised -input cooking.train -output model_cooking
```



**测试验证集**

默认为单label; 会输出precision和recall

```bash
$ ./fasttext test model_cooking.bin cooking.valid 5 # 多label
N	3000
P@5	0.0679 # precision
R@5	0.147 # recall
```

> 文档说明precision和recall区别的例子还挺直观的!



**分类器简单改进** 

- 文本清洗: 用正则表达式去各种符号并统一大小写

- 增加训练次数: `epoch`值默认只有5 (增加参数`-epoch`即可)

- 调整`learning_rate`: 增加参数`-lr`即可

```bash
$ ./fasttext supervised -input cooking.train -output model_cooking -epoch 25 -lr 1.0
```

​	> **注意:** 学习率设置太大,可能会报`NaN`错误.



- **使用n-grams模型**

> 上述默认的方式是基于普通的词袋模型(可以理解为1 gram),即丢弃了词与词之间的顺序; 词序在很多情况下是非常重要的,特别类似于情感分析这种文本分类问题;

使用参数`-wordNgrams N` 即可 (建议取值范围`1-5`)

```bash
$ ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25 -wordNgrams 2
$  ./fasttext test ./pyqTest/model_cooking_processed.bin cooking.valid
N	3000
P@1	0.565  # 提升明显
R@1	0.244
```



**加快训练速度**

增加参数: `-bucket 200000 -dim 50 -loss hs` 会大大加快训练速度!

- hierarchical softmax: 使用树的层级结构替代扁平化的标准Softmax, 后讲;

- bucket: 通过Hash桶解决n-grams词汇量快速增大的问题, 后讲;

  



**多标签分类的思路**

- 方法一: 仍使用softmax实现,只不过对于结果增加一个概率域值,所有大于该域值的类都认为是这个样本的label. 

  这种方法不理想, 原因是你需要保证所有概率和为1;

- 方法二: 把多分类问题转换成一个个的二分类问题,即对于每个类, 都单独判断此样本是否属于它.

  使用参数`-loss one-vs-all` 或`ova`即可实现





---

### 原理了解

​		**核心思想**: 将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。(中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类)

​	 	**效果:** 在文本分类任务中，fastText（浅层网络）往往能取得和深度网络相媲美的精度，却在训练时间上比深度网络快许多数量级。在标准的多核CPU上， 能够训练10亿词级别语料库的词向量在10分钟之内，能够分类有着30万多类别的50多万句子在1分钟之内。

#### 功能

- 生成词向量

- 文本分类



#### 模型

类似于cbow模型, 可以理解为浅层(3层)的神经网络模型 (下图摘自 [fastText原理及实践](https://zhuanlan.zhihu.com/p/32965521))

![](./images/fastText_model.jpg)

注意：此架构图没有展示词向量的训练过程。可以看到，和[CBOW](https://www.zhihu.com/question/44832436/answer/266068967))一样，fastText模型也只有三层：输入层、隐含层、输出层（Hierarchical Softmax），输入都是多个经向量表示的单词，输出都是一个特定的target，隐含层都是对多个词向量的叠加平均。

不同的是，CBOW的输入是目标单词的上下文，fastText的**输入是多个单词及其n-gram特征**，这些特征用来表示单个文档；CBOW的输入单词被onehot编码过，fastText的输入特征是被embedding过；CBOW的输出是目标词汇，fastText的输出是文档对应的类标。

fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。



>  **字符级别的n-grams**的好处
>
> - 对于低频词生成的词向量效果会更好, 因为它们的n-gram可以和其它词共享;
>
> - 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。



#### 部分内容详解

##### 分层softmax (hierarchical softmax)

基本思想是使用树的层级结构替代扁平化的标准Softmax，使得在计算 ![[公式]](https://www.zhihu.com/equation?tex=P%28y%3Dj%29) 时，只需计算一条路径上的所有节点的概率值，无需在意其它的节点. 计算复杂度从|K|降低到log|K|

![](./images/hierarchical_softmax.jpg)

详细原理可以观看 [视频教程](https://www.youtube.com/watch?v=B95LTf2rVWM)



#### Bucket (TODO)

Fasttext采用了Hash桶的方式，把所有的n-gram都哈希到buckets个桶中，哈希到同一个桶的所有n-gram共享一个embedding vector。[详解链接](https://vel.life/fastText体验笔记/)





### 文本分类应用:

**应用一**: 针对THUNews(长文本新闻数据)进行分类

**应用二**: 对于头条新闻标题无监督学习词向量

详细过程见文本分类应用.pdf  (可以直接运行对应Jupyter文件. )



---

#### fastText参考资料

[fastText官方文档](https://fasttext.cc/docs/en/supervised-tutorial.html)

[fastText中文文档](http://fasttext.apachecn.org/#/)

[fastText原理及实践](https://zhuanlan.zhihu.com/p/32965521)

[CBOW模型详解](https://www.zhihu.com/question/44832436/answer/266068967)

[fastText体验笔记](https://vel.life/fastText体验笔记/)

[在分类中如何处理训练集中不平衡问题](https://blog.csdn.net/heyongluoyao8/article/details/49408131)



---



### 知识结构和能力 思考

> Gensim is not a technique itself. Gensim is a NLP package that contains efficient implementations of many well known functionalities for the tasks of topic modeling such as [tf–idf](https://en.wikipedia.org/wiki/Tf–idf), [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), [Latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis).

如果要深入剖析的话,需要补充

- 理论知识

  理解Gensim实现的例如LDA, LSA等模型原理

- 数据处理能力

  其实模型并不是最重要的, 如何获取干净有价值的数据才是关键. 

  要提升爬取数据, 清洗数据的能力

- 工程实践能力

  像是fastText这种文本分类器,其实简单功能用Keras或者pytorch几行代码就能搭出来,但是效果就差太多太多了.  







---

---

## 额外笔记

### python操作文件

**下载及解压**

```python
!curl -O http://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip
import zipfile
with zipfile.ZipFile("THUCNews.zip", 'r') as zip_ref:
    zip_ref.extractall()
```

但是对于windows来的.zip文件,会有乱码啊