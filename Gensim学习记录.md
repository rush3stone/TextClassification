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
  .....
]
```

得到每一篇文档所谓的特征列表。例如，在词袋模型中，文档的特征就是其包含的word：

```
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
....
]
```

**稀疏向量化**

利用Gensim对上述特征建立索引词典, 然后利用索引词典把文本变成向量,如词袋模型下的稀疏向量

```bash
vectors = [[(0, 1), (1, 1), (2, 1)],
 [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
 [(2, 1), (5, 1), (7, 1), (8, 1)],
 ...
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



### 知识结构和能力 思考

> Gensim is not a technique itself. Gensim is a NLP package that contains efficient implementations of many well known functionalities for the tasks of topic modeling such as [tf–idf](https://en.wikipedia.org/wiki/Tf–idf), [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), [Latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis).

如果要深入剖析的话,需要补充

- 理论知识

  理解Gensim实现的例如LDA, LSA等模型原理

- 数据处理能力

  提升清洗数据,得到



---

#### 参考资料

[Gensim官方文档](https://pypi.org/project/gensim/)

[Gensim Core Concepts](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html)

[Gensim Tutorial](https://www.machinelearningplus.com/nlp/gensim-tutorial/)

[Gensim训练模型, 词向量的保存与调用](https://juejin.im/post/5d527ecf518825056144e33f)

[15分钟入门Gensim](https://zhuanlan.zhihu.com/p/37175253)

---

---

