# NLP基础任务指南

> Week 4 Day 2 - 零基础学习NLP核心任务

---

## 一、文本预处理

文本预处理是NLP的基础，决定了后续模型的上限。

### 1.1 分词（Tokenization）

**什么是分词？**
将文本切分成最小的语义单元（词/子词）。

```python
# 英文分词 - 简单空格分词
text = "Natural language processing is amazing!"
tokens = text.split()
print(tokens)  # ['Natural', 'language', 'processing', 'is', 'amazing!']

# 中文分词 - 使用jieba
import jieba
text_cn = "自然语言处理是一门有趣的技术"
words = jieba.lcut(text_cn)
print(words)  # ['自然语言', '处理', '是', '一门', '有趣', '的', '技术']
```

**分词策略对比**：
| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 空格分词 | 英文 | 简单 | 无法处理标点、缩写 |
| jieba | 中文 | 支持词典 | 歧义处理不佳 |
| BPE | 多语言 | 处理未登录词 | 需要训练 |
| WordPiece | 多语言 | 更细粒度 | 词汇表大 |

### 1.2 文本清洗

```python
import re
import string

def clean_text(text):
    # 转小写
    text = text.lower()
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除多余空格
    text = ' '.join(text.split())
    return text

text = "<p>Check out https://example.com for MORE info!</p>"
print(clean_text(text))  # check out  for more info
```

### 1.3 标准化

**词形还原（Lemmatization）**：
```python
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running"))  # run
print(lemmatizer.lemmatize("better", pos='a'))  # good
```

**词干提取（Stemming）**：
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem("running"))   # run
print(stemmer.stem("connection")) # connect
```

---

## 二、词向量表示

### 2.1 为什么需要词向量？

计算机无法直接理解文字，需要将文本转为数值向量。

```
文本 "cat" → [0.2, -0.1, 0.8, ...]  # 300维向量
```

### 2.2 Word2Vec

**核心思想**：词的语义由其上下文决定（"You shall know a word by the company it keeps"）

```python
from gensim.models import Word2Vec

# 训练语料
sentences = [
    ['machine', 'learning', 'is', 'powerful'],
    ['deep', 'learning', 'achieves', 'great', 'results'],
    ['natural', 'language', 'processing', 'is', 'interesting']
]

# 训练模型
model = Word2Vec(sentences, vector_size=100, window=2, min_count=1)

# 获取词向量
vector = model.wv['learning']
print(f"向量维度: {vector.shape}")  # (100,)

# 相似词查询
similar = model.wv.most_similar('learning', topn=2)
print(similar)  # [('deep', 0.3), ('machine', 0.2)]
```

### 2.3 GloVe

**核心思想**：融合全局统计信息（共现矩阵）与局部上下文

```python
import gensim.downloader as api

# 下载预训练GloVe词向量
model = api.load("glove-wiki-gigaword-100")

# 查看词向量
vector = model['computer']
print(f"维度: {vector.shape}")

# 类比推理
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result[0])  # queen
```

### 2.4 FastText

**核心思想**：用子词（subword）增强对未登录词的处理能力

```python
from gensim.models import FastText

sentences = [['human', 'interface', 'computer'], ['survey', 'user', 'machine']]
model = FastText(sentences, vector_size=4, window=3, min_count=1)

# 即使拼写错误也能得到结果
vector = model.wv['humman']  # 模拟错误拼写
print(vector)
```

### 2.5 词向量可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# 训练模型
sentences = [['king', 'queen', 'man', 'woman', 'prince', 'princess'],
             ['apple', 'fruit', 'banana', 'orange']]
model = Word2Vec(sentences, vector_size=10, min_count=1)

# 提取词向量
words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word] for word in words]

# PCA降维
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.annotate(word, (coords[i, 0], coords[i, 1]))
plt.title('Word Embeddings Visualization')
plt.show()
```

---

## 三、文本分类任务

### 3.1 任务定义

```
输入：一段文本
输出：预定义的类别标签
```

**应用场景**：
- 垃圾邮件检测（ spam / not_spam ）
- 情感分析（ positive / negative / neutral ）
- 主题分类（科技 / 体育 / 娱乐 / ...）

### 3.2 文本分类流程

```
原始文本 → 预处理 → 文本表示 → 分类模型 → 预测结果
```

### 3.3 传统方法 vs 深度学习方法

**传统方法（TF-IDF + 机器学习）**：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 构建分类器
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# 训练数据
X_train = ["I love this movie", "This is bad", "Great film", "Terrible experience"]
y_train = [1, 0, 1, 0]  # 1=正面, 0=负面

# 训练
model.fit(X_train, y_train)

# 预测
predictions = model.predict(["Amazing movie!"])
print(predictions)  # [1]
```

**深度学习方法（CNN/RNN/Transformer）**：

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # 平均池化
        return self.fc(pooled)

# 使用示例
model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=2)
```

---

## 四、命名实体识别（NER）

### 4.1 任务定义

```
输入：一段文本
输出：每个词的实体类型标注

示例：
"Apple 成立于 1976年"

标注：
Apple → ORG (组织)
成立 → O (非实体)
于 → O
1976年 → DATE (日期)
```

### 4.2 常见实体类型

| 标签 | 含义 | 示例 |
|------|------|------|
| PER | 人名 | 马斯克、Alice |
| ORG | 机构名 | Apple、Google |
| LOC | 地名 | 北京、纽约 |
| DATE | 日期 | 2024年、昨天 |
| TIME | 时间 | 上午九点 |
| MONEY | 货币 | 100美元 |

### 4.3 NER实现方法

**方法1：基于规则**

```python
import re

def rule_based_ner(text):
    entities = []
    
    # 识别人名（简单示例）
    person_pattern = r'[A-Z][a-z]+ [A-Z][a-z]+'
    persons = re.findall(person_pattern, text)
    for p in persons:
        entities.append((p, 'PER'))
    
    # 识别日期
    date_pattern = r'\d{4}年'
    dates = re.findall(date_pattern, text)
    for d in dates:
        entities.append((d, 'DATE'))
    
    return entities

text = "马云 创办了 阿里巴巴 于 1999年"
print(rule_based_ner(text))
# [('马云', 'PER'), ('1999年', 'DATE')]
```

**方法2：BiLSTM-CRF（传统深度学习方法）**

```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, len(tag_to_ix))
    
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out)
        return out
```

**方法3：预训练模型（推荐）**

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "bert-base-chinese"
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 推理
text = "北京是中国的首都"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
```

### 4.4 NER评估

```python
from seqeval.metrics import classification_report

# 真实标签
y_true = [['O', 'O', 'B-LOC', 'I-LOC', 'O']]
# 预测标签  
y_pred = [['O', 'O', 'B-LOC', 'I-LOC', 'O']]

print(classification_report(y_true, y_pred))
```

---

## 五、学习资源推荐

### 5.1 经典论文

1. **Word2Vec**: "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)
2. **GloVe**: "GloVe: Global Vectors for Word Representation" (Pennington et al., 2014)
3. **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

### 5.2 开源工具

- **Hugging Face Transformers**: 预训练模型库
- **spaCy**: 工业级NLP工具，支持NER
- **AllenNLP**: 深度学习NLP研究框架
- **jieba**: 中文分词

### 5.3 在线学习

- fast.ai NLP课程
- CS224n: Natural Language Processing with Deep Learning

---

## 六、实践项目预告

在配套的《NLP实战项目.py》中，你将：
1. 实现端到端的情感分类模型
2. 训练自己的词向量
3. 使用BiLSTM进行命名实体识别

---

> 📝 笔记区域
> 
> 你今天学到了什么？
> 
> 
> 
> 
> 
