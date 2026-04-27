# Week 4 详细任务清单

> RNN与序列模型 · NLP基础任务 · 预训练模型使用

---

## 📅 Week 4 学习概览

### 阶段定位
- **Week 3** 完成了深度学习基础（神经网络、CNN、Transformer）
- **Week 4** 进入NLP核心领域：序列建模与预训练模型应用

### 学习目标
1. 掌握RNN/LSTM/GRU的原理与实现
2. 理解序列到序列(Seq2Seq)架构
3. 学会使用预训练模型（BERT、GPT等）
4. 完成至少2个NLP实战项目

---

## 📋 每日任务分解

---

### Day 1：RNN基础与循环神经网络

**主题**：理解循环神经网络的核心思想

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 理论学习 | RNN原理：时序展开、隐藏状态、参数共享 | 笔记 |
| 代码实践 | 手动实现一个简单RNN前向传播 | `rnn_basics.py` |
| 代码实践 | 使用PyTorch实现RNN层 | `pytorch_rnn.py` |
| 理解检测 | RNN的梯度流动与梯度消失问题 | 思维导图 |

**核心知识点**：
- RNN如何处理变长序列
- 隐藏状态（Hidden State）的作用
- 时间步展开与参数共享

**实践目标**：能够使用PyTorch的`nn.RNN`构建简单序列分类器

---

### Day 2：LSTM与GRU

**主题**：解决RNN长期依赖问题

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 理论学习 | LSTM：门控机制（遗忘门、输入门、输出门） | 原理图解 |
| 理论学习 | GRU：简化版门控（更新门、重置门） | 对比笔记 |
| 代码实践 | 从零实现LSTM前向传播 | `lstm_from_scratch.py` |
| 代码实践 | 使用PyTorch LSTM完成时间序列预测 | `lstm_timeseries.py` |
| 对比分析 | RNN vs LSTM vs GRU 性能对比实验 | 实验报告 |

**核心知识点**：
- LSTM的细胞状态(Cell State)如何传递长期信息
- 门控机制如何选择性地遗忘/记忆信息
- 为什么LSTM能缓解梯度消失

**实践目标**：能够选择合适的循环单元解决实际问题

---

### Day 3：序列到序列模型(Seq2Seq)

**主题**：Encoder-Decoder架构

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 理论学习 | Seq2Seq架构：编码器-解码器结构 | 架构图 |
| 理论学习 | 注意力机制(Attention)：让解码器关注相关输入 | 原理说明 |
| 代码实践 | 实现英译德的简单Seq2Seq模型 | `seq2seq_translation.py` |
| 代码实践 | 添加Bahdanau注意力机制 | `seq2seq_attention.py` |
| 论文阅读 | "Neural Machine Translation by Jointly Learning to Align and Translate" | 论文笔记 |

**核心知识点**：
- 编码器将变长序列压缩为固定向量
- 解码器如何自回归生成输出序列
- Attention如何解决信息瓶颈

**实践目标**：能够实现一个简单的机器翻译模型

---

### Day 4：NLP基础任务实战

**主题**：文本分类与情感分析

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 任务理解 | NLP基础任务概览（分类、序列标注、生成） | 任务分类表 |
| 数据处理 | 文本预处理：分词、去停用词、词向量 | `text_preprocessing.py` |
| 模型实现 | 使用LSTM实现情感分析 | `sentiment_lstm.py` |
| 模型实现 | 使用CNN进行文本分类 | `text_cnn.py` |
| 对比实验 | LSTM vs CNN 文本分类对比 | 实验记录 |

**核心知识点**：
- 文本的数值化表示（词袋、TF-IDF、词嵌入）
- 文本分类的常见架构选择
- 预训练词向量（Word2Vec、GloVe）的使用

**实践目标**：完成IMDB电影评论情感分类项目

---

### Day 5：预训练模型基础

**主题**：BERT与GPT入门

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 理论学习 | 预训练+微调范式 | 学习路线图 |
| 理论学习 | BERT：MLM + NSP双任务训练 | 原理笔记 |
| 理论学习 | GPT：单向语言模型 | 架构对比 |
| 代码实践 | 使用Hugging Face Transformers加载BERT | `bert_classification.py` |
| 微调实验 | 在下游任务上微调BERT | 微调记录 |

**核心知识点**：
- 预训练任务设计（MLM vs CLM）
- Transformer架构在预训练中的成功
- 如何选择合适的预训练模型

**实践目标**：能够使用Hugging Face加载并微调预训练模型

---

### Day 6：Hugging Face实战

**主题**：现代NLP开发工具链

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 工具学习 | Hugging Face Datasets加载数据集 | 使用指南 |
| 工具学习 | Hugging Face Tokenizers快速分词 | 对比实验 |
| 代码实践 | 使用BERT进行文本分类（完整流程） | `hf_bert_classifier.py` |
| 代码实践 | 使用GPT-2进行文本生成 | `gpt2_generator.py` |
| 项目实践 | 完成一个完整的NLP任务（自选） | 项目代码 |

**核心知识点**：
- Pipeline API的便捷使用
- Dataset和Tokenizer的高效处理
- Model Hub的模型选择与加载

**实践目标**：掌握现代NLP开发的标准流程

---

### Day 7：复盘与阶段总结

**主题**：Week 4闭环与Week 5规划

| 任务类型 | 内容 | 产出 |
|----------|------|------|
| 自评检测 | Week 4知识点测验 | 测验成绩 |
| 项目整理 | 整理本周代码项目 | 项目集 |
| 查漏补缺 | 薄弱点回顾与强化 | 学习记录 |
| 规划制定 | Week 5任务规划 | 计划文档 |
| 资源整理 | 优质NLP学习资源汇总 | 资源列表 |

**本周成果清单**：
- [ ] 掌握RNN/LSTM/GRU原理
- [ ] 完成Seq2Seq模型实现
- [ ] 掌握Hugging Face工具链
- [ ] 完成至少2个NLP实战项目
- [ ] 理解预训练模型范式

---

## 📚 Week 4 核心概念速览

```
Week 4 知识体系
├── 循环神经网络
│   ├── RNN基本原理
│   ├── BPTT（时间反向传播）
│   ├── 梯度消失与长期依赖
│   ├── LSTM（长短期记忆）
│   └── GRU（门控循环单元）
├── Seq2Seq架构
│   ├── Encoder-Decoder
│   ├── 注意力机制
│   └── Beam Search
├── NLP基础任务
│   ├── 文本分类
│   ├── 序列标注
│   ├── 情感分析
│   └── 机器翻译
└── 预训练模型
    ├── BERT（双向Transformer）
    ├── GPT（单向Transformer）
    ├── 预训练+微调范式
    └── Hugging Face生态
```

---

## 🎯 学习建议

### 时间分配建议
- **理论学习**：40%（理解原理）
- **代码实践**：50%（动手实现）
- **项目实战**：10%（综合应用）

### 常见问题应对
| 问题 | 建议 |
|------|------|
| RNN训练慢 | 考虑使用CUDA加速，或减少序列长度 |
| 梯度爆炸 | 使用梯度裁剪(gradient clipping) |
| 显存不足 | 减小batch size或使用梯度累积 |
| 预训练模型太大 | 使用distilbert等轻量模型 |

### 推荐学习顺序
1. Day 1-2：先理解循环神经网络原理
2. Day 3：理解Seq2Seq和Attention
3. Day 4：完成基础NLP任务
4. Day 5-6：学习预训练模型（重点！）
5. Day 7：总结复盘

---

## 📦 Week 4 产出物清单

| 文件名 | 说明 |
|--------|------|
| `rnn_basics.py` | RNN基础实现 |
| `lstm_from_scratch.py` | LSTM从零实现 |
| `seq2seq_attention.py` | 带Attention的Seq2Seq |
| `sentiment_lstm.py` | LSTM情感分析 |
| `bert_classification.py` | BERT文本分类 |
| `Week4综合测验.md` | 知识点测验 |
| `Week4知识总结.md` | 概念整理 |

---

> 🚀 完成Week 4后，你将具备独立完成NLP基础任务的能力！
