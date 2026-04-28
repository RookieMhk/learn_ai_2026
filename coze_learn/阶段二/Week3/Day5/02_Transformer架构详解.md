# Transformer架构详解

> Week 3 Day 5 核心学习材料 | 现代AI的基石架构

---

## 一、自注意力机制（Self-Attention）的数学原理

### 1.1 核心思想
自注意力机制让序列中的每个位置都能关注到序列中的所有其他位置，从而建立全局依赖关系。这解决了RNN长距离依赖失效的问题。

### 1.2 缩放点积注意力（Scaled Dot-Product Attention）

**核心公式**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**各符号含义**：
- **Q (Query)**: 查询向量，代表当前位置"在寻找什么"
- **K (Key)**: 键向量，代表每个位置"自己是什么"
- **V (Value)**: 值向量，代表每个位置的"实际内容"
- **d_k**: 键向量的维度

**计算流程图**：

```
输入序列 → 生成 Q, K, V → 计算 QK^T (相似度矩阵)
         → 除以 √d_k (缩放) → Softmax归一化 → 加权求和
```

### 1.3 为什么要缩放？

当 d_k 较大时，点积的方差会增大，导致Softmax函数进入饱和区域（梯度接近0）。除以 √d_k 可以保持点积结果的方差稳定，确保梯度在合理范围内。

```python
# 示意代码
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    # 计算点积注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（用于解码器的因果注意力）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

---

## 二、多头注意力机制（Multi-Head Attention）

### 2.1 为什么需要多头？

单头注意力只能关注单一的语义关系。多头注意力让模型能够**并行学习不同类型的依赖关系**：

- 有的头关注**局部语法**（相邻词的关系）
- 有的头关注**全局逻辑**（远距离的语义关联）
- 有的头关注**实体关系**（主语、宾语等）

### 2.2 数学公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**参数说明**：
- h: 注意力头的数量
- W_i^Q, W_i^K, W_i^V: 每个头的投影矩阵
- W^O: 最终输出投影矩阵

### 2.3 典型配置

| 模型 | d_model | h (头数) | d_k (每头维度) |
|------|---------|----------|----------------|
| 原论文 | 512 | 8 | 64 |
| BERT-Large | 1024 | 16 | 64 |
| GPT-3 | 12288 | 96 | 128 |

---

## 三、位置编码（Positional Encoding）

### 3.1 为什么需要位置编码？

Transformer完全基于注意力机制，**没有循环结构**，无法天然感知序列中元素的位置信息。位置编码为模型提供了"第几个词"的信息。

### 3.2 正弦/余弦位置编码（原论文方案）

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    生成正弦/余弦位置编码
    seq_len: 序列长度
    d_model: 模型维度
    """
    PE = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    PE[:, 0::2] = np.sin(position * div_term)  # 偶数维度用sin
    PE[:, 1::2] = np.cos(position * div_term)  # 奇数维度用cos
    
    return PE
```

**特点**：
- ✅ 无需训练，可泛化到任意长度
- ✅ 每个位置有唯一编码
- ✅ 相对距离关系可通过线性变换得到
- ❌ 长序列（>1024）位置区分度下降

### 3.3 可学习位置编码（主流方案）

2026年主流模型（LLaMA-3、Qwen等）采用可学习的位置编码：

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(seq_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len)
        position_ids = torch.arange(x.size(1), device=x.device)
        return self.position_embeddings(position_ids)
```

### 3.4 进阶方案

| 方案 | 特点 | 代表模型 |
|------|------|----------|
| **RoPE** (旋转位置编码) | 通过旋转矩阵注入位置信息，适合高效推理 | LLaMA, Qwen |
| **ALiBi** (线性偏置注意力) | 通过线性偏置替代位置编码，缓解长度外推问题 | BLOOM |
| **YaRN** | 基于RoPE的扩展，优化长上下文 | LLaMA-2 (部分版本) |

---

## 四、Encoder-Decoder 结构

### 4.1 整体架构

```
                    Transformer
    ┌─────────────────────────────────────────┐
    │                                         │
    │   输入          N层          输出        │
    │   序列    →    Encoder   →             │
    │                                         │
    │   输出          N层                     │
    │   序列    →    Decoder   →  预测结果    │
    │                                         │
    └─────────────────────────────────────────┘
```

### 4.2 Encoder（编码器）

**每层结构**：
1. Multi-Head Self-Attention（多头自注意力）
2. Add & Norm（残差连接 + 层归一化）
3. Feed-Forward Network（前馈网络）
4. Add & Norm

**关键特点**：
- 双向注意力：每个位置可以关注到序列中的所有其他位置
- 一次性处理整个序列

### 4.3 Decoder（解码器）

**每层结构**：
1. Masked Multi-Head Self-Attention（掩码自注意力）
2. Add & Norm
3. Cross Attention（交叉注意力，Q来自Decoder，K,V来自Encoder）
4. Add & Norm
5. Feed-Forward Network
6. Add & Norm

**关键特点**：
- 因果掩码：确保生成时只能看到前面的词（不能"偷看"未来）
- 交叉注意力：让解码器能够关注到编码器的输出

### 4.4 前馈网络（FFN）

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()  # 或 GELU, SwiGLU 等
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
```

**典型配置**：d_ff = 4 × d_model

---

## 五、Transformer在GPT/BERT中的应用

### 5.1 三种架构范式对比

| 架构类型 | 代表模型 | 注意力类型 | 核心任务 |
|----------|----------|------------|----------|
| **Encoder-Decoder** | T5, BART, 原版Transformer | 双向编码 + 单向解码 | 机器翻译、摘要 |
| **Encoder-Only** | BERT, RoBERTa | 双向自注意力 | 文本分类、NER、MLM |
| **Decoder-Only** | GPT系列, LLaMA, Claude, Qwen | 单向（掩码）自注意力 | 文本生成、对话 |

### 5.2 BERT：Encoder-Only 的 MLM 预训练

**任务**：遮盖语言模型（Masked Language Model）
```
输入：The [MASK] cat sat on the mat
目标：预测 [MASK] → "little"
```

**特点**：
- 双向注意力，可以利用上下文
- 适合理解任务（分类、实体识别）

### 5.3 GPT系列：Decoder-Only 的 NTP 预训练

**任务**：下一个Token预测（Next Token Prediction）
```
输入："The little cat"
目标：预测下一个词 → "sat"
```

**特点**：
- 单向注意力（因果掩码）
- 适合生成任务（对话、写作、代码）
- 扩展性好，成为大语言模型的主流架构

### 5.4 2026年主流架构演进

```
┌─────────────────────────────────────────────────────┐
│              Decoder-Only 一统天下                  │
│                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐  │
│  │  GPT    │  │  LLaMA  │  │      Qwen          │  │
│  │  Claude │  │  DeepSeek│ │      GLM           │  │
│  └─────────┘  └─────────┘  └─────────────────────┘  │
│                                                     │
│  关键技术：                                         │
│  • MoE (混合专家) → 万亿参数，稀疏激活              │
│  • RoPE/ALiBi → 长上下文                           │
│  • GQA (分组查询注意力) → 推理效率                  │
│  • FlashAttention → 显存优化                       │
└─────────────────────────────────────────────────────┘
```

---

## 六、2026年架构前沿突破

### 6.1 何恺明 & LeCun：动态Tanh（DyT）替代归一化层

**发现**：归一化层（LayerNorm）可以用一个简化的缩放tanh函数替代：

```python
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias
```

**意义**：DyT比RMSNorm在H100上更快，有潜力降低训练和推理成本。

### 6.2 注意力机制的可解释性突破

**Transformer算法核心发现**（2026年）：
- 不同随机种子训练的Transformer会收敛到相同的**低维算法核心**
- 这些核心对于任务执行是**必要且充分**的
- 例如：GPT-2中主谓一致任务由**单一轴**控制

---

## 七、关键概念速查表

| 概念 | 公式/要点 | 作用 |
|------|-----------|------|
| 缩放点积注意力 | softmax(QK^T/√d_k)V | 计算加权上下文 |
| 多头注意力 | 并行h个注意力头 | 学习多类型依赖 |
| 位置编码 | PE(pos,2i)=sin, PE(pos,2i+1)=cos | 注入位置信息 |
| 残差连接 | x + sublayer(x) | 缓解梯度消失 |
| 层归一化 | (x - μ) / σ * γ + β | 稳定训练 |

---

## 八、学习资源推荐

### 必读论文
1. **《Attention Is All You Need》** (2017) - Transformer原论文
2. **《BERT: Pre-training》** - Encoder-Only代表
3. **《GPT-3》** - Decoder-Only代表
4. **《FlashAttention》** - 效率优化必读
5. **《RoPE》** - 旋转位置编码

### 实践资源
- PyTorch官方Transformer实现
- Hugging Face Transformers库
- 斯坦福CS224N课程

---

*本文件为Week 3 Day 5学习材料的核心理论部分，建议配合《注意力机制实现.py》进行代码实践。*
