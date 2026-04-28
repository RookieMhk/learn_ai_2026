# Week 3 详细任务清单

> 生成时间：2026-04-18
> 执行周期：2026年4月19日 - 4月25日（7天）
> 学习主题：深度学习基础入门
> 总体目标：掌握神经网络核心概念，理解反向传播原理，初步使用PyTorch框架

---

## 📋 Week 3 总览

```
Week 3: 深度学习基础入门
├── 📅 Day 1 (4.19): 神经网络基础 - 神经元与感知机
├── 📅 Day 2 (4.20): 反向传播实战 - 梯度计算与参数更新
├── 📅 Day 3 (4.21): PyTorch框架入门 - 张量与自动微分
├── 📅 Day 4 (4.22): CNN图像分类实战 - LeNet实现
├── 📅 Day 5 (4.23): Transformer与注意力机制
├── 📅 Day 6 (4.24): 深度学习项目实战
└── 📅 Day 7 (4.25): Week 3 复盘与 Week 4 规划
```

### 🎯 学习目标

| 目标层级 | 具体内容 | 掌握标准 |
|----------|----------|----------|
| **知识目标** | 神经网络架构、反向传播、PyTorch API | 能够手推公式、画出网络结构图 |
| **技能目标** | 使用PyTorch构建模型、训练流程、调试技巧 | 能够独立完成MNIST分类任务 |
| **应用目标** | CNN/Transformer原理理解 | 能够解读经典论文中的模型结构 |

### ⏱️ 时间分配

```
每日学习时长建议：2-3小时
├── 理论学习：40分钟
├── 代码实践：60分钟
├── 复习巩固：20分钟
└── 笔记整理：20分钟

总周时长：14-21小时
```

---

## 📅 Day 1：神经网络基础（4月19日）

### 主题：神经元与感知机

**学习目标**：
- 理解生物神经元到人工神经元的映射
- 掌握单层感知机的原理和局限
- 理解激活函数的作用

### 核心知识点

```
1.1 神经元结构
├── 生物神经元：树突→细胞核→轴突
└── 人工神经元：输入→加权求和→激活→输出

1.2 单层感知机
├── 公式：y = f(w·x + b)
├── 决策边界：线性可分
└── 局限：无法解决XOR问题

1.3 激活函数
├── Sigmoid: 0~1, 易梯度饱和
├── Tanh: -1~1, 输出均值接近0
└── ReLU: max(0,x), 计算高效
```

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 📖 理论学习 | 观看/阅读神经网络基础讲解 | 笔记 | 40min |
| 💻 代码实现 | 使用NumPy实现单层感知机 | perceptron.py | 50min |
| 🔬 实验验证 | 在AND/OR/NOT数据集上测试感知机 | 实验报告 | 20min |
| ✍️ 复习整理 | 整理神经元到感知机的知识脉络 | 思维导图 | 30min |

### 重点代码

```python
# 单层感知机实现（NumPy版本）
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.zeros(n_inputs)
        self.bias = 0
    
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.step_function(z)
    
    def step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def train(self, X, y, epochs=100, lr=0.1):
        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                y_pred = self.forward(x_i)
                error = y_i - y_pred
                # 更新规则
                self.weights += lr * error * x_i
                self.bias += lr * error
```

### 验收标准

- [ ] 能够解释感知机无法解决XOR问题的原因
- [ ] 能够手写感知机的权重更新公式
- [ ] 能够在AND数据集上达到100%准确率

---

## 📅 Day 2：反向传播实战（4月20日）

### 主题：梯度计算与参数更新

**学习目标**：
- 理解链式法则在神经网络中的应用
- 掌握反向传播算法原理
- 能够手动推导简单网络的梯度

### 核心知识点

```
2.1 链式法则
├── 一维：dy/dx = dy/du · du/dx
└── 多维：梯度通过雅可比矩阵传播

2.2 反向传播流程
├── 前向传播：计算每层输出和损失
├── 反向传播：从后向前计算梯度
└── 参数更新：使用梯度下降法

2.3 梯度消失与爆炸
├── 原因：多层链式乘法导致数值不稳定
└── 解决：残差连接、梯度裁剪、归一化
```

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 📖 理论学习 | 理解反向传播数学推导 | 推导笔记 | 50min |
| 💻 代码实现 | NumPy实现两层神经网络+反向传播 | bpn.py | 60min |
| 🔬 实验验证 | 观察不同学习率对训练的影响 | 实验记录 | 30min |
| ✍️ 复习整理 | 绘制反向传播流程图 | 流程图 | 20min |

### 重点代码

```python
# 两层神经网络实现（NumPy版本）
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)
    
    def backward(self, X, y, y_pred):
        m = y.shape[0]
        # 输出层梯度
        dz2 = y_pred - y
        self.dW2 = np.dot(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0) / m
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_grad(self.z1)
        self.dW1 = np.dot(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0) / m
    
    def update(self, lr=0.01):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
```

### 验收标准

- [ ] 能够手推两层网络的梯度计算过程
- [ ] 能够在MNIST数据集上达到85%+准确率
- [ ] 理解梯度消失的表现和原因

---

## 📅 Day 3：PyTorch框架入门（4月21日）

### 主题：张量与自动微分

**学习目标**：
- 掌握PyTorch张量的创建和操作
- 理解自动微分(autograd)机制
- 能够使用PyTorch搭建简单网络

### 核心知识点

```
3.1 张量(Tensor)
├── 创建：torch.tensor(), torch.zeros(), torch.randn()
├── 操作：reshape, view, cat, stack
└── GPU加速：.cuda(), .to('cuda')

3.2 自动微分(autograd)
├── requires_grad=True 开启梯度追踪
├── loss.backward() 自动计算梯度
└── optimizer.step() 参数更新

3.3 nn.Module
├── 定义网络层
└── 前向传播
```

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 📖 理论学习 | PyTorch官方教程Tensors和Autograd章节 | 学习笔记 | 40min |
| 💻 代码实现 | PyTorch实现Day 2的TwoLayerNet | pytorch_nn.py | 60min |
| 🔬 实验验证 | 对比NumPy和PyTorch实现的训练速度 | 速度对比表 | 30min |
| ✍️ 复习整理 | PyTorch常用API速查表 | API速查表 | 30min |

### 重点代码

```python
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch版本的两层网络
class PyTorchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 训练循环
model = PyTorchNet(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()      # 清零梯度
        outputs = model(batch_x)   # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()            # 反向传播
        optimizer.step()           # 更新参数
```

### 验收标准

- [ ] 能够使用torch创建各种类型的张量
- [ ] 理解requires_grad的作用机制
- [ ] 能够使用PyTorch搭建并训练简单网络

---

## 📅 Day 4：CNN图像分类实战（4月22日）

### 主题：LeNet实现与理解

**学习目标**：
- 理解卷积神经网络(CNN)的基本结构
- 掌握卷积层、池化层的原理
- 能够实现LeNet-5并训练MNIST

### 核心知识点

```
4.1 卷积神经网络结构
├── 卷积层：局部连接、权重共享
├── 池化层：下采样、减少参数
└── 全连接层：分类输出

4.2 LeNet-5架构
├── Conv1 → Pool1 → Conv2 → Pool2 → FC1 → FC2 → Output
├── 32×32 → 28×28 → 14×14 → 10×10 → 5×5 → 120 → 84 → 10
└── 参数量约6万

4.3 卷积操作细节
├── Kernel/Filter: 卷积核大小
├── Stride: 步长
├── Padding: 填充
└── 感受野：输出特征图的每个点对应的输入区域
```

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 📖 理论学习 | CNN原理与LeNet结构分析 | 笔记 | 40min |
| 💻 代码实现 | PyTorch实现LeNet-5 | lenet.py | 60min |
| 🔬 实验验证 | 在MNIST上训练LeNet-5 | 训练日志 | 40min |
| ✍️ 复习整理 | 可视化卷积核和特征图 | 可视化图 | 20min |

### 重点代码

```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 1×32×32 → 6×28×28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)       # 6×28×28 → 6×14×14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 6×14×14 → 16×10×10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # 16×10×10 → 16×5×5
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 验收标准

- [ ] 能够解释卷积层与全连接层的区别
- [ ] 能够在MNIST上达到98%+准确率
- [ ] 能够可视化中间层的特征图

---

## 📅 Day 5：Transformer与注意力机制（4月23日）

### 主题：自注意力机制详解

**学习目标**：
- 深入理解Self-Attention机制
- 掌握多头注意力的原理
- 理解Transformer的基本架构

### 核心知识点

```
5.1 Self-Attention机制
├── Q, K, V 向量计算
├── 缩放点积注意力
└── 多头注意力(Multi-Head Attention)

5.2 Transformer结构
├── Encoder: Multi-Head + FFN + LayerNorm
├── Decoder: Masked Multi-Head + Cross Attention
└── 位置编码(Positional Encoding)

5.3 注意力可视化
├── 注意力权重热力图
└── 理解模型学到的依赖关系
```

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 📖 理论学习 | Attention is All You Need 核心章节 | 论文笔记 | 50min |
| 💻 代码实现 | PyTorch实现Self-Attention层 | attention.py | 60min |
| 🔬 实验验证 | 可视化文本任务的注意力权重 | 可视化结果 | 30min |
| ✍️ 复习整理 | Transformer架构思维导图 | 架构图 | 20min |

### 重点代码

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim)
        
        return self.fc_out(out)
```

### 验收标准

- [ ] 能够从零实现Self-Attention机制
- [ ] 理解多头注意力为什么要设置多个头
- [ ] 能够画出Transformer的架构图

---

## 📅 Day 6：深度学习项目实战（4月24日）

### 主题：综合项目实践

**学习目标**：
- 综合运用所学知识完成完整项目
- 掌握深度学习项目的完整流程
- 积累项目经验和技术文档

### 项目选择

从以下项目中选择一个完成：

#### 项目A：MNIST/CIFAR分类器优化
- 基于Day 4的LeNet进行优化
- 使用数据增强、正则化等技术
- 目标：CIFAR-10达到80%+准确率

#### 项目B：IMDb情感分析
- 使用TextCNN或简单Transformer
- 实现文本分类完整流程
- 目标：准确率达到85%+

#### 项目C：自定义数据集分类
- 收集并标注自己的小规模数据集
- 设计合适的网络结构
- 完成训练和评估

### 任务清单

| 任务类型 | 具体内容 | 交付物 | 建议时长 |
|----------|----------|--------|----------|
| 🔬 数据处理 | 数据收集、清洗、增强 | 数据集 | 30min |
| 💻 模型设计 | 选择/设计网络结构 | 模型代码 | 60min |
| 💻 训练调试 | 完成训练、调参 | 训练日志 | 90min |
| 📊 结果分析 | 评估指标、可视化分析 | 分析报告 | 30min |
| 📝 文档整理 | README、技术文档 | 完整项目包 | 30min |

### 验收标准

- [ ] 独立完成数据到模型的全流程
- [ ] 能够分析模型性能并提出改进方案
- [ ] 项目代码和文档规范完整

---

## 📅 Day 7：Week 3 复盘与 Week 4 规划（4月25日）

### 主题：阶段性总结与展望

### 复盘内容

```
Week 3 学习复盘清单：

□ 知识掌握
  ├── [ ] 神经网络基本结构
  ├── [ ] 反向传播原理
  ├── [ ] PyTorch基础操作
  ├── [ ] CNN原理与实现
  └── [ ] Transformer/Attention机制

□ 技能提升
  ├── [ ] 能够使用PyTorch搭建神经网络
  ├── [ ] 能够调试和优化模型训练
  └── [ ] 能够完成端到端项目

□ 项目完成
  ├── [ ] 完成至少一个深度学习项目
  └── [ ] 项目文档完整规范
```

### Week 4 预告：深度学习进阶

```
Week 4 主题：深度学习进阶与工程化

Day 1: 深度学习优化技巧 - 归一化、Dropout、正则化
Day 2: 模型评估与调试 - 诊断工具、超参数调优
Day 3: 迁移学习 - 预训练模型、微调策略
Day 4: 计算机视觉进阶 - ResNet、EfficientNet
Day 5: 循环神经网络 - RNN、LSTM、GRU
Day 6: 项目实战 - 完整模型开发流程
Day 7: Week 4 复盘
```

---

## 📊 Week 3 学习进度追踪表

| 日期 | Day | 理论学习 | 代码实践 | 实验验证 | 笔记整理 | 完成度 | 备注 |
|------|-----|----------|----------|----------|----------|--------|------|
| 4.19 | Day 1 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.20 | Day 2 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.21 | Day 3 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.22 | Day 4 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.23 | Day 5 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.24 | Day 6 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |
| 4.25 | Day 7 | ⏱️ | ⏱️ | ⏱️ | ⏱️ | __% | |

**图例**：
- ⏱️ = 待填写（建议时长）
- ✅ = 已完成
- 🔄 = 进行中
- ❌ = 未完成

---

## 🎯 学习资源推荐

### 必学资源

| 资源 | 链接 | 备注 |
|------|------|------|
| PyTorch官方教程 | https://pytorch.org/tutorials/ | 基础必学 |
| 李宏毅机器学习 | B站搜索 | 深度学习部分 |
| Stanford CS231n | http://cs231n.stanford.edu/ | CNN入门经典 |

### 选学资源

| 资源 | 链接 | 备注 |
|------|------|------|
| Attention论文精读 | 自行搜索PDF | 理解Transformer |
| PyTorch Recipes | 官方文档 | 实用技巧 |

### 环境配置

```bash
# 推荐使用conda环境
conda create -n dl python=3.9
conda activate dl
pip install torch torchvision numpy matplotlib
```

---

## ⚠️ 常见问题与解决方案

### Q1: 梯度消失/爆炸怎么办？

**解决方案**：
- 使用ReLU/Leaky ReLU激活函数
- 添加BatchNorm层
- 使用残差连接(ResNet)
- 梯度裁剪：torch.nn.utils.clip_grad_norm_

### Q2: 模型不收敛怎么办？

**排查步骤**：
1. 检查数据是否正确预处理
2. 检查学习率是否合适
3. 检查损失函数是否正确
4. 检查梯度是否正常更新
5. 增加训练轮数或调整batch size

### Q3: 过拟合严重怎么办？

**解决方案**：
- 增加训练数据
- 使用正则化(L1/L2)
- 添加Dropout层
- 数据增强
- Early Stopping

---

**💪 加油！Week 3 是深度学习的基础阶段，掌握好这些内容将为后续学习打下坚实基础！**
