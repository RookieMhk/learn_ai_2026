# Week 3 预习指南：深度学习基础

> 生成时间：2026-04-17
> 目标：为Week 3深度学习阶段做准备
> 难度定位：入门级，适合有机器学习基础的学习者

---

## 📚 Week 3 学习路线图

```
Week 3: 深度学习基础
│
├── Day 1-2: 神经网络基础
│   ├── 神经元与感知机
│   ├── 多层感知机 (MLP)
│   └── 前向传播与反向传播
│
├── Day 3: 激活函数
│   ├── Sigmoid, Tanh, ReLU
│   ├── 梯度消失与梯度爆炸
│   └── 激活函数选择策略
│
├── Day 4: 损失函数
│   ├── 分类损失 (Cross-Entropy)
│   ├── 回归损失 (MSE, MAE)
│   └── 高级损失函数
│
└── Day 5: 优化器
    ├── SGD, Adam, RMSProp
    ├── 学习率调度
    └── 正则化技术
```

---

## 🧠 第一部分：神经网络基础

### 1.1 从神经元到神经网络

#### 1.1.1 生物学神经元

```
生物神经元：
    ┌─────────────────────────────────────────┐
    │                                         │
    │   树突 ──┐                              │
    │          │     ┌─────────┐     轴突     │
    │   树突 ──┼────▶│ 细胞核  │──────────▶ 神经末梢
    │          │     └─────────┘              │
    │   树突 ──┘                              │
    │                                         │
    └─────────────────────────────────────────┘
    
    输入 → 处理 → 输出
```

#### 1.1.2 人工神经元 (感知机)

**定义**：人工神经元是对生物神经元的数学抽象，是神经网络的基本单元。

**数学原理**：
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    人工神经元结构                            │
│                                                             │
│         x₁ ──┐                                              │
│              │     ┌─────────┐                             │
│         x₂ ──┼────▶│  Σ + f  │─────▶ output y             │
│              │     │  w·x + b │                             │
│         x₃ ──┘     └─────────┘                             │
│                           ▲                                 │
│                      activation                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

单个神经元计算：
    z = w₁x₁ + w₂x₂ + w₃x₃ + b = w·x + b
    y = f(z)

其中：
    - x: 输入特征向量
    - w: 权重向量 (weights)
    - b: 偏置 (bias)
    - f: 激活函数 (activation function)
    - y: 输出
```

**Python实现**：
```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        """初始化权重和偏置"""
        # Xavier初始化
        self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        self.bias = 0
    
    def forward(self, x):
        """前向传播"""
        z = np.dot(self.weights, x) + self.bias
        return z

# 使用示例
neuron = Neuron(n_inputs=3)
output = neuron.forward(np.array([1.0, 2.0, 3.0]))
print(f"神经元输出: {output}")
```

---

### 1.2 感知机 (Perceptron)

#### 1.2.1 什么是感知机

**定义**：感知机是最简单的人工神经网络，由Frank Rosenblatt于1958年提出，只能处理线性可分问题。

**结构**：
```
输入层 ──────▶ 输出层

x₁ ──┐        
x₂ ──┼──▶ Σ ──▶ ŷ ∈ {+1, -1}
x₃ ──┘        
```

**数学原理**：
```python
# 预测函数
def predict(x, w, b):
    z = np.dot(w, x) + b
    return 1 if z > 0 else -1

# 感知机学习算法
def train(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        for i in range(n_samples):
            # 如果预测错误，更新权重
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
    
    return w, b
```

#### 1.2.2 感知机的局限性

**警告**：感知机只能解决线性可分问题！

```
✅ 线性可分：               ❌ 非线性可分（感知机无法解决）：
                          
    ○ ○ ○                      ○   ●
    ─────────                   ╲ ╱
    ● ● ●                      ● ╳ ○
                              ╱ ╲
                              ○   ●
                              
   OR函数                      XOR函数
   (可解决)                   (无法解决)
```

**这导致了多层感知机(MLP)的诞生！**

---

### 1.3 多层感知机 (MLP)

#### 1.3.1 什么是MLP

**定义**：多层感知机是由多个神经元层组成的前馈神经网络，包含输入层、一个或多个隐藏层、输出层。

**结构**：
```
                输入层     隐藏层1     隐藏层2     输出层
                
               ┌───┐     ┌───┐      ┌───┐      ┌───┐
        x₁ ──▶│   │───▶│   │───▶│   │───▶│   │───▶ ŷ₁
               └───┘     └───┘      └───┘      └───┘
               ┌───┐     ┌───┐      ┌───┐      ┌───┐
        x₂ ──▶│   │───▶│   │───▶│   │───▶│   │───▶ ŷ₂
               └───┘     └───┘      └───┘      └───┘
               ┌───┐     ┌───┐      ┌───┐      
        x₃ ──▶│   │───▶│   │───▶│   │───▶  ...
               └───┘     └───┘      └───┘      
               
           (3个特征)   (4个神经元)  (4个神经元)  (2个输出)
```

#### 1.3.2 MLP的数学原理

**前向传播**：
```python
class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: 每层的神经元数量 [input, hidden1, hidden2, ..., output]
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He初始化（适合ReLU）
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], w) + b
            
            # 最后一层不加激活（softmax在外面处理）
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = z
            
            self.activations.append(a)
        
        return self.activations[-1]

# 示例
mlp = MLP([784, 256, 128, 10])  # 手写数字识别
output = mlp.forward(np.random.randn(1, 784))
print(f"输出形状: {output.shape}")  # (1, 10)
```

---

### 1.4 前向传播与反向传播

#### 1.4.1 前向传播 (Forward Propagation)

**定义**：数据从输入层流向输出层，经过每一层的线性变换和非线性激活。

**计算图示例**：
```
输入 x ──▶ [W¹, b¹] ──▶ z¹ ──▶ [σ] ──▶ a¹ ──▶ [W², b²] ──▶ z² ──▶ [σ] ──▶ a² ──▶ [W³, b³] ──▶ z³ ──▶ ŷ
              │                │                 │                │                  │
            权重1            线性组合           激活1            权重2              输出
```

**代码实现**：
```python
import numpy as np

def forwardPropagation(X, weights, biases, activations=['relu', 'relu', 'softmax']):
    """
    前向传播
    
    参数:
        X: 输入数据 (n_samples, n_features)
        weights: 权重列表
        biases: 偏置列表
        activations: 每层的激活函数
    """
    A = X
    cache = {'A0': X}
    
    for i, (W, b, act) in enumerate(zip(weights, biases, activations)):
        Z = np.dot(A, W) + b
        cache[f'Z{i+1}'] = Z
        
        if act == 'relu':
            A = np.maximum(0, Z)
        elif act == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        elif act == 'tanh':
            A = np.tanh(Z)
        elif act == 'softmax':
            A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        else:
            A = Z
        
        cache[f'A{i+1}'] = A
    
    return A, cache

# 使用示例
X = np.random.randn(32, 784)  # 32个样本，784维特征
weights = [np.random.randn(784, 256), np.random.randn(256, 10)]
biases = [np.zeros(256), np.zeros(10)]

output, cache = forwardPropagation(X, weights, biases)
print(f"输出形状: {output.shape}")  # (32, 10)
```

#### 1.4.2 反向传播 (Backpropagation)

**定义**：通过链式法则计算损失函数对每个参数的梯度，从输出层向输入层反向传播。

**链式法则**：
```
损失函数 L 对权重 W 的梯度：
    
    ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
    
链式法则展开：
    
    ∂L/∂W[l] = A[l-1] · δ[l]
    
其中 δ[l] = ∂L/∂z[l] 是第l层的误差
```

**代码实现**：
```python
def backwardPropagation(Y_true, Y_pred, weights, biases, cache, activations):
    """
    反向传播
    
    参数:
        Y_true: 真实标签 (n_samples, n_classes)
        Y_pred: 预测概率 (n_samples, n_classes)
        weights, biases: 模型参数
        cache: 前向传播缓存
        activations: 激活函数列表
    """
    n_samples = Y_true.shape[0]
    grads_W = []
    grads_b = []
    
    # 交叉熵 + Softmax 的梯度简化为 Y_pred - Y_true
    delta = Y_pred - Y_true
    
    for i in reversed(range(len(weights))):
        # 梯度计算
        dW = np.dot(cache[f'A{i}'].T, delta) / n_samples
        db = np.mean(delta, axis=0)
        
        grads_W.insert(0, dW)
        grads_b.insert(0, db)
        
        # 传播到上一层（如果不是第一层）
        if i > 0:
            delta = np.dot(delta, weights[i].T)
            
            # 激活函数梯度
            if activations[i-1] == 'relu':
                delta *= (cache[f'Z{i}'] > 0).astype(float)
            elif activations[i-1] == 'tanh':
                delta *= (1 - np.tanh(cache[f'Z{i}'])**2)
    
    return grads_W, grads_b

# 梯度检查
def gradient_check():
    """简单梯度检查"""
    X = np.random.randn(4, 3)
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    
    # 随机初始化
    W1 = np.random.randn(3, 4) * 0.01
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 3) * 0.01
    b2 = np.zeros(3)
    
    # 前向传播
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = np.dot(A1, W2) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
    
    # 交叉熵损失
    loss = -np.mean(np.log(A2[np.arange(4), np.argmax(Y, axis=1)]))
    
    # 反向传播（计算梯度）
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / 4
    db2 = np.mean(dZ2, axis=0)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)  # ReLU梯度
    dW1 = np.dot(X.T, dZ1) / 4
    db1 = np.mean(dZ1, axis=0)
    
    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = gradient_check()
print("梯度计算完成")
```

---

## 🎯 第二部分：激活函数

### 2.1 为什么需要激活函数

**问题**：如果没有激活函数，神经网络只是线性变换的堆叠，无法解决非线性问题。

```
无激活函数：
    y = W³(W²(W¹x + b¹) + b²) + b³ = Wx + b  # 仍然是线性！

有激活函数：
    y = σ(W³(σ(W²(σ(W¹x + b¹)) + b²)) + b³)  # 可以拟合非线性！
```

---

### 2.2 常用激活函数

#### 2.2.1 Sigmoid

**定义**：将输入压缩到(0,1)区间，常用于二分类输出层。

**数学原理**：
```
σ(x) = 1 / (1 + e^(-x))

导数：
σ'(x) = σ(x) · (1 - σ(x))
```

**代码实现**：
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 绘制图像
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative')
plt.legend()
plt.title('Sigmoid Activation')
plt.grid(True)
```

**特点**：
| 优点 | 缺点 |
|------|------|
| 输出在(0,1)，可表示概率 | 梯度消失（两端趋近于0） |
| 物理意义明确 | 计算开销大（指数运算） |
| 梯度平滑 | 输出不是零中心 |

**应用场景**：二分类输出层

---

#### 2.2.2 Tanh (双曲正切)

**定义**：将输入压缩到(-1,1)区间，输出以0为中心。

**数学原理**：
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

导数：
tanh'(x) = 1 - tanh²(x)
```

**代码实现**：
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

**特点**：
| 优点 | 缺点 |
|------|------|
| 输出以0为中心 | 梯度仍然会消失 |
| 收敛比sigmoid快 | 计算开销大 |

**应用场景**：隐藏层（较少使用）

---

#### 2.2.3 ReLU (线性修正单元)

**定义**：取输入和0的最大值，计算简单且在正区间梯度不消失。

**数学原理**：
```
ReLU(x) = max(0, x)

导数：
ReLU'(x) = 1 if x > 0 else 0
```

**代码实现**：
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**特点**：
| 优点 | 缺点 |
|------|------|
| 计算极快 | Dead ReLU问题（负区间梯度为0） |
| 缓解梯度消失 | 输出不是零中心 |
| 稀疏激活 | |

**变体**：
```python
# Leaky ReLU - 解决Dead ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# PReLU (Parametric ReLU)
# 可学习的参数alpha

# ELU (Exponential Linear Unit)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

---

#### 2.2.4 Softmax

**定义**：将任意实数向量转换为概率分布，所有输出和为1。

**数学原理**：
```
Softmax(x_i) = e^(x_i) / Σⱼ e^(x_j)

数值稳定版本：
Softmax(x_i) = e^(x_i - max(x)) / Σⱼ e^(x_j - max(x))
```

**代码实现**：
```python
def softmax(x):
    """数值稳定的softmax"""
    x = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

**应用场景**：多分类输出层

---

### 2.3 梯度消失与梯度爆炸

#### 2.3.1 问题描述

```
梯度消失：随着层数增加，梯度越来越小，参数几乎不更新
梯度爆炸：随着层数增加，梯度越来越大，参数更新不稳定
```

#### 2.3.2 解决方案

```python
# 1. 权重初始化（Xavier/He初始化）
def xavier_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))

def he_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

# 2. 梯度裁剪
def clip_gradients(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    return [g * clip_coef for g in grads] if total_norm > max_norm else grads

# 3. Batch Normalization
class BatchNorm:
    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
    
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.x_norm = (x - mean) / np.sqrt(var + self.eps)
            out = self.gamma * self.x_norm + self.beta
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            out = self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + self.eps) + self.beta
        
        return out
```

---

## 📉 第三部分：损失函数

### 3.1 分类损失

#### 3.1.1 交叉熵损失 (Cross-Entropy Loss)

**定义**：衡量预测概率分布与真实分布之间的差异。

**数学原理**：
```
L = -Σ y_i · log(ŷ_i)

对于二分类：
L = -[y · log(ŷ) + (1-y) · log(1-ŷ)]

对于多分类（带softmax）：
L = -Σⱼ y_j · log(softmax(z)_j)
```

**代码实现**：
```python
def cross_entropy(Y_true, Y_pred, epsilon=1e-15):
    """交叉熵损失（数值稳定）"""
    # 裁剪预测概率防止log(0)
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))

# PyTorch风格
def cross_entropy_pytorch(logits, targets):
    """logits是未经过softmax的原始输出"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return -np.mean(np.sum(targets * np.log(probs + 1e-15), axis=1))
```

**与softmax的结合**：
```python
# CrossEntropyLoss = LogSoftmax + NLLLoss
# 在反向传播时，简化为 Y_pred - Y_true
def cross_entropy_backward(Y_pred, Y_true):
    """softmax + cross_entropy 的梯度"""
    return Y_pred - Y_true  # 简洁！
```

---

### 3.2 回归损失

#### 3.2.1 MSE (均方误差)

**定义**：预测值与真实值之差的平方的均值。

**数学原理**：
```
L_MSE = (1/n) · Σ(y_i - ŷ_i)²
```

**代码实现**：
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)
```

#### 3.2.2 MAE (平均绝对误差)

**定义**：预测值与真实值之差的绝对值的均值。

**数学原理**：
```
L_MAE = (1/n) · Σ|y_i - ŷ_i|
```

**代码实现**：
```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_grad(y_true, y_pred):
    return np.sign(y_pred - y_true) / len(y_true)
```

#### 3.2.3 Huber Loss

**定义**：结合MSE和MAE的优点，对异常值更鲁棒。

**数学原理**：
```
L_Huber = Σ H_δ(y_i - ŷ_i)

其中 H_δ(z) = {
    z²/2                    if |z| ≤ δ
    δ·|z| - δ²/2            if |z| > δ
}
```

**代码实现**：
```python
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * np.abs(error) - 0.5 * delta**2
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))
```

---

## ⚡ 第四部分：优化器

### 4.1 SGD (随机梯度下降)

**代码实现**：
```python
class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.momentum > 0:
                self.v[i] = self.momentum * self.v[i] + g
                p -= self.lr * self.v[i]
            else:
                p -= self.lr * g
        
        return params
```

---

### 4.2 Adam (Adaptive Moment Estimation)

**定义**：结合动量和自适应学习率的优化器，是深度学习最常用的优化器。

**数学原理**：
```
# 1. 计算梯度
g_t

# 2. 更新一阶矩估计（类似动量）
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

# 3. 更新二阶矩估计（类似RMSProp）
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

# 4. 偏差校正
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# 5. 参数更新
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

**代码实现**：
```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # 更新一阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # 更新二阶矩
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # 更新参数
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params
```

**PyTorch使用**：
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()      # 清零梯度
        outputs = model(batch.x)   # 前向传播
        loss = criterion(outputs, batch.y)
        loss.backward()            # 反向传播
        optimizer.step()           # 更新参数
```

---

### 4.3 优化器对比

```
┌────────────────────────────────────────────────────────────────┐
│                     优化器选择指南                              │
├──────────────────┬─────────────────────────────────────────────┤
│     优化器       │               适用场景                      │
├──────────────────┼─────────────────────────────────────────────┤
│     SGD          │ 经典可靠，调参简单，需要fine-tune             │
│   SGD+Momentum   │ 需要快速收敛，逃逸局部最优                    │
│     Adam          │ 默认选择，快速原型开发                       │
│   AdamW           │ 需要权重衰减正则化                          │
│   RMSProp         │ 梯度变化剧烈的场景                          │
│    LAMB           │ 大批量训练（如>1024）                       │
└──────────────────┴─────────────────────────────────────────────┘
```

---

## 📝 预习作业

### 必做练习

```python
# 练习1: 实现一个单隐藏层MLP
"""
目标：用numpy实现一个两层的神经网络
- 输入层: 784维（MNIST图像）
- 隐藏层: 128神经元 + ReLU
- 输出层: 10类 + Softmax
- 损失: 交叉熵
"""

import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, y_pred):
        n = X.shape[0]
        
        # 输出层梯度 (softmax + cross-entropy)
        dz2 = y_pred - y
        
        dW2 = np.dot(self.a1.T, dz2) / n
        db2 = np.mean(dz2, axis=0)
        
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU梯度
        
        dW1 = np.dot(X.T, dz1) / n
        db1 = np.mean(dz1, axis=0)
        
        return dW1, db1, dW2, db2

# 测试
mlp = SimpleMLP(784, 128, 10)
X = np.random.randn(32, 784)
y_pred = mlp.forward(X)
dW1, db1, dW2, db2 = mlp.backward(X, np.zeros((32, 10)), y_pred)
print("MLP梯度计算测试通过！")
```

### 思考题

1. **为什么ReLU比Sigmoid更常用？**
   - 提示：考虑计算效率、梯度消失问题

2. **为什么分类任务用交叉熵而不是MSE？**
   - 提示：考虑梯度特性、概率解释

3. **Adam中的偏差校正(bias correction)是为了解决什么问题？**
   - 提示：考虑参数初始化时的情况

---

## 📚 参考资源

### 推荐阅读
- 《深度学习》(Ian Goodfellow) - 第6章
- 《神经网络与深度学习》(Michael Nielsen) - 在线免费
- CS231n Lecture 4: Introduction to Neural Networks

### 框架文档
- PyTorch: https://pytorch.org/docs/
- TensorFlow: https://www.tensorflow.org/api_docs

### 视频课程
- 3Blue1Brown: Neural Networks playlist
- Stanford CS231n: https://cs231n.github.io/

---

> 📌 **Week 3 学习建议**：
> - 先理解概念，再用numpy实现一遍
> - 对比PyTorch实现，理解自动求导的便利
> - 完成上面的练习题检验学习效果
