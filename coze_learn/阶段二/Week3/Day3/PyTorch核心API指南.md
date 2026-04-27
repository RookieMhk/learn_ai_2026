# PyTorch核心API指南

> **Week 3 Day 3 - 框架入门核心资料**
> 
> 本指南帮助你从NumPy过渡到PyTorch，掌握深度学习工业化工具的核心API。

---

## 目录

1. [张量(Tensor)基础](#1-张量tensor基础)
2. [自动微分(autograd)](#2-自动微分autograd)
3. [神经网络模块(nn.Module)](#3-神经网络模块nnmodule)
4. [数据加载(DataLoader)](#4-数据加载dataloader)
5. [训练循环标准写法](#5-训练循环标准写法)
6. [GPU加速](#6-gpu加速)

---

## 1. 张量(Tensor)基础

### 1.1 什么是张量

**张量 = NumPy数组 + GPU加速 + 自动微分**

| NumPy | PyTorch |
|-------|---------|
| `np.array` | `torch.tensor` |
| `np.ndarray` | `torch.Tensor` |
| `np.reshape` | `torch.reshape` / `view` |
| `np.dot` | `torch.mm` / `torch.matmul` |

### 1.2 创建张量

```python
import torch
import numpy as np

# 从Python列表创建
x = torch.tensor([1.0, 2.0, 3.0])
print(x)  # tensor([1., 2., 3.])

# 从NumPy数组创建
np_array = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(np_array)
print(x)
# tensor([[1, 2],
#         [3, 4]])

# 特殊张量创建
ones = torch.ones(3, 4)      # 全1矩阵
zeros = torch.zeros(2, 3)   # 全0矩阵
rand = torch.rand(2, 2)    # [0,1)均匀分布
randn = torch.randn(3, 3)  # 标准正态分布
eye = torch.eye(4)          # 单位矩阵
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### 1.3 张量属性

```python
x = torch.randn(3, 4, 5)

print(x.shape)    # torch.Size([3, 4, 5])
print(x.dtype)    # torch.float32
print(x.device)   # cpu (或 cuda:0)
print(x.ndim)     # 3 (维度数)
print(x.numel())  # 60 (元素总数)
```

### 1.4 张量操作

```python
# 索引和切片（与NumPy完全一致）
x = torch.arange(12).reshape(3, 4)
print(x[:, 1])       # 第2列
print(x[1:3, :])     # 第2-3行

# 形状变换
x = torch.randn(2, 6)
y = x.view(3, 4)     # view不会复制数据（推荐）
y = x.reshape(3, 4) # reshape可能复制
y = x.T              # 转置
y = x.unsqueeze(0)   # 增加维度
y = x.squeeze()      # 移除大小为1的维度

# 拼接与分割
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.cat([a, b], dim=0)      # 按维度0拼接 -> (4, 3)
d = torch.stack([a, b], dim=0)    # 增加新维度 -> (2, 2, 3)
first, second = c.split(2, dim=0) # 分割
parts = torch.chunk(c, 2, dim=0)  # 分块

# 数学运算
a = torch.randn(3, 4)
b = torch.randn(3, 4)

c = a + b              # 加法
c = torch.add(a, b)    # 等价
c = a - b              # 减法
c = a * b              # 逐元素乘法 (不是矩阵乘法!)
c = a / b              # 逐元素除法
c = torch.mm(a, b.T)   # 矩阵乘法
c = torch.matmul(a, b.T)  # 更通用的矩阵乘法
c = a @ b.T            # Python 3.5+ 语法糖

# 归约操作
x = torch.randn(3, 4)
print(x.sum())         # 所有元素求和
print(x.sum(dim=0))    # 按列求和
print(x.mean())        # 均值
print(x.max())         # 最大值
print(x.max(dim=1))    # 返回 (values, indices)

# 广播机制（与NumPy相同）
a = torch.randn(3, 1, 4)
b = torch.randn(1, 5, 4)
c = a + b              # 自动广播为 (3, 5, 4)
```

### 1.5 GPU迁移

```python
# 检查GPU可用性
print(torch.cuda.is_available())  # True/False
print(torch.cuda.device_count())  # GPU数量

# 张量移到GPU
x = torch.randn(3, 4)
if torch.cuda.is_available():
    x_gpu = x.cuda()              # 移到GPU
    x_gpu = x.to('cuda')          # 等价
    x_gpu = x.to('cuda:0')        # 指定GPU
    
# 张量移回CPU
x_cpu = x_gpu.cpu()

# 创建时就指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 4, device=device)

# 模型参数也用同样方式
model = MyModel().to(device)
```

### 1.6 与NumPy互转

```python
import numpy as np
import torch

# PyTorch -> NumPy
t = torch.randn(3, 4)
n = t.numpy()           # 共享内存，修改一方会影响另一方

# NumPy -> PyTorch
n = np.random.randn(3, 4)
t = torch.from_numpy(n)  # 共享内存

# 完全独立的副本
t_copy = torch.tensor(t.numpy())  # 或
t_copy = t.clone().numpy()         # 先clone再转
```

---

## 2. 自动微分(autograd)

**autograd是PyTorch最核心的功能：自动计算梯度，无需手动求导！**

### 2.1 基本概念

```python
# requires_grad=True 开启梯度追踪
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True

# 所有在该张量上进行的操作都会被记录
y = x ** 2             # y = x^2
z = y.sum()            # z = sum(y)

# 反向传播计算梯度
z.backward()           # dz/dx = 2x

print(x.grad)          # tensor([2., 4., 6.])
```

### 2.2 计算图

```
x (requires_grad=True)
    │
    ▼
x² ──► y
    │
    ▼
sum(y) ──► z (标量，backward的起点)
```

```python
# 关键概念：
# - backward() 只在标量上调用，或者传入与输出相同形状的梯度
# - 每个张量都有 .grad 属性存储梯度
# - 计算完后可以删除计算图以节省内存

# 阻止梯度追踪
x = torch.randn(3, requires_grad=True)
y = x ** 2

# 方法1: detach()
y_detached = y.detach()

# 方法2: with torch.no_grad:
with torch.no_grad():
    z = x ** 2
    print(z.requires_grad)  # False

# 方法3: 数据赋值（不需要梯度时）
x = torch.randn(3, requires_grad=True)
x.data *= 2  # 修改值但不影响梯度追踪
```

### 2.3 自定义梯度

```python
# 对于复杂函数，可以手动定义梯度
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

# 使用
x = torch.randn(3, requires_grad=True)
y = MyFunc.apply(x)
y.sum().backward()
print(x.grad)  # 2*x
```

### 2.4 梯度计算示例

```python
# 示例：线性回归的梯度下降
# 损失函数: L = (wx + b - y)²
# dL/dw = 2x(wx + b - y)
# dL/db = 2(wx + b - y)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 前向传播
pred = w * x + b
loss = ((pred - y) ** 2).mean()

# 反向传播（自动计算梯度！）
loss.backward()

print(f"dL/dw = {w.grad}")  # dL/dw = -2
print(f"dL/db = {b.grad}")  # dL/db = -1

# 更新参数（手动方式）
lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
    w.grad.zero_()  # 清零梯度
    b.grad.zero_()
```

---

## 3. 神经网络模块(nn.Module)

**nn.Module是构建神经网络的基类，所有网络都应继承它**

### 3.1 基础结构

```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 必需！调用父类构造函数
        
        # 定义层（可学习参数）
        self.linear1 = nn.Linear(10, 32)
        self.linear2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        
        # 定义子模块
        self.encoder = Encoder()
        
    def forward(self, x):
        # 前向传播逻辑
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 使用模型
model = MyModel()
print(model)  # 打印网络结构

# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 3.2 常用层

```python
# 线性层（全连接层）
linear = nn.Linear(in_features=100, out_features=50)
x = torch.randn(32, 100)  # batch_size=32
y = linear(x)              # (32, 50)

# 卷积层
conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
x = torch.randn(8, 3, 32, 32)  # (batch, channels, H, W)
y = conv1(x)                    # (8, 64, 32, 32)

# 池化层
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2)
y = maxpool(x)  # 尺寸减半

# 批归一化
bn = nn.BatchNorm2d(num_features=64)
y = bn(y)

# Dropout
dropout = nn.Dropout(p=0.5)
y = dropout(y)  # 训练时随机置零，推理时不作用

# 激活函数（常用）
x = torch.randn(8, 32)
y = F.relu(x)       # ReLU
y = F.sigmoid(x)    # Sigmoid
y = F.tanh(x)       # Tanh
y = F.gelu(x)       # GELU (Transformer常用)
y = F.softmax(x, dim=1)  # Softmax
```

### 3.3 常用损失函数

```python
# 回归损失
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
smooth_l1 = nn.SmoothL1Loss()

# 分类损失
cross_entropy = nn.CrossEntropyLoss()    # 多分类（ logits -> softmax -> nll）
bce_loss = nn.BCE Loss()                # 二分类
bce_with_logits = nn.BCEWithLogitsLoss() # 数值更稳定

# 使用示例
# MSE Loss
pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = mse_loss(pred, target)

# Cross Entropy（target是类别索引，不是one-hot）
pred = torch.randn(10, 5)     # 10个样本，5个类别
target = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
loss = cross_entropy(pred, target)
```

### 3.4 优化器

```python
import torch.optim as optim

model = MyModel()

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (最常用)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (带权重衰减的Adam，更推荐)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

# 训练循环中更新学习率
for epoch in range(num_epochs):
    train()
    scheduler.step()
    # 或
    scheduler.step(val_loss)
```

---

## 4. 数据加载(DataLoader)

### 4.1 Dataset与DataLoader

```python
from torch.utils.data import Dataset, DataLoader

# 自定义Dataset
class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 加载数据
        self.data = load_data(data_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# 创建DataLoader
dataset = MyDataset('data/train')
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # 训练时True，验证时False
    num_workers=4,          # 多进程加载
    pin_memory=True,        # 加速GPU传输
    drop_last=False         # 丢弃最后一个不完整batch
)

# 迭代使用
for batch_x, batch_y in dataloader:
    # batch_x: (32, features)
    # batch_y: (32,)
    pass
```

### 4.2 常用Transforms

```python
from torchvision import transforms

# 图像预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),  # 转为[0,1]的Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 使用
train_dataset = MyDataset('data/train', transform=train_transform)
test_dataset = MyDataset('data/test', transform=test_transform)
```

### 4.3 内置数据集

```python
from torchvision import datasets
from torch.utils.data import random_split

# 使用内置数据集
train_data = datasets.MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root='data/',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 数据集划分
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)
```

---

## 5. 训练循环标准写法

### 5.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度（重要！）
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, correct / total

# 主训练流程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")
```

### 5.2 模型保存与加载

```python
# 保存整个模型（不推荐，依赖代码结构）
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 保存模型参数（推荐）
torch.save(model.state_dict(), 'model_weights.pth')
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))

# 保存检查点（训练中断恢复）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## 6. GPU加速

### 6.1 设备管理

```python
# 基础设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 多GPU设置
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # 数据并行
    model = model.cuda()

# GPU内存管理
torch.cuda.empty_cache()  # 清理未使用显存

# 查看GPU信息
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
```

### 6.2 混合精度训练（AMP）

**使用FP16加速训练，减少显存占用，加速计算**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in dataloader:
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
    
    optimizer.zero_grad()
    
    # 前向用FP16
    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
    
    # 反向用GradScaler处理梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 6.3 torch.compile (PyTorch 2.0+)

**JIT编译，加速运行**

```python
# PyTorch 2.0+ 编译优化
model = MyModel().cuda()
model = torch.compile(model)  # 编译

# 效果：训练速度提升10-30%，推理速度提升2-10倍
```

### 6.4 GPU加速效果对比

| 任务 | CPU | GPU (FP32) | GPU (FP16) | GPU (compile) |
|------|-----|------------|------------|---------------|
| 小模型训练 | 基准 | 10-20x | 15-30x | 20-50x |
| 大模型推理 | 无法运行 | 基准 | 2x | 3x |

---

## 附录：常见问题

### Q1: 梯度不更新？
```python
# 检查：是否清零梯度
optimizer.zero_grad()  # 每个batch前清零

# 检查：requires_grad
x = torch.randn(3, requires_grad=True)
```

### Q2: 显存不足？
```python
# 1. 减小batch_size
# 2. 使用梯度累积
accumulation_steps = 4
for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 清理显存
torch.cuda.empty_cache()
del loss, outputs
```

### Q3: 模型eval和train模式区别？
- **train模式**: Dropout生效，BatchNorm用当前batch统计
- **eval模式**: Dropout关闭，BatchNorm用全局统计

---

## 学习资源

- 官方文档: https://pytorch.org/docs/
- PyTorch 2.11发布: https://pytorch.org/blog/pytorch-2-11/
- 欧洲PyTorch大会 2026.4.7-8 巴黎
- PyTorch中文文档: https://pytorch.ac.cn/
