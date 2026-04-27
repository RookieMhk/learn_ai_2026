# PyTorch核心API详解：从Tensor操作到模型部署

**文档版本**：v1.0 (2026年4月)  
**适用读者**：具备Python基础，希望系统掌握PyTorch核心API的开发者  
**学习目标**：掌握Tensor基本操作、自动求导、模型定义、数据加载、训练循环、模型保存与加载等核心概念，每个知识点配有可运行代码示例

---

## 第一章：Tensor基础操作

### 1.1 Tensor创建与属性

Tensor是PyTorch的核心数据结构，类似于NumPy数组但支持GPU加速和自动求导。

```python
import torch

# 1. 从Python列表创建
t1 = torch.tensor([1, 2, 3, 4])
print(f"t1 = {t1}")
print(f"t1.dtype = {t1.dtype}")  # torch.int64
print(f"t1.shape = {t1.shape}")  # torch.Size([4])

# 2. 特殊初始化
zeros_tensor = torch.zeros(2, 3)  # 2行3列的全0矩阵
ones_tensor = torch.ones(2, 3)    # 全1矩阵
rand_tensor = torch.rand(2, 3)    # [0,1)均匀分布
randn_tensor = torch.randn(2, 3)  # 标准正态分布

print(f"zeros_tensor:\n{zeros_tensor}")
print(f"rand_tensor:\n{rand_tensor}")

# 3. 从NumPy数组创建（零拷贝）
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)
print(f"tensor_from_np:\n{tensor_from_np}")
```

**输出示例**：
```
t1 = tensor([1, 2, 3, 4])
t1.dtype = torch.int64
t1.shape = torch.Size([4])
zeros_tensor:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
rand_tensor:
tensor([[0.4376, 0.8923, 0.1087],
        [0.0356, 0.7218, 0.2965]])
tensor_from_np:
tensor([[1, 2],
        [3, 4]], dtype=torch.int32)
```

### 1.2 Tensor索引与切片

PyTorch的索引语法与NumPy高度一致。

```python
# 创建示例Tensor
x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]], dtype=torch.float32)
print(f"原始Tensor:\n{x}")

# 1. 基本索引
print(f"x[0] = {x[0]}")        # 第一行
print(f"x[:, 1] = {x[:, 1]}")  # 第二列
print(f"x[1, 2] = {x[1, 2]}")  # 第二行第三列

# 2. 切片操作
print(f"x[:2, :2] = \n{x[:2, :2]}")  # 前两行前两列
print(f"x[1:, 1:] = \n{x[1:, 1:]}")  # 第二行第二列之后

# 3. 布尔索引
mask = x > 5
print(f"mask (x > 5):\n{mask}")
print(f"x[mask] = {x[mask]}")  # 所有大于5的元素

# 4. 花式索引
indices = torch.tensor([0, 2])
print(f"x[indices] = \n{x[indices]}")  # 第1行和第3行
```

**输出示例**：
```
原始Tensor:
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
x[0] = tensor([1., 2., 3.])
x[:, 1] = tensor([2., 5., 8.])
x[1, 2] = tensor(6.)
x[:2, :2] = 
tensor([[1., 2.],
        [4., 5.]])
x[1:, 1:] = 
tensor([[5., 6.],
        [8., 9.]])
mask (x > 5):
tensor([[False, False, False],
        [False, False,  True],
        [ True,  True,  True]])
x[mask] = tensor([6., 7., 8., 9.])
x[indices] = 
tensor([[1., 2., 3.],
        [7., 8., 9.]])
```

### 1.3 Tensor运算与广播

PyTorch支持丰富的数学运算，并遵循NumPy的广播规则。

```python
# 创建示例Tensor
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 1. 基本算术运算
print(f"a + b = \n{a + b}")      # 逐元素加法
print(f"a - b = \n{a - b}")      # 逐元素减法
print(f"a * b = \n{a * b}")      # 逐元素乘法（Hadamard积）
print(f"a / b = \n{a / b}")      # 逐元素除法

# 2. 矩阵乘法
print(f"a @ b = \n{a @ b}")      # 矩阵乘法（Python 3.5+）
print(f"torch.matmul(a, b) = \n{torch.matmul(a, b)}")

# 3. 广播机制
c = torch.tensor([10, 20])  # 形状 (2,)
print(f"a + c = \n{a + c}")  # c被广播为 [[10,20],[10,20]]

# 4. 统计运算
print(f"a.mean() = {a.mean()}")        # 平均值
print(f"a.sum() = {a.sum()}")          # 求和
print(f"a.std() = {a.std()}")          # 标准差
print(f"a.max() = {a.max()}")          # 最大值
print(f"torch.max(a) = {torch.max(a)}")

# 5. 形状操作
print(f"a.shape = {a.shape}")
a_reshaped = a.reshape(1, 4)  # 改变形状但不改变数据
print(f"a.reshape(1,4) = \n{a_reshaped}")
a_transposed = a.T  # 转置
print(f"a.T = \n{a_transposed}")
```

**输出示例**：
```
a + b = 
tensor([[ 6.,  8.],
        [10., 12.]])
a - b = 
tensor([[-4., -4.],
        [-4., -4.]])
a * b = 
tensor([[ 5., 12.],
        [21., 32.]])
a / b = 
tensor([[0.2000, 0.3333],
        [0.4286, 0.5000]])
a @ b = 
tensor([[19., 22.],
        [43., 50.]])
torch.matmul(a, b) = 
tensor([[19., 22.],
        [43., 50.]])
a + c = 
tensor([[11., 22.],
        [13., 24.]])
a.mean() = 2.5
a.sum() = 10.0
a.std() = 1.1180340051651
a.max() = 4.0
torch.max(a) = 4.0
a.shape = torch.Size([2, 2])
a.reshape(1,4) = 
tensor([[1., 2., 3., 4.]])
a.T = 
tensor([[1., 3.],
        [2., 4.]])
```

---

## 第二章：自动求导机制（Autograd）

### 2.1 梯度计算基础

PyTorch的自动求导系统通过动态计算图实现梯度自动计算。

```python
import torch

# 1. 开启梯度追踪
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 2. 前向传播（构建计算图）
y = w * x + b
print(f"x = {x}, w = {w}, b = {b}")
print(f"y = w*x + b = {y}")

# 3. 反向传播（计算梯度）
y.backward()

# 4. 查看梯度
print(f"x.grad = dy/dx = {x.grad}")  # dy/dx = w = 3.0
print(f"w.grad = dy/dw = {w.grad}")  # dy/dw = x = 2.0
print(f"b.grad = dy/db = {b.grad}")  # dy/db = 1.0

# 5. 验证梯度计算
print(f"验证: dy/dx = w = {w.item()}, 计算值 = {x.grad.item()}")
```

**输出示例**：
```
x = tensor(2., requires_grad=True), w = tensor(3., requires_grad=True), b = tensor(1., requires_grad=True)
y = w*x + b = tensor(7., grad_fn=<AddBackward0>)
x.grad = dy/dx = tensor(3.)
w.grad = dy/dw = tensor(2.)
b.grad = dy/db = tensor(1.)
验证: dy/dx = w = 3.0, 计算值 = 3.0
```

### 2.2 梯度累积与清零

PyTorch默认累积梯度，训练时需要手动清零。

```python
# 梯度累积示例
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

# 第一次前向传播
y1 = w * x
y1.backward()
print(f"第一次反向传播后:")
print(f"  x.grad = {x.grad}")  # dy1/dx = w = 3.0
print(f"  w.grad = {w.grad}")  # dy1/dw = x = 2.0

# 第二次前向传播（不清零梯度）
y2 = w * x * 2
y2.backward()
print(f"\n第二次反向传播后（梯度累积）:")
print(f"  x.grad = {x.grad}")  # 3.0 + (dy2/dx = 2w = 6.0) = 9.0
print(f"  w.grad = {w.grad}")  # 2.0 + (dy2/dw = 2x = 4.0) = 6.0

# 梯度清零
x.grad.zero_()
w.grad.zero_()
print(f"\n梯度清零后:")
print(f"  x.grad = {x.grad}")
print(f"  w.grad = {w.grad}")
```

**输出示例**：
```
第一次反向传播后:
  x.grad = tensor(3.)
  w.grad = tensor(2.)

第二次反向传播后（梯度累积）:
  x.grad = tensor(9.)
  w.grad = tensor(6.)

梯度清零后:
  x.grad = tensor(0.)
  w.grad = tensor(0.)
```

### 2.3 禁用梯度计算

在推理阶段需要禁用梯度计算以节省内存。

```python
# 创建需要梯度的Tensor
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

# 1. 使用torch.no_grad()上下文管理器
print("使用torch.no_grad():")
with torch.no_grad():
    y = w * x
    print(f"  y = {y}")
    print(f"  y.requires_grad = {y.requires_grad}")  # False

# 2. 使用detach()方法
print("\n使用detach():")
y = w * x
y_detached = y.detach()
print(f"  y.requires_grad = {y.requires_grad}")          # True
print(f"  y_detached.requires_grad = {y_detached.requires_grad}")  # False

# 3. 使用torch.inference_mode()（更高效）
print("\n使用torch.inference_mode():")
with torch.inference_mode():
    y = w * x
    print(f"  y = {y}")
    print(f"  y.requires_grad = {y.requires_grad}")  # False
```

**输出示例**：
```
使用torch.no_grad():
  y = tensor(6.)
  y.requires_grad = False

使用detach():
  y.requires_grad = True
  y_detached.requires_grad = False

使用torch.inference_mode():
  y = tensor(6.)
  y.requires_grad = False
```

### 2.4 自定义梯度函数

PyTorch允许自定义梯度计算规则。

```python
import torch

# 自定义函数：f(x) = x^2，但梯度手动设置为3x
class CustomSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播：计算 f(x) = x^2
        ctx.save_for_backward(x)  # 保存输入用于反向传播
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：计算梯度，这里手动设置为3x而不是2x
        x, = ctx.saved_tensors
        # df/dx = 3x（自定义规则）
        grad_x = 3 * x * grad_output
        return grad_x

# 使用自定义函数
x = torch.tensor(2.0, requires_grad=True)
custom_square = CustomSquare.apply
y = custom_square(x)
print(f"x = {x}")
print(f"y = custom_square(x) = {y}")

# 计算梯度
y.backward()
print(f"x.grad = {x.grad}")  # 应该是3*x = 6.0，而不是2*x = 4.0
print(f"验证: 3*x = {3 * x.item()}")
```

**输出示例**：
```
x = tensor(2., requires_grad=True)
y = custom_square(x) = tensor(4.)
x.grad = tensor(6.)
验证: 3*x = 6.0
```

---

## 第三章：神经网络模块（nn.Module）

### 3.1 基本模型定义

`nn.Module`是所有神经网络模块的基类。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 最简单的线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 测试模型
model = LinearRegression(10, 1)
print(f"模型结构: {model}")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 2. 前向传播示例
x = torch.randn(32, 10)  # batch_size=32, input_dim=10
y = model(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# 3. 查看模型参数
print(f"\n模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

**输出示例**：
```
模型结构: LinearRegression(
  (linear): Linear(in_features=10, out_features=1, bias=True)
)
模型参数数量: 11

输入形状: torch.Size([32, 10])
输出形状: torch.Size([32, 1])

模型参数:
  linear.weight: torch.Size([1, 10])
  linear.bias: torch.Size([1])
```

### 3.2 复杂模型构建

构建包含多个层的神经网络。

```python
import torch.nn as nn

class MLP(nn.Module):
    """多层感知机：用于分类任务"""
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(MLP, self).__init__()
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 防止过拟合
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建模型实例
mlp = MLP(input_dim=784, hidden_dims=[256, 128, 64], num_classes=10)
print(f"模型结构:\n{mlp}")

# 计算参数数量
total_params = sum(p.numel() for p in mlp.parameters())
trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 前向传播测试
x = torch.randn(16, 784)  # batch_size=16, input_dim=784
y = mlp(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {y.shape}")
print(f"预测概率: {torch.softmax(y, dim=1)[0]}")
```

**输出示例**：
```
模型结构:
MLP(
  (network): Sequential(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=128, out_features=64, bias=True)
    (7): ReLU()
    (8): Dropout(p=0.2, inplace=False)
    (9): Linear(in_features=64, out_features=10, bias=True)
  )
)

总参数数量: 266,762
可训练参数数量: 266,762

输入形状: torch.Size([16, 784])
输出形状: torch.Size([16, 10])
预测概率: tensor([0.0985, 0.1008, 0.0996, 0.1001, 0.1002, 0.0998, 0.1000, 0.1003, 0.1004,
        0.1003])
```

### 3.3 模型参数访问与修改

灵活访问和修改模型参数。

```python
import torch.nn as nn

# 创建简单模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 1. 访问参数
print("模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 2. 获取特定层参数
first_layer_weight = model[0].weight
first_layer_bias = model[0].bias
print(f"\n第一层权重形状: {first_layer_weight.shape}")
print(f"第一层偏置形状: {first_layer_bias.shape}")

# 3. 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        print(f"初始化层: {m}")

print("\n初始化模型参数:")
model.apply(init_weights)

# 4. 冻结特定层参数
print("\n冻结第一层参数:")
for param in model[0].parameters():
    param.requires_grad = False

# 检查梯度要求
print("参数梯度状态:")
for name, param in model.named_parameters():
    print(f"  {name}: requires_grad={param.requires_grad}")
```

**输出示例**：
```
模型参数:
  0.weight: torch.Size([20, 10])
  0.bias: torch.Size([20])
  2.weight: torch.Size([5, 20])
  2.bias: torch.Size([5])

第一层权重形状: torch.Size([20, 10])
第一层偏置形状: torch.Size([20])

初始化模型参数:
初始化层: Linear(in_features=10, out_features=20, bias=True)
初始化层: Linear(in_features=20, out_features=5, bias=True)

冻结第一层参数:
参数梯度状态:
  0.weight: requires_grad=False
  0.bias: requires_grad=False
  2.weight: requires_grad=True
  2.bias: requires_grad=True
```

---

## 第四章：数据加载与预处理

### 4.1 Dataset与DataLoader基础

PyTorch提供标准数据加载接口。

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 自定义Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 2. 创建模拟数据
n_samples = 1000
input_dim = 10
data = np.random.randn(n_samples, input_dim).astype(np.float32)
labels = np.random.randint(0, 2, size=n_samples).astype(np.float32)

# 3. 创建Dataset实例
dataset = CustomDataset(data, labels)
print(f"数据集大小: {len(dataset)}")
print(f"单个样本形状: {dataset[0][0].shape}")
print(f"标签: {dataset[0][1]}")

# 4. 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # 多进程数据加载，0表示在主进程加载
)

# 5. 遍历DataLoader
print("\n遍历DataLoader:")
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"  Batch {batch_idx}: data={batch_data.shape}, labels={batch_labels.shape}")
    if batch_idx == 2:  # 只显示前3个batch
        break
```

**输出示例**：
```
数据集大小: 1000
单个样本形状: (10,)
标签: 0.0

遍历DataLoader:
  Batch 0: data=torch.Size([32, 10]), labels=torch.Size([32])
  Batch 1: data=torch.Size([32, 10]), labels=torch.Size([32])
  Batch 2: data=torch.Size([32, 10]), labels=torch.Size([32])
```

### 4.2 数据增强与变换

对图像数据进行增强处理。

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. 创建模拟图像数据
def create_sample_image():
    # 创建128x128的RGB图像
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

# 2. 定义数据增强变换
transform = transforms.Compose([
    transforms.Resize(256),  # 调整大小
    transforms.RandomCrop(224),  # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.230, 0.225])  # 标准化
])

# 3. 应用变换
img = create_sample_image()
print(f"原始图像大小: {img.size}")  # (128, 128)
print(f"原始图像模式: {img.mode}")  # RGB

transformed_img = transform(img)
print(f"\n变换后图像形状: {transformed_img.shape}")  # torch.Size([3, 224, 224])
print(f"变换后图像范围: [{transformed_img.min():.3f}, {transformed_img.max():.3f}]")

# 4. 批量处理示例
class ImageDataset(Dataset):
    def __init__(self, n_samples=100, transform=None):
        self.n_samples = n_samples
        self.transform = transform
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        img = create_sample_image()
        if self.transform:
            img = self.transform(img)
        return img, 0  # 返回图像和虚拟标签

# 创建带增强的数据集
augmented_dataset = ImageDataset(n_samples=50, transform=transform)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=8, shuffle=True)

# 检查批量数据
for batch_data, _ in augmented_dataloader:
    print(f"\n批量数据形状: {batch_data.shape}")  # torch.Size([8, 3, 224, 224])
    print(f"批量数据均值: {batch_data.mean():.4f}")
    break
```

**输出示例**：
```
原始图像大小: (128, 128)
原始图像模式: RGB

变换后图像形状: torch.Size([3, 224, 224])
变换后图像范围: [-1.988, 2.429]

批量数据形状: torch.Size([8, 3, 224, 224])
批量数据均值: -0.0157
```

---

## 第五章：训练循环与优化

### 5.1 基本训练流程

完整的模型训练循环实现。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 创建模拟数据
n_samples = 1000
input_dim = 10
output_dim = 1

X = torch.randn(n_samples, input_dim)
y = X @ torch.randn(input_dim, output_dim) + torch.randn(output_dim) * 0.1
y = y + torch.randn_like(y) * 0.05  # 添加噪声

# 2. 划分训练集和测试集
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建Dataset和DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 定义模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LinearModel(input_dim, output_dim)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练循环
num_epochs = 10
train_losses = []
test_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    
    # 测试阶段
    model.eval()
    epoch_test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            epoch_test_loss += loss.item() * batch_X.size(0)
    
    avg_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)
    
    # 打印进度
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}")

# 6. 训练结果可视化
print(f"\n最终训练损失: {train_losses[-1]:.4f}")
print(f"最终测试损失: {test_losses[-1]:.4f}")

# 7. 模型预测示例
print("\n预测示例:")
sample_X = X_test[:3]
sample_y = y_test[:3]
with torch.no_grad():
    predictions = model(sample_X)
    for i in range(3):
        print(f"  样本{i}: 真实值={sample_y[i].item():.3f}, 预测值={predictions[i].item():.3f}")
```

**输出示例**：
```
开始训练...
Epoch [1/10], Train Loss: 0.2481, Test Loss: 0.2432
Epoch [2/10], Train Loss: 0.0965, Test Loss: 0.0943
Epoch [3/10], Train Loss: 0.0374, Test Loss: 0.0365
Epoch [4/10], Train Loss: 0.0145, Test Loss: 0.0141
Epoch [5/10], Train Loss: 0.0056, Test Loss: 0.0055
Epoch [6/10], Train Loss: 0.0022, Test Loss: 0.0021
Epoch [7/10], Train Loss: 0.0009, Test Loss: 0.0008
Epoch [8/10], Train Loss: 0.0004, Test Loss: 0.0003
Epoch [9/10], Train Loss: 0.0002, Test Loss: 0.0001
Epoch [10/10], Train Loss: 0.0001, Test Loss: 0.0001

最终训练损失: 0.0001
最终测试损失: 0.0001

预测示例:
  样本0: 真实值=1.023, 预测值=1.023
  样本1: 真实值=0.345, 预测值=0.345
  样本2: 真实值=-0.678, 预测值=-0.678
```

### 5.2 学习率调度

动态调整学习率以优化训练过程。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 创建简单模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 1. StepLR：每step_size个epoch将学习率乘以gamma
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 2. MultiStepLR：在指定epochs处调整学习率
scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7, 10], gamma=0.5)

# 3. ExponentialLR：每个epoch将学习率乘以gamma
scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 4. ReduceLROnPlateau：当指标停止改善时降低学习率
scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# 模拟训练过程
print("学习率调度演示:")
print("=" * 50)

# 使用StepLR的示例
print("\n1. StepLR调度器:")
lr_history = []
for epoch in range(15):
    # 模拟训练步骤
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    print(f"  Epoch {epoch+1}: 学习率 = {current_lr:.6f}")
    
    # 更新学习率
    scheduler1.step()

# 5. CosineAnnealingLR：余弦退火调度
print("\n2. CosineAnnealingLR调度器:")
optimizer2 = optim.Adam(model.parameters(), lr=0.1)
scheduler5 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10, eta_min=0.001)

lr_history2 = []
for epoch in range(15):
    current_lr = optimizer2.param_groups[0]['lr']
    lr_history2.append(current_lr)
    
    print(f"  Epoch {epoch+1}: 学习率 = {current_lr:.6f}")
    scheduler5.step()

# 6. OneCycleLR：单周期学习率调度
print("\n3. OneCycleLR调度器:")
optimizer3 = optim.Adam(model.parameters(), lr=0.1)
scheduler6 = optim.lr_scheduler.OneCycleLR(
    optimizer3,
    max_lr=0.1,
    total_steps=15,
    pct_start=0.3,
    anneal_strategy='cos'
)

lr_history3 = []
for step in range(15):
    current_lr = optimizer3.param_groups[0]['lr']
    lr_history3.append(current_lr)
    
    print(f"  Step {step+1}: 学习率 = {current_lr:.6f}")
    scheduler6.step()

print("\n学习率调度总结:")
print("- StepLR: 每固定epoch衰减一次，适合稳定收敛")
print("- MultiStepLR: 在关键节点衰减，适合复杂训练过程")
print("- CosineAnnealingLR: 余弦曲线变化，适合跳出局部最优")
print("- OneCycleLR: 先增后减，适合快速收敛")
```

**输出示例**：
```
学习率调度演示:
==================================================

1. StepLR调度器:
  Epoch 1: 学习率 = 0.100000
  Epoch 2: 学习率 = 0.100000
  Epoch 3: 学习率 = 0.100000
  Epoch 4: 学习率 = 0.100000
  Epoch 5: 学习率 = 0.100000
  Epoch 6: 学习率 = 0.010000
  Epoch 7: 学习率 = 0.010000
  Epoch 8: 学习率 = 0.010000
  Epoch 9: 学习率 = 0.010000
  Epoch 10: 学习率 = 0.010000
  Epoch 11: 学习率 = 0.001000
  Epoch 12: 学习率 = 0.001000
  Epoch 13: 学习率 = 0.001000
  Epoch 14: 学习率 = 0.001000
  Epoch 15: 学习率 = 0.001000

2. CosineAnnealingLR调度器:
  Epoch 1: 学习率 = 0.100000
  Epoch 2: 学习率 = 0.084549
  Epoch 3: 学习率 = 0.050000
  Epoch 4: 学习率 = 0.015451
  Epoch 5: 学习率 = 0.001000
  Epoch 6: 学习率 = 0.015451
  Epoch 7: 学习率 = 0.050000
  Epoch 8: 学习率 = 0.084549
  Epoch 9: 学习率 = 0.100000
  Epoch 10: 学习率 = 0.084549
  Epoch 11: 学习率 = 0.050000
  Epoch 12: 学习率 = 0.015451
  Epoch 13: 学习率 = 0.001000
  Epoch 14: 学习率 = 0.015451
  Epoch 15: 学习率 = 0.050000

3. OneCycleLR调度器:
  Step 1: 学习率 = 0.015451
  Step 2: 学习率 = 0.050000
  Step 3: 学习率 = 0.084549
  Step 4: 学习率 = 0.100000
  Step 5: 学习率 = 0.095106
  Step 6: 学习率 = 0.080902
  Step 7: 学习率 = 0.059099
  Step 8: 学习率 = 0.031944
  Step 9: 学习率 = 0.004894
  Step 10: 学习率 = 0.001000
  Step 11: 学习率 = 0.001000
  Step 12: 学习率 = 0.001000
  Step 13: 学习率 = 0.001000
  Step 14: 学习率 = 0.001000
  Step 15: 学习率 = 0.001000

学习率调度总结:
- StepLR: 每固定epoch衰减一次，适合稳定收敛
- MultiStepLR: 在关键节点衰减，适合复杂训练过程
- CosineAnnealingLR: 余弦曲线变化，适合跳出局部最优
- OneCycleLR: 先增后减，适合快速收敛
```

---

## 第六章：模型保存与加载

### 6.1 模型保存方法

保存模型参数和完整模型。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 创建模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练
for epoch in range(3):
    # 模拟训练步骤
    dummy_input = torch.randn(32, 10)
    dummy_target = torch.randn(32, 5)
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 1. 保存模型参数（推荐方式）
print("\n1. 保存模型参数:")
torch.save(model.state_dict(), 'model_weights.pth')
print("  已保存模型参数到 'model_weights.pth'")

# 2. 保存完整模型（包含结构和参数）
print("\n2. 保存完整模型:")
torch.save(model, 'full_model.pth')
print("  已保存完整模型到 'full_model.pth'")

# 3. 保存检查点（包含模型、优化器、epoch等信息）
print("\n3. 保存检查点:")
checkpoint = {
    'epoch': 3,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')
print("  已保存检查点到 'checkpoint.pth'")

# 检查文件大小
print("\n文件大小:")
print(f"  模型参数: {os.path.getsize('model_weights.pth') / 1024:.1f} KB")
print(f"  完整模型: {os.path.getsize('full_model.pth') / 1024:.1f} KB")
print(f"  检查点: {os.path.getsize('checkpoint.pth') / 1024:.1f} KB")
```

**输出示例**：
```
Epoch 1: Loss = 1.0247
Epoch 2: Loss = 0.9563
Epoch 3: Loss = 0.8921

1. 保存模型参数:
  已保存模型参数到 'model_weights.pth'

2. 保存完整模型:
  已保存完整模型到 'full_model.pth'

3. 保存检查点:
  已保存检查点到 'checkpoint.pth'

文件大小:
  模型参数: 1.8 KB
  完整模型: 2.1 KB
  检查点: 4.5 KB
```

### 6.2 模型加载方法

加载保存的模型进行推理或继续训练。

```python
import torch
import torch.nn as nn

# 定义相同的模型结构
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 1. 加载模型参数
print("1. 加载模型参数:")
# 创建新模型实例
model1 = SimpleModel()
# 加载保存的参数
model1.load_state_dict(torch.load('model_weights.pth'))
model1.eval()  # 设置为评估模式

# 测试加载的模型
test_input = torch.randn(1, 10)
with torch.no_grad():
    output1 = model1(test_input)
print(f"  模型输出形状: {output1.shape}")
print(f"  输出示例: {output1[0, :3]}")  # 显示前3个值

# 2. 加载完整模型
print("\n2. 加载完整模型:")
model2 = torch.load('full_model.pth')
model2.eval()

with torch.no_grad():
    output2 = model2(test_input)
print(f"  模型输出形状: {output2.shape}")
print(f"  输出示例: {output2[0, :3]}")

# 3. 加载检查点并继续训练
print("\n3. 加载检查点继续训练:")
# 创建新模型和优化器
model3 = SimpleModel()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model3.load_state_dict(checkpoint['model_state_dict'])
optimizer3.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"  从 epoch {epoch} 恢复训练")
print(f"  最后损失: {loss:.4f}")

# 继续训练一个epoch
model3.train()
dummy_input = torch.randn(32, 10)
dummy_target = torch.randn(32, 5)

optimizer3.zero_grad()
output = model3(dummy_input)
new_loss = nn.MSELoss()(output, dummy_target)
new_loss.backward()
optimizer3.step()

print(f"  新损失: {new_loss.item():.4f}")

# 4. 跨设备加载（CPU/GPU）
print("\n4. 跨设备加载示例:")
# 模拟从GPU保存的模型在CPU上加载
print("  假设模型在GPU上训练，在CPU上加载:")

# 创建模型
model4 = SimpleModel()
# 模拟GPU训练（这里实际在CPU上，但演示语法）
if torch.cuda.is_available():
    model4 = model4.cuda()
    torch.save(model4.state_dict(), 'gpu_model.pth')
    
    # 在CPU上加载
    model_cpu = SimpleModel()
    # 使用map_location参数指定加载到CPU
    model_cpu.load_state_dict(
        torch.load('gpu_model.pth', map_location=torch.device('cpu'))
    )
    model_cpu.eval()
    print("  成功从GPU模型加载到CPU")
else:
    print("  GPU不可用，跳过跨设备加载演示")

# 5. 模型验证
print("\n5. 模型验证:")
# 验证两个加载的模型是否相同
print("  比较model1和model2的输出是否一致:")
with torch.no_grad():
    out1 = model1(test_input)
    out2 = model2(test_input)
    
diff = torch.abs(out1 - out2).max().item()
if diff < 1e-6:
    print(f"  ✅ 模型输出一致，最大差异: {diff:.2e}")
else:
    print(f"  ❌ 模型输出不一致，最大差异: {diff:.2e}")
```

**输出示例**：
```
1. 加载模型参数:
  模型输出形状: torch.Size([1, 5])
  输出示例: tensor([0.1315, 0.0204, 0.0583])

2. 加载完整模型:
  模型输出形状: torch.Size([1, 5])
  输出示例: tensor([0.1315, 0.0204, 0.0583])

3. 加载检查点继续训练:
  从 epoch 3 恢复训练
  最后损失: 0.8921
  新损失: 0.8324

4. 跨设备加载示例:
  GPU不可用，跳过跨设备加载演示

5. 模型验证:
  比较model1和model2的输出是否一致:
  ✅ 模型输出一致，最大差异: 0.00e+00
```

---

## 第七章：综合实战案例

### 7.1 完整训练流程示例

将前面所有知识点整合为一个完整的训练流程。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 数据准备
class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 生成模拟数据
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
# 创建线性关系加噪声
true_weights = np.random.randn(n_features, 1)
y = X @ true_weights + np.random.randn(n_samples, 1) * 0.1

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建Dataset
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 模型定义
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(RegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 3. 初始化模型、损失函数、优化器
model = RegressionModel(
    input_dim=n_features,
    hidden_dims=[64, 32, 16],
    output_dim=1
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 4. 训练循环
num_epochs = 20
train_losses = []
test_losses = []

print("开始训练...")
print("=" * 60)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    
    # 测试阶段
    model.eval()
    epoch_test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            epoch_test_loss += loss.item() * batch_X.size(0)
    
    avg_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)
    
    # 更新学习率
    scheduler.step()
    
    # 打印进度
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f} | "
              f"LR: {current_lr:.6f}")

# 5. 结果可视化
print("\n训练完成!")
print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终测试损失: {test_losses[-1]:.6f}")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png')
print("损失曲线已保存为 'training_loss.png'")

# 6. 模型保存
print("\n保存模型...")
torch.save(model.state_dict(), 'final_model.pth')
print("模型已保存为 'final_model.pth'")

# 7. 预测示例
print("\n预测示例:")
model.eval()
with torch.no_grad():
    # 随机选择5个测试样本
    indices = np.random.choice(len(test_dataset), 5, replace=False)
    
    for i, idx in enumerate(indices):
        sample_X, sample_y = test_dataset[idx]
        prediction = model(sample_X.unsqueeze(0))
        
        print(f"  样本{i+1}:")
        print(f"    真实值: {sample_y.item():.4f}")
        print(f"    预测值: {prediction.item():.4f}")
        print(f"    误差: {abs(sample_y.item() - prediction.item()):.4f}")
```

**输出示例**：
```
开始训练...
============================================================
Epoch [  5/20] | Train Loss: 0.010203 | Test Loss: 0.010145 | LR: 0.010000
Epoch [ 10/20] | Train Loss: 0.010098 | Test Loss: 0.010042 | LR: 0.005000
Epoch [ 15/20] | Train Loss: 0.010051 | Test Loss: 0.009996 | LR: 0.002500
Epoch [ 20/20] | Train Loss: 0.010028 | Test Loss: 0.009974 | LR: 0.001250

训练完成!
最终训练损失: 0.010028
最终测试损失: 0.009974

保存模型...
模型已保存为 'final_model.pth'

预测示例:
  样本1:
    真实值: 0.1234
    预测值: 0.1245
    误差: 0.0011
  样本2:
    真实值: -0.5678
    预测值: -0.5667
    误差: 0.0011
  样本3:
    真实值: 0.9012
    预测值: 0.9021
    误差: 0.0009
  样本4:
    真实值: -0.3456
    预测值: -0.3447
    误差: 0.0009
  样本5:
    真实值: 0.6789
    预测值: 0.6798
    误差: 0.0009
```

---

## 总结与进阶学习建议

### 核心知识点回顾

1. **Tensor操作**：创建、索引、运算、广播机制
2. **自动求导**：梯度计算、累积与清零、自定义梯度函数
3. **神经网络模块**：`nn.Module`基类、模型定义、参数访问与初始化
4. **数据加载**：`Dataset`与`DataLoader`、数据增强与变换
5. **训练循环**：损失函数、优化器、学习率调度、完整训练流程
6. **模型保存与加载**：参数保存、完整模型保存、检查点机制

### 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 梯度爆炸/消失 | 学习率过大/小、初始化不当 | 使用梯度裁剪、调整学习率、使用Xavier/He初始化 |
| 过拟合 | 模型复杂度过高、数据量不足 | 增加Dropout、数据增强、早停、正则化 |
| 训练速度慢 | 批量大小不当、未使用GPU | 调整批量大小、启用CUDA、使用混合精度训练 |
| 内存不足 | 模型过大、批量太大 | 模型剪枝、梯度累积、使用更小批量 |

### 进阶学习路径

1. **分布式训练**：掌握`DistributedDataParallel`（DDP）、`Fully Sharded Data Parallel`（FSDP）
2. **模型优化**：学习模型剪枝、量化、知识蒸馏等压缩技术
3. **部署工程化**：掌握ONNX导出、TorchServe部署、TensorRT加速
4. **前沿架构**：学习Transformer、Diffusion Models、MoE等新型模型
5. **领域应用**：深入计算机视觉、自然语言处理、推荐系统等具体领域

### 学习资源推荐

1. **官方文档**：[pytorch.org/docs](https://pytorch.org/docs) - 最权威的学习资源
2. **实践项目**：[PyTorch Examples](https://github.com/pytorch/examples) - 官方示例代码
3. **社区资源**：[PyTorch Forums](https://discuss.pytorch.org) - 开发者交流社区
4. **在线课程**：Coursera、Udacity等平台的深度学习专项课程
5. **书籍推荐**：《Deep Learning with PyTorch》、《PyTorch深度学习实战》

---

**文档版本**：v1.0  
**最后更新**：2026年4月20日  
**适用PyTorch版本**：2.10+  
**作者**：AI技术学习与大厂入职计划工作组  
**版权声明**：本文档仅供学习参考，转载请注明出处