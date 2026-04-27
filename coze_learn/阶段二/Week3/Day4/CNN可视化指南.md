# CNN可视化指南

> 通过可视化手段，直观理解卷积神经网络"看到了什么"

## 目录
1. [为什么需要可视化？](#1-为什么需要可视化)
2. [卷积核可视化](#2-卷积核可视化)
3. [特征图可视化](#3-特征图可视化)
4. [激活热力图](#4-激活热力图)
5. [CAM 类激活映射](#5-cam-类激活映射)
6. [Deep Dream 效果](#6-deep-dream-效果)
7. [实践练习](#7-实践练习)

---

## 1. 为什么需要可视化？

### 1.1 深度学习的"黑箱"问题

CNN 被称为"黑箱"模型，因为：
- 传统的全连接网络很难解释
- 百万级参数难以直观理解
- 我们不知道网络实际学习到了什么

### 1.2 可视化的价值

```
输入图像 → [卷积层] → [池化层] → ... → 分类输出
              ↓           ↓
         卷积核可视化   特征图可视化
              ↓           ↓
         理解特征提取   理解信息流动
```

**可视化帮助我们**：
- 诊断模型学习是否正常
- 理解网络学到了什么特征
- 发现潜在问题（过拟合、欠拟合）
- 向非技术人员解释模型

---

## 2. 卷积核可视化

### 2.1 什么是卷积核？

卷积核（Convolution Kernel/Filter）是 CNN 中用于提取特征的权重矩阵：

```
3×3 卷积核示例：
[[ 1,  0, -1],
 [ 2,  0, -2],
 [ 1,  0, -1]]  ← 边缘检测核

[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]  ← 锐化核
```

### 2.2 可视化代码

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def visualize_conv_filters(model, layer_name):
    """
    可视化指定卷积层的卷积核
    """
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data
            
            n_filters = min(16, weights.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(n_filters):
                # 获取第 i 个卷积核
                w = weights[i]
                
                # 如果是 RGB 输入 (C×H×W)，显示第一个通道
                if w.shape[0] == 3:
                    w = w[0].cpu().numpy()  # 取 R 通道
                else:
                    w = w[0].cpu().numpy()
                
                axes[i].imshow(w, cmap='RdBu', vmin=-1, vmax=1)
                axes[i].set_title(f'Filter {i}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

# 使用示例
model = SimpleCNN()
visualize_conv_filters(model, 'conv1.0')  # conv1 的第一个卷积层
```

### 2.3 可视化结果解读

| 层级 | 卷积核特点 | 解释 |
|------|------------|------|
| **浅层** (Conv1) | 颜色斑块、简单边缘 | 提取低级特征（颜色、边缘） |
| **中层** (Conv2-3) | 纹理、形状片段 | 提取中级特征（纹理、局部形状） |
| **深层** (Conv4-5) | 复杂纹理模式 | 提取高级特征（部件、物体组件） |

### 2.4 常见卷积核模式

```
【浅层卷积核的典型模式】

边缘检测型：
  水平边缘:     垂直边缘:
  [[ 1, 1, 1],  [[ 1, 0, -1],
   [ 0, 0, 0],   [ 1, 0, -1],
   [-1,-1,-1]]   [ 1, 0, -1]]

颜色斑块型：
  单一颜色:     颜色组合:
  [[ 1, 1, 1],  [[ 1, 0, 0],
   [ 1, 1, 1],   [ 0, 0, 0],
   [ 1, 1, 1]]   [ 0, 0, 1]]
```

---

## 3. 特征图可视化

### 3.1 什么是特征图？

特征图（Feature Map）是卷积层输出的中间结果：

```
输入图像 (3×H×W)
       ↓
    Conv2d(3, 32, 3×3)
       ↓
特征图 (32×H'×W')  ← 32 个通道，每个通道是一张激活图
```

### 3.2 可视化代码

```python
import torch

def visualize_feature_maps(model, image_tensor):
    """
    可视化网络中各层的特征图
    """
    model.eval()
    
    # 存储中间层激活
    activations = {}
    
    def hook_fn(module, input, output, name):
        activations[name] = output.detach()
    
    # 注册 hook
    hooks = []
    for name, module in model.named_modules():
        if 'conv' in name or 'pool' in name:
            hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
            )
    
    # 前向传播
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    
    # 移除 hook
    for hook in hooks:
        hook.remove()
    
    # 可视化
    for layer_name, activation in activations.items():
        # activation: (batch, channels, H, W)
        if len(activation.shape) == 4:
            n_show = min(16, activation.shape[1])
            
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(n_show):
                ax = axes[i // 8, i % 8]
                fmap = activation[0, i].cpu().numpy()
                ax.imshow(fmap, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Ch{i}')
            
            plt.suptitle(f'Feature Maps: {layer_name}', fontsize=14)
            plt.tight_layout()
            plt.show()
```

### 3.3 特征图解读

```
【从输入到输出的特征图变化】

输入层 (RGB):
┌─────────────────────┐
│    原始图像          │
│    3×224×224        │
└─────────────────────┘
         ↓ Conv1 (边缘检测)
┌─────────────────────┐
│  边缘特征图          │
│  64×112×112         │
└─────────────────────┘
         ↓ Conv2-3 (纹理)
┌─────────────────────┐
│  纹理特征图          │
│  128×56×56          │
└─────────────────────┘
         ↓ Conv4-5 (语义)
┌─────────────────────┐
│  物体部件特征图       │
│  256×28×28          │
└─────────────────────┘
         ↓ ...
┌─────────────────────┐
│  类别相关特征        │
│  512×7×7            │
└─────────────────────┘
```

### 3.4 特征图随层深变化

| 层级 | 特征图特点 | 含义 |
|------|------------|------|
| **浅层** | 清晰的边缘和颜色区块 | 保留原始信息，提取低级特征 |
| **中层** | 纹理和形状轮廓 | 组合低级特征形成中级特征 |
| **深层** | 模糊的语义区域 | 关注与任务相关的语义信息 |

---

## 4. 激活热力图

### 4.1 什么是激活热力图？

激活热力图显示网络中每个神经元的激活程度：
- 暖色（红/黄）：高激活 → 该特征被强烈检测到
- 冷色（蓝/绿）：低激活 → 该特征未被检测到

### 4.2 可视化代码

```python
import torch
import cv2
import numpy as np

def plot_activation_heatmap(model, image_tensor, layer_name):
    """
    绘制特定层的激活热力图
    """
    model.eval()
    
    # 获取目标层的输出
    x = image_tensor.unsqueeze(0)
    
    for name, module in model.named_modules():
        if name == layer_name:
            x = module(x)
            break
    
    # 计算平均激活
    activation = x[0].mean(dim=0).cpu().numpy()
    
    # 归一化到 [0, 1]
    activation = (activation - activation.min()) / (activation.max() - activation.min())
    
    # 上采样到原始图像大小
    activation = cv2.resize(activation, (224, 224))
    
    # 绘制
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.cpu().numpy().transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(activation, cmap='jet', alpha=0.7)
    plt.title(f'Activation Heatmap: {layer_name}')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 叠加图
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.imshow(activation, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    plt.show()
```

---

## 5. CAM 类激活映射

### 5.1 什么是 CAM？

CAM（Class Activation Mapping，类激活映射）是一种可视化技术，显示网络关注图像的哪些区域来进行分类。

### 5.2 CAM 原理

```
CAM 生成过程：

1. 全局平均池化后的特征图 (512×7×7)
         ↓
2. 加权求和 (每个类别的权重 × 特征图)
         ↓
3. ReLU 激活
         ↓
4. 上采样到原图大小
         ↓
5. 叠加在原图上
```

### 5.3 Grad-CAM 代码实现

```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册 hook
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 加权激活图
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()


def visualize_cam(model, image_tensor, image_path, target_layer_name):
    """
    可视化 Grad-CAM
    """
    # 初始化 Grad-CAM
    for name, module in model.named_modules():
        if name == target_layer_name:
            gradcam = GradCAM(model, module)
            break
    
    # 生成 CAM
    heatmap = gradcam.generate_cam(image_tensor)
    
    # 上采样
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # 转换为彩色
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 读取原图
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # 叠加
    superimposed = (heatmap * 0.4 + img * 0.6).astype(np.uint8)
    
    # 显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### 5.4 CAM 结果解读

```
【Grad-CAM 可视化结果示例】

分类目标: "猫"

        原始图像            Grad-CAM            叠加效果
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │              │    │  ████  ████  │    │  ████  ████  │
    │    🐱        │ →  │ ████████  █ │ →  │ █猫咪███  █ │
    │              │    │  ████  ████  │    │  ████  ████  │
    └──────────────┘    └──────────────┘    └──────────────┘
    
    网络关注的区域用暖色标记
    
    → 可以清楚地看到网络在识别"猫"时，
      主要关注了猫的脸部和身体轮廓
```

---

## 6. Deep Dream 效果

### 6.1 什么是 Deep Dream？

Deep Dream 通过最大化特定层的激活来生成梦幻般的图像效果。

### 6.2 Deep Dream 代码

```python
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DeepDream:
    """
    简化版 Deep Dream 实现
    """
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activation = None
        
        # 注册 hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.save_activation)
    
    def save_activation(self, module, input, output):
        self.activation = output
    
    def dream(self, image_tensor, iterations=20, lr=0.1):
        """
        生成 Deep Dream 图像
        """
        image = image_tensor.clone().requires_grad_(True)
        
        for i in range(iterations):
            self.model(image)
            
            # 最大化激活
            loss = -self.activation.mean()
            loss.backward()
            
            # 梯度上升
            image.data += lr * image.grad.data
            image.grad.zero_()
            
            # 每次迭代后归一化
            image.data = (image - image.mean()) / (image.std() + 1e-8)
        
        return image.detach()


def apply_deep_dream(model, image_path, layer_name='conv1.0'):
    """
    对图像应用 Deep Dream 效果
    """
    # 加载并预处理图像
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(img)
    
    # 创建 Deep Dream 对象
    dreamer = DeepDream(model, layer_name)
    
    # 生成 Deep Dream 图像
    result = dreamer.dream(image_tensor, iterations=50)
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    result = result * std + mean
    result = torch.clamp(result, 0, 1)
    
    # 显示
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img.resize((224, 224)))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(result.cpu().numpy().transpose(1, 2, 0))
    axes[1].set_title(f'Deep Dream ({layer_name})')
    axes[1].axis('off')
    
    plt.show()
```

---

## 7. 实践练习

### 7.1 练习 1：观察不同层的特征图

```python
# 加载预训练模型
model = SimpleCNN()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint)

# 加载一张图像
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('test_image.jpg')
img_tensor = transform(img)

# 观察不同层的特征图
for layer_name in ['conv1', 'conv2', 'conv3']:
    visualize_feature_maps(model, img_tensor, layer_name)
```

**观察要点**：
- 浅层特征图是否保留了原始图像的纹理？
- 深层特征图是否变得抽象和语义化？
- 哪些通道激活最强？它们对应什么特征？

### 7.2 练习 2：比较不同架构的特征图

```python
# 比较 VGG 和 ResNet 的特征图
model_vgg = models.vgg16(pretrained=True)
model_resnet = models.resnet18(pretrained=True)

# 观察它们的中间层激活
# VGG 浅层特征更清晰
# ResNet 通过残差连接保留更多信息
```

### 7.3 练习 3：Grad-CAM 可视化分类决策

```python
# 对分类错误的样本进行 Grad-CAM 分析
# 找出网络误判的原因
# 是关注了错误的区域？还是特征混淆？
```

---

## 8. 总结

### 可视化技术对比

| 技术 | 用途 | 优点 | 缺点 |
|------|------|------|------|
| **卷积核可视化** | 理解特征提取器 | 直观展示权重 | 深层核难以解释 |
| **特征图可视化** | 理解信息流 | 展示激活分布 | 需要想象力 |
| **激活热力图** | 定位高激活区域 | 直观定位 | 无法解释为什么 |
| **Grad-CAM** | 解释分类决策 | 可定位决策依据 | 需要梯度计算 |
| **Deep Dream** | 理解语义表示 | 艺术效果强 | 偏向娱乐 |

### 实践建议

1. **训练时**：定期可视化中间层，确保网络在学习有意义的特征
2. **调试时**：通过可视化发现问题（如卷积核崩溃、特征图稀疏）
3. **解释时**：使用 Grad-CAM 向他人解释模型决策
4. **调优时**：观察哪些通道重要，考虑 channel pruning

---

## 下一步

📝 **代码实现**：参考 `CNN图像分类项目.py` 中的可视化函数

📋 **执行练习**：使用 `Day4今日执行卡片模板.md` 完成今日实践任务
