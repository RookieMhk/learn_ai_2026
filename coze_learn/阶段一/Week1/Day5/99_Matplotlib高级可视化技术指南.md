# Matplotlib高级可视化技术指南
**适用阶段：** Week 1 Day 5  
**核心目标：** 掌握Matplotlib高级功能，创建专业级数据可视化，理解AI可视化应用场景

---

## 🎯 学习指南概览
今日将系统学习Matplotlib的高级功能，涵盖子图布局、样式定制、图表类型选择原则，并深入分析AI可视化（特别是Transformer注意力机制可视化）的技术实现。

**你将掌握：**
- 复杂多图布局的设计与实现
- 专业图表的美学定制（色彩、线型、标记、字体）
- 数据特性与图表类型的匹配原则
- AI模型可视化专项技术（注意力热力图、特征激活图等）

---

## 📐 第一章：子图布局（Subplots）专业级应用

### 1.1 基础子图布局
Matplotlib提供从简单的`plt.subplots()`到复杂的`GridSpec`等多种布局方式。

**常用方法对比：**
| 方法 | 适用场景 | 代码示例 |
|------|---------|---------|
| `plt.subplots()` | 等分网格布局 | `fig, axs = plt.subplots(2, 3)` |
| `plt.subplot_mosaic()` | 不规则布局 | `mosaic = [[A, A], [B, C]]` |
| `GridSpec` | 精确控制位置和尺寸 | `gs = GridSpec(3, 3)` |
| `subplot2grid()` | 跨行列布局 | `ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)` |

### 1.2 GridSpec高级布局
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))

# 定义3行3列的网格
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 1], width_ratios=[2, 1, 1])

# 第一行跨三列
ax_main = fig.add_subplot(gs[0, :])
# 第二行左列
ax_left = fig.add_subplot(gs[1, 0])
# 第二行中右列
ax_right_top = fig.add_subplot(gs[1, 1])
ax_right_bottom = fig.add_subplot(gs[1, 2])
# 第三行跨三列
ax_bottom = fig.add_subplot(gs[2, :])

# 设置标题和标签
ax_main.set_title('主图区域', fontsize=14, fontweight='bold')
ax_left.set_title('左侧分析图', fontsize=12)
ax_right_top.set_title('右上角图', fontsize=12)
ax_right_bottom.set_title('右下角图', fontsize=12)
ax_bottom.set_title('底部总结图', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

**布局设计原则：**
1. **视觉层次**：重要图表占据更大空间，次要图表作为辅助
2. **阅读流线**：从左到右，从上到下，符合自然阅读习惯
3. **留白艺术**：适当留白提升可读性，避免信息过载

### 1.3 不规则布局实例：AI模型分析报告
```python
# AI模型性能对比报告布局
mosaic = '''
    AABB
    AABB
    CCCC
    DDDD
'''

fig, axs = plt.subplot_mosaic(mosaic, figsize=(14, 10))
axs['A'].set_title('模型准确率对比')
axs['B'].set_title('损失函数收敛曲线')
axs['C'].set_title('注意力权重分布热力图')
axs['D'].set_title('特征激活可视化')
```

---

## 🎨 第二章：样式定制与视觉美学

### 2.1 色彩系统与调色板
**Matplotlib内置色彩映射：**
- **Sequential（顺序）**：viridis, plasma, inferno, magma（适合数值大小表示）
- **Diverging（发散）**：RdBu, RdYlBu, coolwarm（适合正负差异）
- **Qualitative（分类）**：tab10, Set3, Pastel1（适合分类数据）

**自定义色彩映射：**
```python
from matplotlib.colors import LinearSegmentedColormap

# 创建蓝-白-红发散色彩映射
colors = ['blue', 'white', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
```

### 2.2 线型、标记与填充
**专业图表样式配置：**
```python
import matplotlib.pyplot as plt
import numpy as np

# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# 多线型设计
fig, ax = plt.subplots(figsize=(10, 6))

# 实线+圆形标记
ax.plot(x, y1, linestyle='-', marker='o', markersize=6, 
        label='正弦函数', linewidth=2, color='#1f77b4')

# 虚线+方形标记
ax.plot(x, y2, linestyle='--', marker='s', markersize=5,
        label='余弦函数', linewidth=2, color='#ff7f0e')

# 点划线+三角形标记
ax.plot(x, y3, linestyle='-.', marker='^', markersize=5,
        label='乘积函数', linewidth=2, color='#2ca02c')

# 区域填充（置信区间示例）
y_err = 0.2 * np.random.randn(100)
ax.fill_between(x, y1 - y_err, y1 + y_err, 
                alpha=0.3, color='#1f77b4', label='95%置信区间')

ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)
ax.set_title('函数可视化与置信区间', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()
```

### 2.3 字体与文本定制
```python
# 中文字体配置（必须）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 标题与标签的专业设置
ax.set_title('Transformer注意力机制分析', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('输入词元位置', fontsize=12, fontweight='medium')
ax.set_ylabel('注意力权重', fontsize=12, fontweight='medium')

# 文本标注
ax.annotate('最高关注点', xy=(5, 0.9), xytext=(7, 0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, fontweight='bold', color='red')
```

---

## 📊 第三章：图表类型选择与设计原则

### 3.1 数据特性与图表匹配矩阵
| 数据类型 | 分析目的 | 推荐图表 | Matplotlib函数 |
|---------|---------|---------|---------------|
| 连续变量分布 | 展示数据分布形状 | 直方图、密度图 | `hist()`, `kdeplot()` |
| 类别对比 | 比较不同类别数值 | 柱状图、条形图 | `bar()`, `barh()` |
| 时间序列 | 展示趋势与周期性 | 折线图、面积图 | `plot()`, `stackplot()` |
| 相关性 | 探索变量间关系 | 散点图、热力图 | `scatter()`, `imshow()` |
| 多变量 | 高维数据降维 | 平行坐标、雷达图 | `parallel_coordinates()` |

### 3.2 AI可视化专用图表
**注意力热力图（Attention Heatmap）：**
```python
def plot_attention_heatmap(attention_weights, tokens, layer_idx=0, head_idx=0):
    """
    可视化Transformer注意力权重
    
    参数：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    tokens: 词元列表
    layer_idx: 层索引
    head_idx: 头索引
    """
    import numpy as np
    
    # 提取特定层和头的注意力权重
    attn = attention_weights[layer_idx][head_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('注意力权重', rotation=90, va='bottom', fontsize=12)
    
    # 网格线
    ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_title(f'Transformer注意力热力图 (层{layer_idx+1}, 头{head_idx+1})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig
```

**神经网络特征激活图：**
```python
def plot_feature_activations(activations, layer_name, n_examples=5):
    """
    可视化神经网络特征激活
    
    参数：
    activations: 特征激活张量 [batch, channels, height, width]
    layer_name: 层名称
    n_examples: 显示示例数量
    """
    fig, axs = plt.subplots(n_examples, 1, figsize=(12, 2*n_examples))
    
    for i in range(min(n_examples, activations.shape[0])):
        # 计算通道平均激活
        channel_mean = activations[i].mean(dim=0)
        
        axs[i].imshow(channel_mean.cpu().numpy(), cmap='hot', 
                      aspect='auto', interpolation='nearest')
        axs[i].axis('off')
        axs[i].set_title(f'{layer_name} - 示例 {i+1}', fontsize=10)
    
    plt.suptitle(f'神经网络特征激活可视化 - {layer_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
```

### 3.3 专业图表设计规范
**黄金分割布局：**
- 图表宽高比建议1.618:1（黄金分割比例）
- 主标题字体大小为轴标签的1.5倍
- 图例置于图表外部，避免遮挡数据

**视觉编码一致性：**
- 同一数据系列使用相同颜色
- 分类数据使用分类色彩映射
- 连续数据使用顺序色彩映射

---

## 🤖 第四章：AI可视化专项技术

### 4.1 Transformer注意力机制可视化
**多头注意力分析：**
```python
def analyze_multihead_attention(attention_weights, tokens, 
                                query_idx=None, highlight_heads=None):
    """
    分析多头注意力模式
    
    参数：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    tokens: 词元列表
    query_idx: 特定查询位置（默认为[CLS]）
    highlight_heads: 需要高亮显示的头索引列表
    """
    if query_idx is None:
        query_idx = 0  # 默认分析[CLS] token
        
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    fig, axs = plt.subplots(n_layers, n_heads, 
                           figsize=(3*n_heads, 2*n_layers),
                           squeeze=False)
    
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axs[layer, head]
            
            # 提取该头的注意力权重
            attn = attention_weights[layer, head, query_idx]
            
            # 绘制条形图
            bars = ax.bar(range(seq_len), attn.cpu().numpy(), 
                         color='skyblue', edgecolor='navy')
            
            # 高亮重要头
            if highlight_heads and head in highlight_heads:
                bars[query_idx].set_color('red')
            
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_ylim([0, 1])
            ax.set_title(f'L{layer+1}H{head+1}', fontsize=8, fontweight='bold')
    
    plt.suptitle(f'多头注意力分析 (查询位置: {tokens[query_idx]})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
```

### 4.2 模型决策过程可视化
**梯度加权类激活映射（Grad-CAM）：**
```python
def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    """
    生成Grad-CAM可视化
    
    参数：
    model: PyTorch模型
    input_tensor: 输入张量
    target_layer: 目标层
    target_class: 目标类别（默认为预测类别）
    """
    import torch
    import torch.nn.functional as F
    
    model.eval()
    
    # 前向传播
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # 计算梯度
    model.zero_grad()
    output[0, target_class].backward()
    
    # 获取目标层的梯度和激活
    gradients = target_layer.weight.grad
    activations = target_layer.activation
    
    # 计算权重
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # 生成热力图
    gradcam = (weights * activations).sum(dim=1, keepdim=True)
    gradcam = F.relu(gradcam)
    gradcam = F.interpolate(gradcam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
    
    # 归一化
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    
    return gradcam.detach()
```

---

## 🛠 第五章：实战工作流与最佳实践

### 5.1 AI可视化项目工作流
1. **需求分析**：明确可视化目标（模型理解、性能展示、可解释性验证）
2. **数据准备**：提取模型中间表示（注意力权重、特征激活、梯度）
3. **可视化设计**：选择合适图表类型，设计布局和色彩方案
4. **代码实现**：使用Matplotlib高级功能实现设计
5. **效果优化**：调整视觉参数，提升信息传达效率
6. **成果交付**：生成高质量图像，撰写分析报告

### 5.2 性能优化技巧
**大数据集可视化：**
- 使用`rasterized=True`参数将部分元素栅格化
- 对于散点图，使用透明度或2D直方图
- 分批处理，避免一次性绘制过多元素

**交互式探索：**
- 结合`mplcursors`库添加悬停提示
- 使用`matplotlib.widgets`创建交互控件
- 对于复杂可视化，考虑使用Plotly等交互式库

### 5.3 可复用代码模板
提供AI可视化专用模板，包含注意力分析、特征可视化、模型对比等常用功能模块。

---

## 📝 总结与进阶方向

### 核心收获
1. **布局设计能力**：掌握从简单网格到复杂不规则布局的实现方法
2. **视觉定制技巧**：学会色彩、线型、字体等美学元素的专业配置
3. **AI可视化专项**：理解Transformer注意力机制可视化等AI特有需求
4. **工程化思维**：建立从数据到高质量可视化的完整工作流

### 后续学习建议
1. **扩展工具链**：学习Plotly、Bokeh等交互式可视化库
2. **深度学习可视化**：深入神经网络特征可视化、决策边界分析
3. **可解释性研究**：结合最新研究成果，实现更透明的AI模型分析
4. **项目实践**：将可视化技术应用于个人AI项目，提升展示效果

---

**附录：** 完整代码示例见`AI数据可视化案例研究.py`和`综合可视化项目.py`