# Week 3 综合知识测验

**测试说明：**
- 本测验包含20道题目，涵盖Week 3所有核心学习领域
- 题目类型包括：选择题、判断题、编程题
- 建议完成时间：60分钟
- 完成后可对照答案进行自我评估

---

## 第一部分：神经网络数学基础（Day15）

### 1. 选择题：前向传播计算
在神经网络中，给定输入向量 x = [1, 2]，权重矩阵 W = [[0.5, -0.3], [0.2, 0.4]]，偏置向量 b = [0.1, -0.2]，使用ReLU激活函数。请问经过第一层线性变换后的输出是多少？

A. [0.6, 0.8]
B. [0.8, 0.6]
C. [0.8, 0.8]
D. [0.6, 0.6]

**计算过程：**
z = W·x + b = [[0.5, -0.3], [0.2, 0.4]]·[1, 2] + [0.1, -0.2]
= [0.5*1 + (-0.3)*2 + 0.1, 0.2*1 + 0.4*2 + (-0.2)]
= [0.5 - 0.6 + 0.1, 0.2 + 0.8 - 0.2] = [0.0, 0.8]
ReLU激活后：max(0, 0.0) = 0, max(0, 0.8) = 0.8
所以输出为：[0.0, 0.8]

**答案：** C

### 2. 判断题：梯度消失问题
使用Sigmoid激活函数时，当输入值很大或很小时，梯度会接近0，导致梯度消失问题。

A. 正确
B. 错误

**解析：** Sigmoid函数的导数在输入值很大或很小时接近0，确实会导致梯度消失问题。

**答案：** A

### 3. 编程题：激活函数实现
请实现ReLU、Sigmoid和Tanh三种激活函数及其导数。

```python
import numpy as np

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU导数"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh导数"""
    return 1 - np.tanh(x) ** 2

# 测试
x = np.array([-2, -1, 0, 1, 2])
print("ReLU:", relu(x))
print("ReLU导数:", relu_derivative(x))
print("Sigmoid:", sigmoid(x))
print("Sigmoid导数:", sigmoid_derivative(x))
print("Tanh:", tanh(x))
print("Tanh导数:", tanh_derivative(x))
```

---

## 第二部分：反向传播与优化（Day16）

### 4. 选择题：反向传播链式法则
在反向传播中，对于损失函数L，权重W的梯度计算使用链式法则。如果某一层的输出为a = σ(z)，其中z = Wx + b，σ是激活函数，那么∂L/∂W的正确计算方式是：

A. ∂L/∂a · ∂a/∂z · ∂z/∂W
B. ∂L/∂z · ∂z/∂W
C. ∂L/∂a · ∂a/∂W
D. ∂L/∂x · ∂x/∂W

**解析：** 根据链式法则，∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W。

**答案：** A

### 5. 判断题：Adam优化器
Adam优化器结合了动量法和RMSprop的优点，同时计算梯度的一阶矩估计和二阶矩估计。

A. 正确
B. 错误

**解析：** Adam确实结合了动量法（一阶矩）和RMSprop（二阶矩）的优点。

**答案：** A

### 6. 编程题：梯度下降实现
实现一个简单的梯度下降算法，用于优化函数 f(x) = x² + 2x + 1。

```python
import numpy as np

def gradient_descent(learning_rate=0.1, iterations=100):
    """
    梯度下降优化 f(x) = x² + 2x + 1
    
    参数:
        learning_rate: 学习率
        iterations: 迭代次数
    """
    # 初始值
    x = 5.0  # 随机初始值
    history = []
    
    for i in range(iterations):
        # 计算梯度: f'(x) = 2x + 2
        gradient = 2 * x + 2
        
        # 更新参数
        x = x - learning_rate * gradient
        
        # 计算当前损失
        loss = x**2 + 2*x + 1
        
        history.append({
            'iteration': i + 1,
            'x': x,
            'gradient': gradient,
            'loss': loss
        })
        
        # 打印每10次迭代的结果
        if (i + 1) % 10 == 0:
            print(f"迭代 {i+1}: x = {x:.4f}, 梯度 = {gradient:.4f}, 损失 = {loss:.4f}")
    
    return history, x

# 运行梯度下降
history, final_x = gradient_descent(learning_rate=0.1, iterations=50)
print(f"\n最终结果: x = {final_x:.4f}, 最小值点应为 x = -1")
```

---

## 第三部分：PyTorch核心API（Day17）

### 7. 选择题：PyTorch张量操作
在PyTorch中，想要将两个张量在指定维度上连接，应该使用哪个函数？

A. torch.cat()
B. torch.stack()
C. torch.concat()
D. torch.merge()

**解析：** torch.cat()用于在现有维度上连接张量，torch.stack()会创建新维度。

**答案：** A

### 8. 判断题：自动求导
在PyTorch中，默认情况下张量不会跟踪梯度计算，需要显式设置 requires_grad=True。

A. 正确
B. 错误

**解析：** 正确，PyTorch张量默认requires_grad=False。

**答案：** A

### 9. 编程题：PyTorch模型定义
使用PyTorch定义一个简单的全连接神经网络，包含两个隐藏层，用于二分类任务。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """简单的全连接神经网络"""
    
    def __init__(self, input_size=10, hidden_size1=64, hidden_size2=32):
        super(SimpleNN, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)  # 二分类输出
        
        # 批量归一化
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc3(x)
        x = torch.sigmoid(x)  # 二分类使用sigmoid
        
        return x

# 测试模型
model = SimpleNN(input_size=20)
print("模型结构:")
print(model)

# 创建随机输入
input_tensor = torch.randn(4, 20)  # batch_size=4, input_size=20
print(f"\n输入形状: {input_tensor.shape}")

# 前向传播
output = model(input_tensor)
print(f"输出形状: {output.shape}")
print(f"输出值: {output}")
```

---

## 第四部分：TensorFlow核心API（Day18）

### 10. 选择题：TensorFlow 2.x特性
TensorFlow 2.x默认启用的是哪种执行模式？

A. 图执行模式
B. 即时执行模式（Eager Execution）
C. 静态图模式
D. 延迟执行模式

**解析：** TensorFlow 2.x默认启用即时执行模式。

**答案：** B

### 11. 判断题：Keras API
Keras是TensorFlow的高级API，可以简化神经网络的构建和训练过程。

A. 正确
B. 错误

**解析：** 正确，Keras已集成到TensorFlow中作为高级API。

**答案：** A

### 12. 编程题：TensorFlow模型构建
使用TensorFlow/Keras构建一个CNN模型，用于MNIST手写数字识别。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_model():
    """构建CNN模型用于MNIST分类"""
    
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(28, 28, 1)),
        
        # 第一卷积层
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # 第二卷积层
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# 构建模型
model = build_cnn_model()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型摘要
print("CNN模型结构:")
model.summary()

# 加载MNIST数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f"\n训练数据形状: {x_train.shape}")
print(f"测试数据形状: {x_test.shape}")
```

---

## 第五部分：CNN基础与图像分类（Day19）

### 13. 选择题：卷积层参数计算
对于一个输入图像大小为 32×32×3（高×宽×通道），使用16个3×3的卷积核，步长为1，填充为1，请问输出特征图的大小是多少？

A. 32×32×16
B. 30×30×16
C. 32×32×3
D. 30×30×3

**解析：** 由于填充为1，输入32×32经过3×3卷积后大小不变，通道数变为卷积核数量16。

**答案：** A

### 14. 判断题：池化层作用
池化层的主要作用是减少参数数量，防止过拟合，同时保持特征的位置不变性。

A. 正确
B. 错误

**解析：** 池化层确实可以减少参数和计算量，但会丢失位置信息，所以位置不变性说法不准确。

**答案：** B

### 15. 编程题：CNN特征可视化
编写代码可视化CNN中间层的特征图。

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def visualize_feature_maps(model, image, layer_names=None):
    """
    可视化CNN中间层的特征图
    
    参数:
        model: 训练好的模型
        image: 输入图像
        layer_names: 要可视化的层名称列表
    """
    if layer_names is None:
        # 默认可视化所有卷积层
        layer_names = [layer.name for layer in model.layers 
                      if 'conv' in layer.name]
    
    # 创建中间层输出模型
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # 获取中间层激活
    activations = activation_model.predict(image[np.newaxis, ...])
    
    # 可视化每个层的特征图
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]  # 特征图数量
        
        # 设置子图布局
        size = layer_activation.shape[1]
        n_cols = min(n_features, 8)  # 每行最多8个
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        fig.suptitle(f'Layer: {layer_name} ({n_features}个特征图)', fontsize=16)
        
        # 绘制每个特征图
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[row]
            
            # 获取特征图并归一化
            feature_map = layer_activation[0, :, :, i]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[row]
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

# 示例：使用预训练的VGG16模型
def example_visualization():
    # 加载预训练模型
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    
    # 加载示例图像
    from tensorflow.keras.preprocessing import image
    img_path = 'path/to/your/image.jpg'  # 替换为实际路径
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = keras.applications.vgg16.preprocess_input(img_array)
    
    # 可视化特征图
    visualize_feature_maps(model, img_array, ['block1_conv1', 'block2_conv1'])

# 注意：实际运行需要替换图像路径
print("特征可视化函数已定义，请提供图像路径运行示例。")
```

---

## 第六部分：RNN/LSTM基础与序列建模（Day20）

### 16. 选择题：LSTM门控机制
在LSTM中，哪个门负责决定细胞状态中哪些信息应该被遗忘？

A. 输入门
B. 遗忘门
C. 输出门
D. 更新门

**解析：** 遗忘门负责决定从细胞状态中丢弃哪些信息。

**答案：** B

### 17. 判断题：RNN梯度问题
RNN在处理长序列时容易遇到梯度爆炸问题，而LSTM通过门控机制有效缓解了这个问题。

A. 正确
B. 错误

**解析：** RNN主要问题是梯度消失，LSTM确实缓解了梯度消失问题。

**答案：** B

### 18. 编程题：LSTM时间序列预测
使用LSTM进行简单的时间序列预测。

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def lstm_time_series_forecasting():
    """LSTM时间序列预测示例"""
    
    # 生成示例数据：正弦波 + 噪声
    np.random.seed(42)
    time_steps = np.arange(0, 200, 0.1)
    data = np.sin(time_steps) + np.random.normal(0, 0.1, len(time_steps))
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建序列
    seq_length = 20
    X, y = create_sequences(data_scaled, seq_length)
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 调整形状以适应LSTM输入 [样本数, 时间步长, 特征数]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 构建LSTM模型
    model = keras.Sequential([
        layers.LSTM(50, activation='relu', return_sequences=True, 
                   input_shape=(seq_length, 1)),
        layers.LSTM(50, activation='relu'),
        layers.Dense(1)
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 反归一化
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练损失')
    plt.xlabel('轮次')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 预测结果
    plt.subplot(1, 2, 2)
    plt.plot(y_test_original, label='真实值', alpha=0.7)
    plt.plot(y_pred_original, label='预测值', alpha=0.7)
    plt.title('时间序列预测结果')
    plt.xlabel('时间步')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    print(f"预测性能评估:")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")
    
    return model, scaler, history

# 运行LSTM预测
print("开始LSTM时间序列预测...")
model, scaler, history = lstm_time_series_forecasting()
```

---

## 第七部分：综合应用题

### 19. 综合题：模型选择与优化
假设你需要构建一个系统来识别社交媒体图片中的情感（正面、负面、中性）。请回答以下问题：

1. **模型选择**：你会选择CNN还是RNN？为什么？
2. **架构设计**：描述你的模型架构（层数、每层类型、激活函数等）
3. **优化策略**：你会使用哪些优化技术来提升模型性能？
4. **评估指标**：你会使用哪些指标来评估模型性能？

**参考答案：**

1. **模型选择**：选择CNN。因为情感识别主要基于图像内容，CNN擅长提取图像的局部特征和空间层次结构，而RNN更适合处理序列数据。

2. **架构设计**：
   - 输入层：224×224×3
   - 卷积层1：32个3×3卷积核，ReLU激活，批归一化，最大池化2×2
   - 卷积层2：64个3×3卷积核，ReLU激活，批归一化，最大池化2×2
   - 卷积层3：128个3×3卷积核，ReLU激活，批归一化，最大池化2×2
   - 展平层
   - 全连接层1：256个神经元，ReLU激活，Dropout 0.5
   - 全连接层2：128个神经元，ReLU激活，Dropout 0.3
   - 输出层：3个神经元，Softmax激活

3. **优化策略**：
   - 使用Adam优化器，学习率衰减
   - 数据增强（旋转、翻转、缩放）
   - 早停法防止过拟合
   - 学习率调度器
   - 迁移学习（使用预训练的ResNet/VGG作为特征提取器）

4. **评估指标**：
   - 准确率
   - 精确率、召回率、F1分数（针对每个类别）
   - 混淆矩阵
   - ROC曲线和AUC（针对二分类可扩展为多分类）

### 20. 综合题：实际问题解决
你在一家电商公司工作，需要构建一个推荐系统。用户的历史行为数据包括：
- 浏览记录（商品ID序列）
- 购买记录（商品ID序列）
- 停留时间（每个商品的停留秒数）

请设计一个深度学习模型来预测用户可能感兴趣的下一个商品。

**参考答案：**

**模型设计：基于注意力机制的序列推荐模型**

1. **输入处理**：
   - 商品ID嵌入层：将商品ID映射为稠密向量
   - 时间特征嵌入：停留时间归一化后作为额外特征

2. **模型架构**：
   - **嵌入层**：商品ID → 128维向量
   - **位置编码**：加入序列位置信息
   - **多头自注意力层**：捕捉商品间的复杂关系
   - **前馈网络层**：非线性变换
   - **层归一化**：稳定训练
   - **输出层**：全连接层 + Softmax，预测下一个商品概率

3. **关键技术**：
   - **注意力机制**：让模型关注对预测最重要的历史行为
   - **残差连接**：缓解梯度消失问题
   - **掩码注意力**：确保只能看到历史信息，不能看到未来信息

4. **训练策略**：
   - 负采样：随机采样未交互商品作为负样本
   - 序列截断/填充：处理变长序列
   - 多任务学习：同时预测浏览和购买行为

5. **评估方法**：
   - Top-K准确率（K=1,5,10）
   - 平均倒数排名（MRR）
   - 归一化折损累计增益（NDCG）

---

## 测验答案与评分标准

### 评分标准：
- 选择题（1-17题）：每题5分，共85分
- 编程题（18题）：15分
- 总分：100分

### 答案汇总：
1. C
2. A
3. 编程题（代码正确即可）
4. A
5. A
6. 编程题（代码正确即可）
7. A
8. A
9. 编程题（代码正确即可）
10. B
11. A
12. 编程题（代码正确即可）
13. A
14. B
15. 编程题（代码正确即可）
16. B
17. B
18. 编程题（代码正确即可）
19. 综合题（回答合理即可）
20. 综合题（回答合理即可）

### 成绩等级：
- 90-100分：优秀，Week 3掌握得很好
- 75-89分：良好，需要复习个别知识点
- 60-74分：及格，建议系统复习核心概念
- 60分以下：需要重新学习Week 3内容

---

**学习建议：**
1. 对于错题，回顾相关学习材料
2. 编程题要多实践，理解每个函数的作用
3. 综合题要注重实际应用场景的思考
4. 建议将本测验作为复习材料，定期回顾

祝学习进步！