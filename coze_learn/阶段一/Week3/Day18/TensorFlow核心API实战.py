"""
TensorFlow 2.x 核心API实战教程
Day 18：TensorFlow核心API入门

本教程涵盖TensorFlow 2.x的核心API，包括：
1. Tensor创建与运算
2. 自动求导（GradientTape）
3. 模型构建（Sequential/Functional API）
4. 数据管道（tf.data）
5. 训练循环（compile/fit）
6. 模型保存与加载
7. 与PyTorch对比示例

所有代码均可直接运行，无需额外依赖。
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU可用: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# 1. Tensor创建与运算
# ============================================================================
print("\n" + "="*60)
print("1. Tensor创建与运算")
print("="*60)

# 1.1 创建Tensor
# 标量
scalar = tf.constant(3.0)
print(f"标量: {scalar}, 形状: {scalar.shape}, 数据类型: {scalar.dtype}")

# 向量
vector = tf.constant([1.0, 2.0, 3.0])
print(f"向量: {vector}, 形状: {vector.shape}")

# 矩阵
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(f"矩阵: {matrix}, 形状: {matrix.shape}")

# 随机Tensor
random_tensor = tf.random.normal(shape=(2, 3), mean=0.0, stddev=1.0)
print(f"随机Tensor: {random_tensor}")

# 1.2 Tensor运算
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 元素级运算
print(f"加法: {a + b}")
print(f"乘法: {a * b}")
print(f"矩阵乘法: {tf.matmul(a, b)}")

# 广播
c = tf.constant([1.0, 2.0])  # 形状 (2,)
print(f"广播加法: {a + c}")

# 1.3 Tensor操作
# 形状操作
tensor = tf.constant(np.arange(12).reshape(3, 4))
print(f"原始Tensor: {tensor}")
print(f"重塑为(4, 3): {tf.reshape(tensor, (4, 3))}")
print(f"转置: {tf.transpose(tensor)}")

# 切片
print(f"切片[0:2, 1:3]: {tensor[0:2, 1:3]}")

# 聚合操作
print(f"求和: {tf.reduce_sum(tensor)}")
print(f"平均值: {tf.reduce_mean(tensor)}")
print(f"最大值: {tf.reduce_max(tensor)}")

# ============================================================================
# 2. 自动求导（GradientTape）
# ============================================================================
print("\n" + "="*60)
print("2. 自动求导（GradientTape）")
print("="*60)

# 2.1 基础求导
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1  # y = x² + 2x + 1

grad = tape.gradient(y, x)
print(f"当 x={x.numpy()} 时，dy/dx = {grad.numpy()}")  # 应为 2x+2 = 8

# 2.2 多变量求导
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = x1**2 + x2**3  # z = x1² + x2³

grads = tape.gradient(z, [x1, x2])
print(f"梯度: ∂z/∂x1 = {grads[0].numpy()}, ∂z/∂x2 = {grads[1].numpy()}")

# 2.3 二阶导数
x = tf.Variable(2.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = x**3  # y = x³
    dy_dx = inner_tape.gradient(y, x)  # 一阶导数: 3x² = 12
d2y_dx2 = outer_tape.gradient(dy_dx, x)  # 二阶导数: 6x = 12

print(f"一阶导数 dy/dx = {dy_dx.numpy()}")
print(f"二阶导数 d²y/dx² = {d2y_dx2.numpy()}")

# ============================================================================
# 3. 模型构建（Sequential/Functional API）
# ============================================================================
print("\n" + "="*60)
print("3. 模型构建（Sequential/Functional API）")
print("="*60)

# 3.1 Sequential API（顺序模型）
sequential_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出层
])

print("Sequential模型结构:")
sequential_model.summary()

# 3.2 Functional API（函数式模型）
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)

functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
print("\nFunctional模型结构:")
functional_model.summary()

# ============================================================================
# 4. 数据管道（tf.data）
# ============================================================================
print("\n" + "="*60)
print("4. 数据管道（tf.data）")
print("="*60)

# 创建模拟数据
num_samples = 1000
X = np.random.randn(num_samples, 10).astype(np.float32)
y = np.random.randn(num_samples, 1).astype(np.float32)

# 创建tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=num_samples)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print(f"数据集大小: {num_samples}")
print(f"批次大小: 32")
print(f"总批次数: {len(list(dataset))}")

# 查看一个批次
for batch_x, batch_y in dataset.take(1):
    print(f"批次X形状: {batch_x.shape}, 批次y形状: {batch_y.shape}")
    break

# ============================================================================
# 5. 训练循环（compile/fit）
# ============================================================================
print("\n" + "="*60)
print("5. 训练循环（compile/fit）")
print("="*60)

# 5.1 编译模型
sequential_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae']  # 平均绝对误差
)

print("模型编译完成，使用Adam优化器和MSE损失函数")

# 5.2 训练模型（小规模演示）
# 使用小数据集快速演示
train_dataset = tf.data.Dataset.from_tensor_slices((X[:100], y[:100]))
train_dataset = train_dataset.batch(16).shuffle(100)

print("\n开始训练（演示用，仅5个epoch）...")
history = sequential_model.fit(
    train_dataset,
    epochs=5,
    verbose=1
)

# 5.3 可视化训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE', color='orange')
plt.title('MAE曲线')
plt.xlabel('Epoch')
plt.ylabel('平均绝对误差')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/阶段一/Week3/Day18/训练曲线.png')
print("训练曲线已保存: outputs/阶段一/Week3/Day18/训练曲线.png")

# ============================================================================
# 6. 模型保存与加载
# ============================================================================
print("\n" + "="*60)
print("6. 模型保存与加载")
print("="*60)

# 6.1 保存为SavedModel格式
save_path = 'outputs/阶段一/Week3/Day18/my_model'
sequential_model.save(save_path)
print(f"模型已保存为SavedModel格式: {save_path}")

# 6.2 加载模型
loaded_model = tf.keras.models.load_model(save_path)
print(f"模型已加载，结构相同: {loaded_model.summary()}")

# 6.3 保存为H5格式
h5_path = 'outputs/阶段一/Week3/Day18/my_model.h5'
sequential_model.save(h5_path)
print(f"模型已保存为H5格式: {h5_path}")

# 6.4 仅保存权重
weights_path = 'outputs/阶段一/Week3/Day18/model_weights.h5'
sequential_model.save_weights(weights_path)
print(f"模型权重已保存: {weights_path}")

# ============================================================================
# 7. 与PyTorch对比示例
# ============================================================================
print("\n" + "="*60)
print("7. TensorFlow与PyTorch对比示例")
print("="*60)

print("""
TensorFlow与PyTorch在设计哲学和API风格上存在显著差异：

1. **API设计**:
   - TensorFlow: 高度封装的Keras API，声明式编程风格
   - PyTorch: Pythonic设计，命令式编程风格

2. **执行模式**:
   - TensorFlow: 默认即时执行，可通过tf.function转换为图模式
   - PyTorch: 默认动态图，可通过torch.compile进行编译优化

3. **模型定义**:
   - TensorFlow: Sequential/Functional API，结构清晰
   - PyTorch: nn.Module子类化，灵活性强

4. **训练循环**:
   - TensorFlow: 高级API（model.fit）简化训练
   - PyTorch: 手动编写训练循环，控制精细

5. **部署生态**:
   - TensorFlow: TensorFlow Serving、TFLite、TF.js生态成熟
   - PyTorch: TorchServe、ExecuTorch快速发展

下面是相同功能的TensorFlow和PyTorch代码对比：
""")

print("\n【TensorFlow版本】 - 线性回归模型训练")
print("-"*40)

# TensorFlow线性回归示例
class LinearRegressionTF(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.linear = tf.keras.layers.Dense(1, input_shape=(1,))
    
    def call(self, x):
        return self.linear(x)

# 准备数据
X_tf = tf.constant(np.arange(10, dtype=np.float32).reshape(-1, 1))
y_tf = tf.constant(2 * np.arange(10, dtype=np.float32).reshape(-1, 1) + 1)

# 创建模型
model_tf = LinearRegressionTF()
model_tf.compile(optimizer='sgd', loss='mse')

# 训练
history_tf = model_tf.fit(X_tf, y_tf, epochs=10, verbose=0)
print(f"TensorFlow训练完成，最终损失: {history_tf.history['loss'][-1]:.4f}")

print("\n【PyTorch版本（伪代码）】 - 线性回归模型训练")
print("-"*40)
print("""
import torch
import torch.nn as nn

class LinearRegressionPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# 准备数据
X_pt = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
y_pt = 2 * torch.arange(10, dtype=torch.float32).reshape(-1, 1) + 1

# 创建模型
model_pt = LinearRegressionPT()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model_pt(X_pt)
    loss = criterion(outputs, y_pt)
    loss.backward()
    optimizer.step()

print(f"PyTorch训练完成，最终损失: {loss.item():.4f}")
""")

print("\n【关键差异总结】")
print("-"*40)
print("""
1. **梯度计算**:
   - TensorFlow: 使用GradientTape上下文管理器
   - PyTorch: 自动调用backward()方法

2. **优化器使用**:
   - TensorFlow: optimizer.apply_gradients()
   - PyTorch: optimizer.step()

3. **模型保存**:
   - TensorFlow: SavedModel格式（推荐）
   - PyTorch: .pt或.pth文件

4. **设备管理**:
   - TensorFlow: 自动使用GPU（如可用）
   - PyTorch: 需要显式调用.to(device)
""")

# ============================================================================
# 8. 综合实战：MNIST分类
# ============================================================================
print("\n" + "="*60)
print("8. 综合实战：MNIST手写数字分类")
print("="*60)

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

# 构建CNN模型
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

mnist_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("MNIST CNN模型结构:")
mnist_model.summary()

# 快速训练（演示用）
print("\n开始MNIST训练（演示用，仅2个epoch）...")
mnist_history = mnist_model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=2,
    batch_size=64,
    verbose=1
)

# 评估
test_loss, test_acc = mnist_model.evaluate(x_test, y_test, verbose=0)
print(f"测试准确率: {test_acc:.4f}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*60)
print("TensorFlow核心API实战总结")
print("="*60)

print("""
✅ **已掌握的核心技能**:

1. **Tensor操作**: 创建、运算、切片、聚合
2. **自动求导**: GradientTape基础与高阶用法
3. **模型构建**: Sequential和Functional API
4. **数据管道**: tf.data高效数据加载
5. **训练循环**: compile/fit高级API
6. **模型部署**: SavedModel、H5、权重保存
7. **框架对比**: 理解TensorFlow与PyTorch关键差异

📊 **实际应用场景**:
- 工业级模型部署（TensorFlow Serving、TFLite）
- 移动端AI应用（量化、硬件加速）
- 大规模分布式训练（TPU、多GPU）
- 研究到生产的完整流水线（TFX）

🔧 **后续学习方向**:
1. 深入TensorFlow Extended（TFX）MLOps平台
2. 掌握TensorFlow Lite模型量化与优化
3. 学习分布式训练策略（MirroredStrategy等）
4. 探索TensorFlow.js浏览器端AI

📁 **本教程生成的文件**:
- TensorFlow核心API实战.py（本文件）
- 训练曲线.png（训练过程可视化）
- my_model/（SavedModel格式）
- my_model.h5（H5格式）
- model_weights.h5（仅权重）

**所有代码均可直接运行，无需额外配置。**
""")

print("\n" + "="*60)
print("Day 18 TensorFlow核心API入门 - 实战完成")
print("="*60)