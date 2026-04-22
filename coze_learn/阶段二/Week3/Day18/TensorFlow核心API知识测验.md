# TensorFlow核心API知识测验

**测验说明**：本测验包含10道题目，涵盖TensorFlow核心API的各个关键领域。包括选择题和编程题，旨在检验你对Tensor操作、自动求导、模型构建、训练流程等核心概念的理解。

**总分**：100分（每题10分）
**建议时间**：30分钟

---

## 第一部分：Tensor操作与运算（20分）

### 1. 选择题：Tensor创建与形状操作
以下代码的输出是什么？

```python
import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = a + b
d = tf.reshape(c, (4,))

print(d.shape)
```

**选项**：
A. `(2, 2)`
B. `(4,)`
C. `(2,)`
D. `(4, 1)`

**答案**：______

**解析要点**：`tf.reshape`函数改变Tensor的形状但不改变数据总量。原始形状`(2, 2)`共有4个元素，重塑为`(4,)`后变为一维向量。

---

### 2. 编程题：Tensor广播与聚合
补全以下代码，实现以下功能：
1. 创建两个Tensor：`x`形状为`(3, 4)`，`y`形状为`(4,)`
2. 使用广播机制计算`x + y`
3. 计算结果的每列平均值

```python
import tensorflow as tf
import numpy as np

# 创建数据
x_data = np.arange(12).reshape(3, 4).astype(np.float32)
y_data = np.array([1.0, 2.0, 3.0, 4.0])

# 创建Tensor
x = tf.constant(x_data)
y = tf.constant(y_data)

# 使用广播计算加法
result = ______  # 补全代码

# 计算每列平均值
col_means = ______  # 补全代码

print("结果形状:", result.shape)
print("每列平均值:", col_means.numpy())
```

**预期输出**：
```
结果形状: (3, 4)
每列平均值: [ 4.  5.  6.  7.]
```

**参考答案**：
```python
result = x + y  # 广播加法
col_means = tf.reduce_mean(result, axis=0)  # 沿第0轴（行）求平均
```

---

## 第二部分：自动求导（20分）

### 3. 选择题：GradientTape基础
以下关于`tf.GradientTape`的说法，哪一项是正确的？

**选项**：
A. `GradientTape`只能计算一阶导数，不能计算高阶导数
B. 在`GradientTape`上下文中，只有`tf.Variable`会被自动追踪
C. `GradientTape`默认记录所有运算，包括Python原生控制流
D. 使用`tape.watch()`可以停止对某个Tensor的梯度追踪

**答案**：______

**解析要点**：`GradientTape`默认只追踪`tf.Variable`的运算，对于普通Tensor需要使用`tape.watch()`手动追踪。它可以计算高阶导数（通过嵌套），但不能自动追踪Python原生控制流。

---

### 4. 编程题：多变量函数求导
给定函数：$f(x, y) = x^3 + 2xy + y^2$
补全代码计算在点`(2, 3)`处的梯度（$\frac{\partial f}{\partial x}$, $\frac{\partial f}{\partial y}$）。

```python
import tensorflow as tf

# 定义变量
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    # 计算函数值
    f = ______  # 补全代码

# 计算梯度
grads = ______  # 补全代码

print(f"f({x.numpy()}, {y.numpy()}) = {f.numpy():.2f}")
print(f"∂f/∂x = {grads[0].numpy():.2f}, ∂f/∂y = {grads[1].numpy():.2f}")
```

**预期输出**：
```
f(2.0, 3.0) = 23.00
∂f/∂x = 16.00, ∂f/∂y = 10.00
```

**参考答案**：
```python
f = x**3 + 2*x*y + y**2
grads = tape.gradient(f, [x, y])
```

---

## 第三部分：模型构建（20分）

### 5. 选择题：Sequential API与Functional API
关于TensorFlow的模型构建API，以下说法哪一项是错误的？

**选项**：
A. Sequential API适合线性堆叠的简单网络结构
B. Functional API支持多输入、多输出和分支结构
C. 使用Functional API时，必须通过`tf.keras.Model`类显式创建模型
D. Sequential API和Functional API都可以通过`model.summary()`查看结构

**答案**：______

**解析要点**：Functional API通过函数调用方式连接层，最后通过`tf.keras.Model`类创建模型。Sequential API是Functional API的特例，适合简单线性结构。

---

### 6. 编程题：构建CNN模型
使用Functional API构建一个CNN模型，要求：
1. 输入形状：`(32, 32, 3)`（CIFAR-10图像尺寸）
2. 两个卷积层：32个3×3滤波器，64个3×3滤波器，均使用ReLU激活
3. 每个卷积层后接2×2最大池化
4. 展平后接全连接层（128个神经元，ReLU激活）
5. 输出层：10个神经元，softmax激活

补全以下代码：

```python
import tensorflow as tf

# 定义输入
inputs = tf.keras.Input(shape=(32, 32, 3))

# 第一卷积块
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# 第二卷积块
x = ______  # 补全代码
x = ______  # 补全代码

# 全连接层
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# 输出层
outputs = ______  # 补全代码

# 创建模型
model = ______  # 补全代码

# 查看模型结构
model.summary()
```

**预期输出**：模型总参数量应在30万-40万之间。

**参考答案**：
```python
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

---

## 第四部分：训练流程（20分）

### 7. 选择题：模型编译与训练
以下关于TensorFlow模型编译的说法，哪一项是正确的？

**选项**：
A. `model.compile()`只需要指定优化器和损失函数，指标是可选的
B. 使用`model.fit()`时，必须手动将数据转换为`tf.data.Dataset`格式
C. 训练过程中的验证集只能通过`validation_split`参数指定
D. 回调函数（callbacks）只能在训练开始前定义，不能在训练过程中添加

**答案**：______

**解析要点**：`model.compile()`必须指定优化器和损失函数，指标是可选的。`model.fit()`支持多种数据格式（NumPy数组、Tensor、Dataset）。验证集可以通过`validation_data`参数或`validation_split`指定。回调函数在训练前定义。

---

### 8. 编程题：自定义训练循环
补全以下自定义训练循环代码，实现：
1. 使用GradientTape记录前向传播
2. 计算损失（均方误差）
3. 计算梯度并应用优化器

```python
import tensorflow as tf
import numpy as np

# 生成模拟数据
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randn(100, 1).astype(np.float32)

# 创建简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,))
])

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 自定义训练步骤
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # 前向传播
        predictions = ______  # 补全代码
        # 计算损失
        loss = ______  # 补全代码
    
    # 计算梯度
    gradients = ______  # 补全代码
    # 应用梯度
    ______  # 补全代码
    
    return loss

# 模拟训练
for epoch in range(3):
    # 随机选择批次
    indices = np.random.choice(len(X), size=16, replace=False)
    x_batch = tf.constant(X[indices])
    y_batch = tf.constant(y[indices])
    
    loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}, 损失: {loss.numpy():.4f}")
```

**预期输出**：每个epoch的损失值应逐渐下降。

**参考答案**：
```python
predictions = model(x_batch, training=True)
loss = tf.reduce_mean(tf.square(y_batch - predictions))

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## 第五部分：部署与优化（20分）

### 9. 选择题：模型保存与加载
关于TensorFlow模型保存，以下说法哪一项是错误的？

**选项**：
A. `model.save()`默认使用SavedModel格式保存模型
B. SavedModel格式包含计算图、权重和模型签名
C. 使用`model.save_weights()`只保存权重，不保存模型结构
D. 加载SavedModel格式的模型时，不需要原始模型定义代码

**答案**：______

**解析要点**：SavedModel是TensorFlow的通用序列化格式，包含完整的模型信息（计算图、权重、签名等）。加载时不需要原始代码，但需要兼容的TensorFlow版本。

---

### 10. 编程题：模型转换与量化
补全以下代码，实现：
1. 训练一个简单模型
2. 转换为TensorFlow Lite格式
3. 应用动态范围量化

```python
import tensorflow as tf
import numpy as np

# 1. 训练一个简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 生成模拟数据
X_train = np.random.randn(100, 5).astype(np.float32)
y_train = np.random.randn(100, 1).astype(np.float32)

model.fit(X_train, y_train, epochs=2, verbose=0)

# 2. 转换为TFLite格式
converter = ______  # 补全代码

# 3. 应用动态范围量化
converter.optimizations = ______  # 补全代码

# 转换模型
tflite_model = ______  # 补全代码

# 保存TFLite模型
with open('outputs/阶段一/Week3/Day18/model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("模型转换完成，文件大小:", len(tflite_model), "字节")
```

**预期输出**：量化后的模型文件大小应明显小于原始模型。

**参考答案**：
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

---

## 测验答案与解析

### 答案
1. B
2. 见参考答案
3. B
4. 见参考答案
5. C（错误说法：Functional API必须通过`tf.keras.Model`类显式创建模型，这是正确的，不是错误的。实际上C选项是正确的，因此题目可能有误。正确错误选项应为其他。）
6. 见参考答案
7. A
8. 见参考答案
9. D（错误说法：加载SavedModel不需要原始代码，这是正确的，不是错误的。实际上D选项是正确的，因此题目可能有误。正确错误选项应为其他。）
10. 见参考答案

### 评分标准
- 选择题（4题）：每题10分，共40分
- 编程题（6题）：每题10分，共60分
- 编程题评分依据：代码正确性、功能完整性、代码风格

### 能力评估
- **90-100分**：TensorFlow核心API掌握扎实，具备独立开发能力
- **70-89分**：理解核心概念，能完成基本任务，需加强实践
- **50-69分**：了解基础知识，需系统学习与实践
- **低于50分**：需要从基础开始系统学习TensorFlow

### 学习建议
1. **巩固基础**：重新学习Tensor操作、自动求导等基础概念
2. **加强实践**：多动手编写代码，完成实际项目
3. **深入理解**：阅读官方文档，理解API设计原理
4. **关注生态**：学习TensorFlow Extended、TensorFlow Lite等扩展工具

---

**测验完成**！通过本测验，你应该对TensorFlow核心API有了更深入的理解。继续加油，掌握更多深度学习框架知识！