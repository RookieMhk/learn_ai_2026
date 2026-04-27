"""
TensorFlow vs PyTorch 核心API对比示例
Day 18：框架对比实战

本文件展示TensorFlow和PyTorch在相同任务上的实现差异，
涵盖：模型定义、训练循环、自动求导、数据加载等核心概念。

注意：运行PyTorch部分需要安装torch库：
    pip install torch torchvision
"""

import numpy as np

# ============================================================================
# 1. 模型定义对比
# ============================================================================
print("="*60)
print("1. 模型定义对比")
print("="*60)

print("\n【任务】定义一个简单的全连接神经网络：")
print("  输入层: 10个特征")
print("  隐藏层: 64个神经元, ReLU激活")
print("  输出层: 1个神经元")

# ------------------------------------------------------------
# 1.1 TensorFlow实现
print("\n" + "-"*40)
print("TensorFlow实现 (Sequential API)")
print("-"*40)

import tensorflow as tf

def create_model_tf():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    return model

model_tf = create_model_tf()
print("TensorFlow模型结构:")
model_tf.summary()

# ------------------------------------------------------------
# 1.2 PyTorch实现
print("\n" + "-"*40)
print("PyTorch实现 (nn.Module子类化)")
print("-"*40)

try:
    import torch
    import torch.nn as nn
    
    class SimpleModelPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(10, 64)
            self.output = nn.Linear(64, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            return x
    
    model_pt = SimpleModelPT()
    print("PyTorch模型结构:")
    print(model_pt)
    print(f"总参数量: {sum(p.numel() for p in model_pt.parameters())}")
    
except ImportError:
    print("PyTorch未安装，跳过PyTorch代码执行")
    print("安装命令: pip install torch torchvision")

# ============================================================================
# 2. 自动求导对比
# ============================================================================
print("\n" + "="*60)
print("2. 自动求导对比")
print("="*60)

print("\n【任务】计算 y = x² 在 x=3 处的导数")

# ------------------------------------------------------------
# 2.1 TensorFlow实现
print("\n" + "-"*40)
print("TensorFlow实现 (GradientTape)")
print("-"*40)

x_tf = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y_tf = x_tf ** 2

grad_tf = tape.gradient(y_tf, x_tf)
print(f"TensorFlow结果: dy/dx = {grad_tf.numpy()} (预期: 6.0)")

# ------------------------------------------------------------
# 2.2 PyTorch实现
print("\n" + "-"*40)
print("PyTorch实现 (autograd)")
print("-"*40)

try:
    import torch
    
    x_pt = torch.tensor(3.0, requires_grad=True)
    y_pt = x_pt ** 2
    
    y_pt.backward()
    grad_pt = x_pt.grad
    
    print(f"PyTorch结果: dy/dx = {grad_pt.item()} (预期: 6.0)")
    
except ImportError:
    print("PyTorch未安装，跳过PyTorch代码执行")

# ============================================================================
# 3. 训练循环对比
# ============================================================================
print("\n" + "="*60)
print("3. 训练循环对比")
print("="*60)

print("\n【任务】使用随机数据训练一个简单的线性回归模型")

# 生成数据
np.random.seed(42)
X_data = np.random.randn(100, 10).astype(np.float32)
y_data = np.random.randn(100, 1).astype(np.float32)

# ------------------------------------------------------------
# 3.1 TensorFlow实现（高级API）
print("\n" + "-"*40)
print("TensorFlow实现 (Keras高级API)")
print("-"*40)

# 创建模型
model_tf_train = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,))
])

# 编译
model_tf_train.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mse'
)

print("模型编译完成，开始训练...")
history = model_tf_train.fit(
    X_data, y_data,
    epochs=5,
    batch_size=16,
    verbose=0
)

print(f"训练完成，最终损失: {history.history['loss'][-1]:.4f}")

# ------------------------------------------------------------
# 3.2 TensorFlow实现（自定义训练循环）
print("\n" + "-"*40)
print("TensorFlow实现 (自定义训练循环)")
print("-"*40)

# 重新创建模型
model_tf_custom = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,))
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 自定义训练步骤
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model_tf_custom(x_batch, training=True)
        loss = tf.reduce_mean(tf.square(y_batch - predictions))
    
    gradients = tape.gradient(loss, model_tf_custom.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_tf_custom.trainable_variables))
    return loss

# 模拟训练
for epoch in range(3):
    # 模拟批次数据
    indices = np.random.choice(len(X_data), size=16, replace=False)
    x_batch = tf.constant(X_data[indices])
    y_batch = tf.constant(y_data[indices])
    
    loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}, 损失: {loss.numpy():.4f}")

# ------------------------------------------------------------
# 3.3 PyTorch实现
print("\n" + "-"*40)
print("PyTorch实现 (手动训练循环)")
print("-"*40)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # 转换数据为PyTorch Tensor
    X_pt = torch.from_numpy(X_data)
    y_pt = torch.from_numpy(y_data)
    
    # 定义模型
    class LinearRegressionPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model_pt_train = LinearRegressionPT()
    criterion = nn.MSELoss()
    optimizer_pt = optim.SGD(model_pt_train.parameters(), lr=0.01)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(3):
        # 模拟批次数据
        indices = torch.randperm(len(X_pt))[:16]
        x_batch = X_pt[indices]
        y_batch = y_pt[indices]
        
        # 前向传播
        outputs = model_pt_train(x_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer_pt.zero_grad()
        loss.backward()
        optimizer_pt.step()
        
        print(f"Epoch {epoch+1}, 损失: {loss.item():.4f}")
        
except ImportError:
    print("PyTorch未安装，跳过PyTorch代码执行")

# ============================================================================
# 4. 数据加载对比
# ============================================================================
print("\n" + "="*60)
print("4. 数据加载对比")
print("="*60)

print("\n【任务】创建数据管道，支持打乱、分批、预取")

# ------------------------------------------------------------
# 4.1 TensorFlow实现 (tf.data)
print("\n" + "-"*40)
print("TensorFlow实现 (tf.data.Dataset)")
print("-"*40)

# 创建数据集
dataset_tf = tf.data.Dataset.from_tensor_slices((X_data, y_data))
dataset_tf = dataset_tf.shuffle(buffer_size=100)
dataset_tf = dataset_tf.batch(16)
dataset_tf = dataset_tf.prefetch(tf.data.AUTOTUNE)

print("TensorFlow数据集创建完成")
print(f"批次大小: 16")
print(f"总批次数: {len(list(dataset_tf))}")

# 查看一个批次
for batch_x, batch_y in dataset_tf.take(1):
    print(f"批次形状: X={batch_x.shape}, y={batch_y.shape}")

# ------------------------------------------------------------
# 4.2 PyTorch实现 (DataLoader)
print("\n" + "-"*40)
print("PyTorch实现 (DataLoader)")
print("-"*40)

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建TensorDataset
    dataset_pt = TensorDataset(
        torch.from_numpy(X_data),
        torch.from_numpy(y_data)
    )
    
    # 创建DataLoader
    dataloader_pt = DataLoader(
        dataset_pt,
        batch_size=16,
        shuffle=True,
        num_workers=0  # 多进程需要特殊处理
    )
    
    print("PyTorch DataLoader创建完成")
    print(f"批次大小: 16")
    print(f"总批次数: {len(dataloader_pt)}")
    
    # 查看一个批次
    for batch_x, batch_y in dataloader_pt:
        print(f"批次形状: X={batch_x.shape}, y={batch_y.shape}")
        break
        
except ImportError:
    print("PyTorch未安装，跳过PyTorch代码执行")

# ============================================================================
# 5. 模型保存与加载对比
# ============================================================================
print("\n" + "="*60)
print("5. 模型保存与加载对比")
print("="*60)

# ------------------------------------------------------------
# 5.1 TensorFlow实现
print("\n" + "-"*40)
print("TensorFlow实现 (SavedModel格式)")
print("-"*40)

# 保存模型
save_path_tf = 'outputs/阶段一/Week3/Day18/tf_model'
model_tf.save(save_path_tf)
print(f"TensorFlow模型已保存: {save_path_tf}")

# 加载模型
loaded_model_tf = tf.keras.models.load_model(save_path_tf)
print(f"TensorFlow模型已加载，结构相同")

# ------------------------------------------------------------
# 5.2 PyTorch实现
print("\n" + "-"*40)
print("PyTorch实现 (.pt格式)")
print("-"*40)

try:
    import torch
    
    # 保存模型
    save_path_pt = 'outputs/阶段一/Week3/Day18/pt_model.pt'
    torch.save(model_pt_train.state_dict(), save_path_pt)
    print(f"PyTorch模型已保存: {save_path_pt}")
    
    # 加载模型
    loaded_model_pt = LinearRegressionPT()
    loaded_model_pt.load_state_dict(torch.load(save_path_pt))
    loaded_model_pt.eval()
    print(f"PyTorch模型已加载，结构相同")
    
except ImportError:
    print("PyTorch未安装，跳过PyTorch代码执行")

# ============================================================================
# 6. 关键差异总结
# ============================================================================
print("\n" + "="*60)
print("6. TensorFlow vs PyTorch 关键差异总结")
print("="*60)

print("""
| 维度 | TensorFlow | PyTorch |
|------|------------|---------|
| **设计哲学** | "约定优于配置"，高度封装 | "灵活性优先"，暴露底层 |
| **执行模式** | 默认即时执行，可转静态图 | 默认动态图，可编译优化 |
| **API风格** | 声明式（Define-and-Run） | 命令式（Define-by-Run） |
| **调试体验** | 需切换eager/图模式 | 原生Python调试工具 |
| **模型定义** | Sequential/Functional API | nn.Module子类化 |
| **训练循环** | 高级API（model.fit）简化 | 手动编写，控制精细 |
| **自动求导** | GradientTape上下文管理器 | autograd自动记录 |
| **部署生态** | TensorFlow Serving、TFLite成熟 | TorchServe、ExecuTorch发展 |
| **研究生态** | 企业应用为主，生产部署强 | 学术研究主导，论文复现易 |
| **移动端** | TFLite统治边缘设备（27亿+） | PyTorch Mobile追赶 |

**核心差异详解**:

1. **计算图构建**:
   - TensorFlow: 默认即时执行，但通过`tf.function`可转换为静态图进行优化
   - PyTorch: 动态图构建，前向传播时实时构建计算图

2. **梯度计算**:
   - TensorFlow: 使用`GradientTape`显式记录运算，通过`tape.gradient()`计算
   - PyTorch: 自动调用`backward()`，梯度累积在张量的`.grad`属性

3. **设备管理**:
   - TensorFlow: 自动使用可用GPU，设备放置策略透明
   - PyTorch: 需要显式调用`.to(device)`移动张量和模型

4. **序列化**:
   - TensorFlow: SavedModel格式（包含计算图、权重、签名）
   - PyTorch: `.pt`或`.pth`文件（通常只保存权重或完整模型）

5. **分布式训练**:
   - TensorFlow: `tf.distribute.Strategy`高级抽象，TPU原生支持
   - PyTorch: `DistributedDataParallel`精细控制，FSDP成熟

**选择建议**:

✅ **选择TensorFlow当**:
   - 需要生产部署（移动端、边缘设备、云服务）
   - 企业级MLOps流水线（TFX）
   - 大规模分布式训练（TPU集群）
   - 标准化任务快速原型（Keras API）

✅ **选择PyTorch当**:
   - 进行前沿学术研究（85%新论文使用）
   - 需要高度自定义模型结构
   - 使用Hugging Face生态（NLP、大模型）
   - 调试友好性优先级高

🎯 **最佳实践**: "研究用PyTorch，生产用TensorFlow"
   - 研究阶段：PyTorch快速验证想法，复现论文
   - 生产阶段：转换为TensorFlow格式，利用成熟部署生态
""")

print("\n" + "="*60)
print("TensorFlow vs PyTorch 对比示例完成")
print("="*60)