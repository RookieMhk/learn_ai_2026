#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch实践练习 - Week 3 Day 3
包含：
1. 使用PyTorch重构Week 1的线性回归
2. 使用PyTorch实现MLP分类器
3. GPU加速对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================================
# 第一部分：使用PyTorch重构Week 1的线性回归
# ============================================================================

print("=" * 60)
print("第一部分：PyTorch线性回归")
print("=" * 60)


class PyTorchLinearRegression:
    """
    PyTorch版本的线性回归，使用自动微分替代手动梯度计算
    """
    
    def __init__(self, lr=0.01, n_epochs=1000):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 将NumPy转为PyTorch张量
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        # 初始化参数 (requires_grad=True 开启梯度追踪)
        self.weights = torch.randn(n_features, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        
        # 优化器（自动更新参数）
        optimizer = optim.SGD([self.weights, self.bias], lr=self.lr)
        
        # 训练
        for epoch in range(self.n_epochs):
            # 前向传播
            predictions = X_tensor @ self.weights + self.bias
            loss = ((predictions - y_tensor) ** 2).mean()
            
            # 反向传播（自动计算梯度！）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1:4d}/{self.n_epochs}, Loss: {loss.item():.6f}")
        
        # 转为numpy
        self.w_final = self.weights.detach().numpy()
        self.b_final = self.bias.detach().numpy()
        
        return self
    
    def predict(self, X):
        """预测"""
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            return (X_tensor @ self.weights + self.bias).numpy()


def demo_linear_regression():
    """线性回归演示"""
    print("\n--- 生成数据 ---")
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100个样本，3个特征
    true_w = np.array([2.0, -1.5, 3.0])
    true_b = 5.0
    y = X @ true_w + true_b + np.random.randn(100) * 0.1
    
    print(f"真实权重: {true_w}, 真实偏置: {true_b}")
    
    print("\n--- 训练模型 ---")
    model = PyTorchLinearRegression(lr=0.1, n_epochs=1000)
    model.fit(X, y)
    
    print(f"\n学习权重: {model.w_final.flatten()}")
    print(f"学习偏置: {model.b_final[0]:.4f}")
    
    # 评估
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    print(f"\n测试MSE: {mse:.6f}")


# ============================================================================
# 第二部分：使用PyTorch实现MLP分类器
# ============================================================================

print("\n" + "=" * 60)
print("第二部分：MLP多分类器")
print("=" * 60)


class MLPClassifier(nn.Module):
    """
    多层感知机分类器
    """
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_mlp_classifier():
    """训练MLP分类器"""
    print("\n--- 生成模拟数据 ---")
    # 生成3类数据
    np.random.seed(42)
    n_samples = 300
    
    # 类别1
    class1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    # 类别2
    class2 = np.random.randn(n_samples, 2) + np.array([-2, 2])
    # 类别3
    class3 = np.random.randn(n_samples, 2) + np.array([0, -2])
    
    # 合并数据
    X = np.vstack([class1, class2, class3])
    y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
    
    # 打乱顺序
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # 转为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # 数据划分
    n_train = int(0.7 * len(X))
    X_train, X_test = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]
    
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    print("\n--- 创建模型 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = MLPClassifier(
        input_size=2,
        hidden_sizes=[32, 16],
        num_classes=3,
        dropout=0.2
    ).to(device)
    
    print(model)
    
    print("\n--- 训练配置 ---")
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 训练
    num_epochs = 200
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"训练轮数: {num_epochs}, 批量大小: {batch_size}")
    print("\n--- 开始训练 ---")
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        
        # 计算准确率
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(device)
            outputs = model(X_test_dev)
            _, predicted = outputs.max(1)
            test_acc = predicted.eq(y_test.to(device)).float().mean().item()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    print("\n--- 训练完成 ---")
    print(f"最终训练准确率: {train_accs[-1]:.4f}")
    print(f"最终测试准确率: {test_accs[-1]:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'mlp_classifier.pth')
    print("模型已保存至 mlp_classifier.pth")
    
    return model, train_losses, train_accs, test_accs


# ============================================================================
# 第三部分：GPU加速对比实验
# ============================================================================

print("\n" + "=" * 60)
print("第三部分：GPU加速对比实验")
print("=" * 60)


def benchmark_matrix_multiplication():
    """矩阵乘法性能对比"""
    print("\n--- 矩阵乘法性能测试 ---")
    
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda') if torch.cuda.is_available() else None
    
    sizes = [500, 1000, 2000]
    
    results = {"CPU": [], "GPU": [], "加速比": []}
    
    for size in sizes:
        print(f"\n矩阵大小: {size}x{size}")
        
        # 创建随机矩阵
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # CPU测试
        start = time.time()
        for _ in range(10):
            c_cpu = a @ b
        cpu_time = (time.time() - start) / 10
        results["CPU"].append(cpu_time)
        print(f"  CPU时间: {cpu_time*1000:.2f} ms")
        
        # GPU测试
        if device_gpu:
            a_gpu = a.to(device_gpu)
            b_gpu = b.to(device_gpu)
            
            # 预热
            _ = a_gpu @ b_gpu
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(10):
                c_gpu = a_gpu @ b_gpu
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 10
            
            results["GPU"].append(gpu_time)
            results["加速比"].append(cpu_time / gpu_time)
            
            print(f"  GPU时间: {gpu_time*1000:.2f} ms")
            print(f"  加速比:  {cpu_time/gpu_time:.2f}x")
        else:
            results["GPU"].append(None)
            results["加速比"].append(None)
            print("  GPU不可用，跳过")
    
    return results


def benchmark_training():
    """训练速度对比"""
    print("\n--- 神经网络训练速度对比 ---")
    
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda') if torch.cuda.is_available() else None
    
    # 生成大数据集
    n_samples = 10000
    n_features = 100
    n_classes = 10
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 定义模型
    def create_model():
        return nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    criterion = nn.CrossEntropyLoss()
    
    def train_epoch(model, loader, device):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            model.zero_grad()
            loss.backward()
            
            # 简单的梯度下降（跳过optimizer.step以公平比较）
            with torch.no_grad():
                for param in model.parameters():
                    param -= 0.001 * param.grad
        
        return total_loss
    
    # CPU训练
    print(f"\nn_samples={n_samples}, n_features={n_features}")
    print(f"训练轮数: 5 epochs\n")
    
    print("CPU训练中...")
    model_cpu = create_model().to(device_cpu)
    start = time.time()
    for epoch in range(5):
        train_epoch(model_cpu, dataloader, device_cpu)
    cpu_time = time.time() - start
    print(f"CPU总时间: {cpu_time:.2f} s")
    
    # GPU训练
    if device_gpu:
        print("\nGPU训练中...")
        model_gpu = create_model().to(device_gpu)
        
        # 预热
        train_epoch(model_gpu, dataloader, device_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for epoch in range(5):
            train_epoch(model_gpu, dataloader, device_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"GPU总时间: {gpu_time:.2f} s")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        return cpu_time, gpu_time
    else:
        print("GPU不可用")
        return cpu_time, None


def benchmark_mixed_precision():
    """混合精度训练对比"""
    print("\n--- 混合精度训练对比 ---")
    
    if not torch.cuda.is_available():
        print("GPU不可用，跳过混合精度测试")
        return
    
    device = torch.device('cuda')
    
    # 模型和优化器
    model = MLPClassifier(100, [256, 128], 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 生成数据
    X = torch.randn(1000, 100).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    # FP32训练
    print("FP32训练中...")
    model.train()
    start = time.time()
    for i in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    print(f"FP32时间: {fp32_time:.2f} s")
    
    # FP16训练 (AMP)
    print("FP16 (AMP) 训练中...")
    scaler = torch.cuda.amp.GradScaler()
    model.zero_grad()
    
    start = time.time()
    for i in range(100):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = nn.CrossEntropyLoss()(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    fp16_time = time.time() - start
    print(f"FP16时间: {fp16_time:.2f} s")
    print(f"混合精度加速比: {fp32_time/fp16_time:.2f}x")
    
    # 显存对比
    torch.cuda.empty_cache()
    
    model_fp32 = MLPClassifier(100, [256, 128], 10).to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters())
    
    x_dummy = torch.randn(500, 100).to(device)
    y_dummy = torch.randint(0, 10, (500,)).to(device)
    
    for _ in range(10):
        output = model_fp32(x_dummy)
        loss = nn.CrossEntropyLoss()(output, y_dummy)
        loss.backward()
    
    mem_fp32 = torch.cuda.memory_allocated() / 1024**2
    
    torch.cuda.empty_cache()
    del model_fp32, optimizer_fp32
    
    model_fp16 = MLPClassifier(100, [256, 128], 10).to(device)
    optimizer_fp16 = optim.Adam(model_fp16.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    for _ in range(10):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model_fp16(x_dummy)
            loss = nn.CrossEntropyLoss()(output, y_dummy)
        scaler.scale(loss).backward()
        scaler.step(optimizer_fp16)
        scaler.update()
    
    mem_fp16 = torch.cuda.memory_allocated() / 1024**2
    
    print(f"\n显存占用对比:")
    print(f"FP32: {mem_fp32:.2f} MB")
    print(f"FP16: {mem_fp16:.2f} MB")
    print(f"显存节省: {(mem_fp32-mem_fp16)/mem_fp32*100:.1f}%")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("PyTorch 实践练习 - Week 3 Day 3")
    print("=" * 60)
    
    # 第一部分：线性回归
    demo_linear_regression()
    
    # 第二部分：MLP分类器
    train_mlp_classifier()
    
    # 第三部分：GPU加速实验
    benchmark_matrix_multiplication()
    benchmark_training()
    benchmark_mixed_precision()
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
