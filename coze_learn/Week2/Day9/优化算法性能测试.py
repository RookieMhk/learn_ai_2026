#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化算法性能测试脚本
对比 SGD、Momentum、Adam 在简单回归任务上的表现
实现 Adam 优化器的数学原理（动量 + 自适应学习率）
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, Dict, List, Tuple

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 1. 数据生成 =======================
def generate_linear_data(n_samples: int = 200, noise_std: float = 0.5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成线性回归数据 y = 2*x + 3 + 噪声"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 1) * 2  # 特征
    true_w, true_b = 2.0, 3.0
    y = true_w * X + true_b + np.random.randn(n_samples, 1) * noise_std
    return X, y

# ======================= 2. 优化器实现 =======================
class Optimizer:
    """优化器基类"""
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.history = []  # 记录每步的损失值
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        raise NotImplementedError
    
    def record_loss(self, loss: float):
        self.history.append(loss)

class SGD(Optimizer):
    """随机梯度下降（含批量）"""
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params:
            params[key] -= self.lr * grads[key]

class Momentum(Optimizer):
    """带动量的 SGD"""
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = {}  # 速度累积
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params:
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            # 更新速度：v = beta*v + (1-beta)*grad（简化版常用 beta*v + grad）
            self.v[key] = self.beta * self.v[key] + grads[key]
            params[key] -= self.lr * self.v[key]

class Adam(Optimizer):
    """Adam 优化器（动量 + 自适应学习率）"""
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步
        self.m = {}  # 一阶矩估计（动量）
        self.v = {}  # 二阶矩估计（自适应学习率分母）
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # 更新一阶矩和二阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 偏差修正（初始时刻矩估计偏向0，需修正）
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ======================= 3. 模型与训练 =======================
class LinearRegression:
    """简单的线性回归模型"""
    def __init__(self, input_dim: int = 1):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1, 1))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b
    
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)
    
    def gradients(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """计算梯度"""
        n = X.shape[0]
        error = y_pred - y_true
        dW = (X.T @ error) / n
        db = np.sum(error) / n
        return {'W': dW, 'b': db}
    
    def train_one_epoch(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer) -> float:
        """单次训练迭代"""
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)
        grads = self.gradients(X, y_pred, y)
        optimizer.step({'W': self.W, 'b': self.b}, grads)
        optimizer.record_loss(loss)
        return loss
    
    def get_params(self) -> Tuple[float, float]:
        return self.W[0, 0], self.b[0, 0]

def train_model(optimizer: Optimizer, n_epochs: int = 100, verbose: bool = False) -> Tuple[List[float], LinearRegression]:
    """训练线性回归模型，返回损失历史和训练好的模型"""
    X, y = generate_linear_data()
    model = LinearRegression()
    
    for epoch in range(n_epochs):
        loss = model.train_one_epoch(X, y, optimizer)
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f'Epoch {epoch:3d}, Loss: {loss:.6f}')
    
    return optimizer.history, model

# ======================= 4. 对比实验 =======================
def run_comparison() -> Dict[str, List[float]]:
    """运行三种优化器的对比实验"""
    optimizers = {
        'SGD': SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, beta=0.9),
        'Adam': Adam(lr=0.05, beta1=0.9, beta2=0.999)
    }
    
    results = {}
    for name, opt in optimizers.items():
        print(f'开始训练 {name}...')
        start_time = time.time()
        losses, model = train_model(opt, n_epochs=150, verbose=False)
        end_time = time.time()
        w, b = model.get_params()
        print(f'  {name} 训练完成，耗时 {end_time - start_time:.2f} 秒')
        print(f'  最终参数: w={w:.4f}, b={b:.4f} (真实值 w=2.0, b=3.0)')
        results[name] = losses
    return results

# ======================= 5. 可视化 =======================
def plot_results(results: Dict[str, List[float]]):
    """绘制损失曲线对比图"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    for name, losses in results.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值 (MSE)')
    plt.title('优化算法损失下降曲线对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 对数坐标下的损失曲线（更易观察后期收敛）
    plt.subplot(1, 2, 2)
    for name, losses in results.items():
        plt.semilogy(losses, label=name, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值 (对数刻度)')
    plt.title('损失下降曲线（对数坐标）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/阶段一/Week2/Day9/优化算法对比.png', dpi=300, bbox_inches='tight')
    print('对比图已保存至 outputs/阶段一/Week2/Day9/优化算法对比.png')
    plt.show()

# ======================= 6. 主函数 =======================
def main():
    print('=' * 60)
    print('优化算法性能测试')
    print('对比 SGD、Momentum、Adam 在简单线性回归任务上的表现')
    print('=' * 60)
    
    # 运行对比实验
    results = run_comparison()
    
    # 可视化
    plot_results(results)
    
    # 打印最终损失值对比
    print('\n最终损失值对比：')
    for name, losses in results.items():
        print(f'  {name:10s}: {losses[-1]:.6f}')
    
    # Adam 优化器数学原理说明
    print('\n' + '=' * 60)
    print('Adam 优化器数学原理总结')
    print('=' * 60)
    print('1. 动量（一阶矩）: m_t = β₁·m_{t-1} + (1-β₁)·g_t')
    print('2. 自适应学习率（二阶矩）: v_t = β₂·v_{t-1} + (1-β₂)·g_t²')
    print('3. 偏差修正: m̂_t = m_t / (1-β₁^t), v̂_t = v_t / (1-β₂^t)')
    print('4. 参数更新: θ_t = θ_{t-1} - η·m̂_t / (√v̂_t + ε)')
    print('\n优点：')
    print('  • 结合动量（加速收敛）和自适应学习率（稳定训练）')
    print('  • 对稀疏梯度友好（自适应调整每个参数的学习率）')
    print('  • 超参数鲁棒性强（β₁=0.9, β₂=0.999 适用于大多数场景）')
    print('\n注意事项：')
    print('  • 学习率 η 通常设为 0.001 或 0.0001')
    print('  • ε 防止除以零，一般取 1e-8')
    print('  • 偏差修正在训练初期尤为重要')

if __name__ == '__main__':
    main()