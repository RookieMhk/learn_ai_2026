#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性回归优化对比实验
实现批量梯度下降、随机梯度下降、小批量梯度下降的性能对比
包含学习率衰减策略的数学原理实现
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict, Callable

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

class LinearRegression:
    """线性回归模型类"""
    def __init__(self, n_features: int):
        """
        初始化线性回归模型
        
        参数:
            n_features: 特征数量（不包括偏置项）
        """
        self.n_features = n_features
        self.weights = np.random.randn(n_features + 1) * 0.01  # 包含偏置项
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测函数
        
        参数:
            X: 输入特征矩阵 (n_samples, n_features)
            
        返回:
            y_pred: 预测值 (n_samples,)
        """
        # 添加偏置项
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return X_bias @ self.weights
    
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算均方误差损失
        
        参数:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
            
        返回:
            mse: 均方误差
        """
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)
    
    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算梯度（批量）
        
        参数:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
            
        返回:
            grad: 梯度向量
        """
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]
        y_pred = X_bias @ self.weights
        error = y_pred - y
        grad = (2 / n_samples) * (X_bias.T @ error)
        return grad

class LearningRateScheduler:
    """学习率调度器基类"""
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        
    def step(self, epoch: int) -> float:
        """
        更新学习率
        
        参数:
            epoch: 当前迭代轮次
            
        返回:
            lr: 更新后的学习率
        """
        raise NotImplementedError
    
    def reset(self):
        """重置学习率"""
        self.current_lr = self.initial_lr

class ConstantLR(LearningRateScheduler):
    """恒定学习率"""
    def step(self, epoch: int) -> float:
        return self.current_lr

class TimeBasedDecayLR(LearningRateScheduler):
    """时间衰减学习率：lr = initial_lr / (1 + decay_rate * epoch)"""
    def __init__(self, initial_lr: float, decay_rate: float):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        
    def step(self, epoch: int) -> float:
        self.current_lr = self.initial_lr / (1 + self.decay_rate * epoch)
        return self.current_lr

class ExponentialDecayLR(LearningRateScheduler):
    """指数衰减学习率：lr = initial_lr * decay_rate^epoch"""
    def __init__(self, initial_lr: float, decay_rate: float):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        
    def step(self, epoch: int) -> float:
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.current_lr

class StepDecayLR(LearningRateScheduler):
    """阶梯衰减学习率：每step_size轮衰减gamma倍"""
    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        
    def step(self, epoch: int) -> float:
        self.current_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        return self.current_lr

def generate_data(n_samples: int = 1000, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟数据
    
    参数:
        n_samples: 样本数量
        noise_std: 噪声标准差
        
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    np.random.seed(42)
    n_features = 5
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 生成真实权重
    true_weights = np.array([2.0, -1.5, 0.8, 0.3, -0.6, 1.2])  # 包含偏置项
    
    # 生成标签（线性关系 + 噪声）
    X_bias = np.c_[np.ones(n_samples), X]
    y = X_bias @ true_weights + np.random.randn(n_samples) * noise_std
    
    return X, y

def batch_gradient_descent(model: LinearRegression, X: np.ndarray, y: np.ndarray,
                          lr_scheduler: LearningRateScheduler, n_epochs: int = 100) -> Dict:
    """
    批量梯度下降
    
    参数:
        model: 线性回归模型
        X: 特征矩阵
        y: 标签向量
        lr_scheduler: 学习率调度器
        n_epochs: 迭代轮数
        
    返回:
        history: 训练历史记录
    """
    n_samples = X.shape[0]
    losses = []
    lrs = []
    
    for epoch in range(n_epochs):
        # 获取当前学习率
        lr = lr_scheduler.step(epoch)
        lrs.append(lr)
        
        # 计算梯度并更新权重
        grad = model.gradient(X, y)
        model.weights -= lr * grad
        
        # 记录损失
        loss = model.loss(X, y)
        losses.append(loss)
        
    return {
        'losses': losses,
        'learning_rates': lrs,
        'final_weights': model.weights.copy()
    }

def stochastic_gradient_descent(model: LinearRegression, X: np.ndarray, y: np.ndarray,
                               lr_scheduler: LearningRateScheduler, n_epochs: int = 100) -> Dict:
    """
    随机梯度下降
    
    参数:
        model: 线性回归模型
        X: 特征矩阵
        y: 标签向量
        lr_scheduler: 学习率调度器
        n_epochs: 迭代轮数
        
    返回:
        history: 训练历史记录
    """
    n_samples = X.shape[0]
    losses = []
    lrs = []
    
    for epoch in range(n_epochs):
        # 获取当前学习率
        lr = lr_scheduler.step(epoch)
        lrs.append(lr)
        
        epoch_loss = 0
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 逐个样本更新
        for i in range(n_samples):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            
            # 计算梯度并更新权重
            grad = model.gradient(xi, yi)
            model.weights -= lr * grad
            
            # 计算当前样本损失
            epoch_loss += model.loss(xi, yi)
        
        # 记录平均损失
        losses.append(epoch_loss / n_samples)
        
    return {
        'losses': losses,
        'learning_rates': lrs,
        'final_weights': model.weights.copy()
    }

def mini_batch_gradient_descent(model: LinearRegression, X: np.ndarray, y: np.ndarray,
                               lr_scheduler: LearningRateScheduler, n_epochs: int = 100,
                               batch_size: int = 32) -> Dict:
    """
    小批量梯度下降
    
    参数:
        model: 线性回归模型
        X: 特征矩阵
        y: 标签向量
        lr_scheduler: 学习率调度器
        n_epochs: 迭代轮数
        batch_size: 批量大小
        
    返回:
        history: 训练历史记录
    """
    n_samples = X.shape[0]
    losses = []
    lrs = []
    
    for epoch in range(n_epochs):
        # 获取当前学习率
        lr = lr_scheduler.step(epoch)
        lrs.append(lr)
        
        epoch_loss = 0
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 按批次更新
        n_batches = n_samples // batch_size
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # 计算梯度并更新权重
            grad = model.gradient(X_batch, y_batch)
            model.weights -= lr * grad
            
            # 计算当前批次损失
            epoch_loss += model.loss(X_batch, y_batch)
        
        # 记录平均损失
        losses.append(epoch_loss / n_batches)
        
    return {
        'losses': losses,
        'learning_rates': lrs,
        'final_weights': model.weights.copy()
    }

def plot_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    绘制优化算法对比图
    
    参数:
        results: 各算法的结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线对比
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result['losses'], label=name, linewidth=2)
    ax.set_xlabel('迭代轮数', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title('损失收敛曲线对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_yscale('log')  # 对数尺度便于观察
    
    # 学习率变化
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result['learning_rates'], label=name, linewidth=2)
    ax.set_xlabel('迭代轮数', fontsize=12)
    ax.set_ylabel('学习率', fontsize=12)
    ax.set_title('学习率衰减策略对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # 最终损失对比
    ax = axes[1, 0]
    names = list(results.keys())
    final_losses = [results[name]['losses'][-1] for name in names]
    bars = ax.bar(names, final_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_xlabel('优化算法', fontsize=12)
    ax.set_ylabel('最终损失', fontsize=12)
    ax.set_title('最终损失对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{loss:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 收敛速度对比（达到特定损失阈值所需的轮数）
    ax = axes[1, 1]
    threshold = 0.05
    convergence_epochs = []
    for name in names:
        losses = results[name]['losses']
        for i, loss in enumerate(losses):
            if loss < threshold:
                convergence_epochs.append(i)
                break
        else:
            convergence_epochs.append(n_epochs)
    
    bars = ax.bar(names, convergence_epochs, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax.set_xlabel('优化算法', fontsize=12)
    ax.set_ylabel('收敛轮数', fontsize=12)
    ax.set_title(f'达到损失<{threshold}的收敛速度', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, epoch in zip(bars, convergence_epochs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{epoch}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    plt.show()

def print_algorithm_analysis(results: Dict[str, Dict]):
    """
    打印算法性能分析
    
    参数:
        results: 各算法的结果字典
    """
    print("=" * 80)
    print("优化算法性能分析")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  最终损失: {result['losses'][-1]:.8f}")
        print(f"  初始损失: {result['losses'][0]:.8f}")
        print(f"  损失降低倍数: {result['losses'][0]/result['losses'][-1]:.2f}")
        print(f"  最终学习率: {result['learning_rates'][-1]:.8f}")
        print(f"  权重范数: {np.linalg.norm(result['final_weights']):.4f}")

def main():
    """主函数"""
    print("线性回归优化算法对比实验")
    print("=" * 80)
    
    # 生成数据
    X, y = generate_data(n_samples=1000, noise_std=0.1)
    print(f"数据生成完成: {X.shape[0]}个样本, {X.shape[1]}个特征")
    
    # 划分训练集和测试集
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本")
    
    # 定义学习率调度器
    initial_lr = 0.1
    schedulers = {
        'BGD': ConstantLR(initial_lr),
        'SGD': TimeBasedDecayLR(initial_lr, decay_rate=0.01),
        'MiniBatch': ExponentialDecayLR(initial_lr, decay_rate=0.995)
    }
    
    # 运行不同优化算法
    results = {}
    n_epochs = 200
    
    # 批量梯度下降
    print("\n1. 批量梯度下降 (Batch Gradient Descent) 运行中...")
    model_bgd = LinearRegression(n_features=X.shape[1])
    start_time = time.time()
    results['BGD'] = batch_gradient_descent(model_bgd, X_train, y_train, 
                                           schedulers['BGD'], n_epochs=n_epochs)
    bgd_time = time.time() - start_time
    print(f"   运行时间: {bgd_time:.2f}秒")
    
    # 随机梯度下降
    print("\n2. 随机梯度下降 (Stochastic Gradient Descent) 运行中...")
    model_sgd = LinearRegression(n_features=X.shape[1])
    start_time = time.time()
    results['SGD'] = stochastic_gradient_descent(model_sgd, X_train, y_train,
                                                schedulers['SGD'], n_epochs=n_epochs)
    sgd_time = time.time() - start_time
    print(f"   运行时间: {sgd_time:.2f}秒")
    
    # 小批量梯度下降
    print("\n3. 小批量梯度下降 (Mini-Batch Gradient Descent) 运行中...")
    model_mb = LinearRegression(n_features=X.shape[1])
    start_time = time.time()
    results['MiniBatch'] = mini_batch_gradient_descent(model_mb, X_train, y_train,
                                                      schedulers['MiniBatch'], 
                                                      n_epochs=n_epochs, batch_size=32)
    mb_time = time.time() - start_time
    print(f"   运行时间: {mb_time:.2f}秒")
    
    # 打印性能分析
    print_algorithm_analysis(results)
    
    # 绘制对比图
    print("\n生成可视化对比图...")
    plot_comparison(results, save_path='outputs/阶段一/Week2/Day8/优化算法对比.png')
    
    # 测试集性能评估
    print("\n" + "=" * 80)
    print("测试集性能评估")
    print("=" * 80)
    
    models = {'BGD': model_bgd, 'SGD': model_sgd, 'MiniBatch': model_mb}
    for name, model in models.items():
        test_loss = model.loss(X_test, y_test)
        print(f"{name}: 测试集损失 = {test_loss:.8f}")
    
    # 学习率衰减策略对比
    print("\n" + "=" * 80)
    print("学习率衰减策略对比")
    print("=" * 80)
    
    lr_strategies = {
        '恒定': ConstantLR(0.1),
        '时间衰减': TimeBasedDecayLR(0.1, decay_rate=0.01),
        '指数衰减': ExponentialDecayLR(0.1, decay_rate=0.995),
        '阶梯衰减': StepDecayLR(0.1, step_size=50, gamma=0.5)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, scheduler in lr_strategies.items():
        lrs = [scheduler.step(epoch) for epoch in range(n_epochs)]
        ax.plot(lrs, label=name, linewidth=2)
        scheduler.reset()
    
    ax.set_xlabel('迭代轮数', fontsize=12)
    ax.set_ylabel('学习率', fontsize=12)
    ax.set_title('不同学习率衰减策略对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/阶段一/Week2/Day8/学习率衰减策略对比.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("学习率衰减策略对比图已保存")
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()