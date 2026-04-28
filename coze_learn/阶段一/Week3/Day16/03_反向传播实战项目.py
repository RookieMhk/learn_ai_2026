"""
反向传播与梯度下降优化实战项目
实现2层全连接神经网络，手动实现前向传播、反向传播算法
对比SGD、Momentum、Adam三种优化器的性能
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子确保可重复性
np.random.seed(42)

class TwoLayerNet:
    """2层全连接神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化网络参数
        
        参数:
        input_size: 输入层维度
        hidden_size: 隐藏层维度
        output_size: 输出层维度
        """
        # He初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # 缓存前向传播的中间结果，用于反向传播
        self.cache = {}
        
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据，形状 (batch_size, input_size)
        
        返回:
        y_pred: 预测值，形状 (batch_size, output_size)
        """
        # 第一层: 线性变换 + ReLU激活
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU激活
        
        # 第二层: 线性变换 (无激活，用于回归或分类的logits)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # 缓存中间结果用于反向传播
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        
        return z2
    
    def backward(self, dout):
        """
        反向传播计算梯度
        
        参数:
        dout: 损失函数对输出的梯度，形状 (batch_size, output_size)
        
        返回:
        梯度字典，包含各参数的梯度
        """
        # 从缓存中获取前向传播的中间结果
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        batch_size = X.shape[0]
        
        # 第二层的梯度
        dW2 = np.dot(a1.T, dout) / batch_size
        db2 = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        # 第一层的梯度 (通过链式法则)
        da1 = np.dot(dout, self.W2.T)
        # ReLU激活的梯度
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0  # ReLU导数: z<=0时为0，z>0时为1
        
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        gradients = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
        
        return gradients
    
    def loss(self, y_pred, y_true):
        """
        计算均方误差损失
        
        参数:
        y_pred: 预测值，形状 (batch_size, output_size)
        y_true: 真实值，形状 (batch_size, output_size)
        
        返回:
        loss: 标量损失值
        dout: 损失对输出的梯度
        """
        batch_size = y_pred.shape[0]
        # 均方误差损失
        loss = 0.5 * np.sum((y_pred - y_true) ** 2) / batch_size
        # 损失对输出的梯度
        dout = (y_pred - y_true) / batch_size
        return loss, dout

class Optimizer:
    """优化器基类"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.t = 0  # 时间步，用于Adam等优化器
        
    def update(self, params, gradients):
        """更新参数，子类需实现"""
        raise NotImplementedError

class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def update(self, params, gradients):
        """
        标准SGD更新: param = param - learning_rate * gradient
        """
        updated_params = {}
        for key in params:
            updated_params[key] = params[key] - self.learning_rate * gradients[key]
        return updated_params

class MomentumSGD(Optimizer):
    """带动量的SGD优化器"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, gradients):
        """
        带动量的SGD更新:
        v = momentum * v - learning_rate * gradient
        param = param + v
        """
        if self.velocity is None:
            self.velocity = {}
            for key in params:
                self.velocity[key] = np.zeros_like(params[key])
        
        updated_params = {}
        for key in params:
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * gradients[key]
            updated_params[key] = params[key] + self.velocity[key]
        
        return updated_params

class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计
        
    def update(self, params, gradients):
        """
        Adam更新算法:
        t = t + 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        """
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        updated_params = {}
        
        for key in params:
            # 更新一阶矩和二阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params

def generate_synthetic_data(num_samples=1000, input_size=10, output_size=1):
    """
    生成合成数据用于训练
    
    参数:
    num_samples: 样本数量
    input_size: 输入维度
    output_size: 输出维度
    
    返回:
    X: 输入数据，形状 (num_samples, input_size)
    y: 目标数据，形状 (num_samples, output_size)
    """
    # 生成随机权重矩阵用于生成数据
    true_W1 = np.random.randn(input_size, 5)
    true_W2 = np.random.randn(5, output_size)
    
    # 生成输入数据
    X = np.random.randn(num_samples, input_size)
    
    # 生成目标数据: 通过一个简单的两层网络
    hidden = np.maximum(0, np.dot(X, true_W1))
    y = np.dot(hidden, true_W2) + 0.1 * np.random.randn(num_samples, output_size)
    
    return X, y

def train_model(net, optimizer, X_train, y_train, epochs=100):
    """
    训练模型
    
    参数:
    net: 神经网络实例
    optimizer: 优化器实例
    X_train: 训练数据
    y_train: 训练标签
    epochs: 训练轮数
    
    返回:
    losses: 每轮的损失值列表
    """
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        y_pred = net.forward(X_train)
        
        # 计算损失和梯度
        loss, dout = net.loss(y_pred, y_train)
        losses.append(loss)
        
        # 反向传播
        gradients = net.backward(dout)
        
        # 获取当前参数
        params = {
            'W1': net.W1, 'b1': net.b1,
            'W2': net.W2, 'b2': net.b2
        }
        
        # 优化器更新参数
        updated_params = optimizer.update(params, gradients)
        
        # 更新网络参数
        net.W1 = updated_params['W1']
        net.b1 = updated_params['b1']
        net.W2 = updated_params['W2']
        net.b2 = updated_params['b2']
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    return losses

def plot_optimizer_comparison(sgd_losses, momentum_losses, adam_losses, save_path=None):
    """
    绘制优化器对比图
    
    参数:
    sgd_losses: SGD优化器的损失曲线
    momentum_losses: Momentum SGD优化器的损失曲线
    adam_losses: Adam优化器的损失曲线
    save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 6))
    
    epochs = range(len(sgd_losses))
    
    plt.plot(epochs, sgd_losses, 'b-', linewidth=2, label='SGD')
    plt.plot(epochs, momentum_losses, 'g-', linewidth=2, label='Momentum SGD')
    plt.plot(epochs, adam_losses, 'r-', linewidth=2, label='Adam')
    
    plt.xlabel('训练轮数', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.title('不同优化器的损失曲线对比', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴为对数刻度，更好地显示变化
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数：运行完整的优化器对比实验"""
    print("=" * 60)
    print("反向传播与梯度下降优化实战项目")
    print("=" * 60)
    
    # 1. 生成合成数据
    print("\n1. 生成合成数据...")
    X, y = generate_synthetic_data(num_samples=500, input_size=20, output_size=1)
    
    # 分割训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")
    
    # 2. 使用SGD优化器训练
    print("\n2. 使用SGD优化器训练...")
    net_sgd = TwoLayerNet(input_size=20, hidden_size=10, output_size=1)
    optimizer_sgd = SGD(learning_rate=0.01)
    sgd_losses = train_model(net_sgd, optimizer_sgd, X_train, y_train, epochs=200)
    
    # 3. 使用Momentum SGD优化器训练
    print("\n3. 使用Momentum SGD优化器训练...")
    net_momentum = TwoLayerNet(input_size=20, hidden_size=10, output_size=1)
    optimizer_momentum = MomentumSGD(learning_rate=0.01, momentum=0.9)
    momentum_losses = train_model(net_momentum, optimizer_momentum, X_train, y_train, epochs=200)
    
    # 4. 使用Adam优化器训练
    print("\n4. 使用Adam优化器训练...")
    net_adam = TwoLayerNet(input_size=20, hidden_size=10, output_size=1)
    optimizer_adam = Adam(learning_rate=0.001)
    adam_losses = train_model(net_adam, optimizer_adam, X_train, y_train, epochs=200)
    
    # 5. 绘制对比图
    print("\n5. 绘制优化器对比图...")
    plot_optimizer_comparison(
        sgd_losses, momentum_losses, adam_losses,
        save_path='optimizer_comparison.png' if __name__ == '__main__' else None
    )
    
    # 6. 在测试集上评估模型
    print("\n6. 在测试集上评估模型性能...")
    
    def evaluate_model(net, X_test, y_test):
        y_pred = net.forward(X_test)
        loss, _ = net.loss(y_pred, y_test)
        return loss
    
    sgd_test_loss = evaluate_model(net_sgd, X_test, y_test)
    momentum_test_loss = evaluate_model(net_momentum, X_test, y_test)
    adam_test_loss = evaluate_model(net_adam, X_test, y_test)
    
    print(f"SGD测试损失: {sgd_test_loss:.6f}")
    print(f"Momentum SGD测试损失: {momentum_test_loss:.6f}")
    print(f"Adam测试损失: {adam_test_loss:.6f}")
    
    # 7. 输出优化器对比总结
    print("\n" + "=" * 60)
    print("优化器性能对比总结")
    print("=" * 60)
    
    print("\n1. 收敛速度:")
    print("   - Adam: 最快，通常在50轮内达到较低损失")
    print("   - Momentum SGD: 中等，比SGD快但比Adam慢")
    print("   - SGD: 最慢，需要更多轮数才能收敛")
    
    print("\n2. 稳定性:")
    print("   - Adam: 最稳定，损失曲线平滑")
    print("   - Momentum SGD: 较稳定，但可能有轻微波动")
    print("   - SGD: 波动较大，对学习率敏感")
    
    print("\n3. 超参数敏感性:")
    print("   - SGD: 对学习率非常敏感")
    print("   - Momentum SGD: 对学习率和动量因子敏感")
    print("   - Adam: 相对鲁棒，但需要调整学习率")
    
    print("\n4. 推荐使用场景:")
    print("   - Adam: 大多数深度学习任务的默认选择")
    print("   - Momentum SGD: 当Adam表现不佳时的备选")
    print("   - SGD: 需要精细控制优化过程时使用")
    
    return {
        'sgd_losses': sgd_losses,
        'momentum_losses': momentum_losses,
        'adam_losses': adam_losses,
        'test_losses': {
            'sgd': sgd_test_loss,
            'momentum': momentum_test_loss,
            'adam': adam_test_loss
        }
    }

if __name__ == '__main__':
    # 如果直接运行此脚本，执行主函数
    results = main()
    
    # 保存损失数据到文件
    np.savez('optimizer_results.npz',
             sgd_losses=results['sgd_losses'],
             momentum_losses=results['momentum_losses'],
             adam_losses=results['adam_losses'])
    
    print("\n实战项目已完成！所有结果已保存。")