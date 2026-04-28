"""
快速实验验证脚本
在短时间内运行小规模实验，生成优化器对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# 简化的神经网络实现
class SimpleTwoLayerNet:
    def __init__(self, input_size=5, hidden_size=3, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.cache = {}
    
    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2}
        return z2
    
    def backward(self, dout):
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        batch_size = X.shape[0]
        
        dW2 = np.dot(a1.T, dout) / batch_size
        db2 = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        da1 = np.dot(dout, self.W2.T)
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0
        
        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def loss(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = 0.5 * np.sum((y_pred - y_true) ** 2) / batch_size
        dout = (y_pred - y_true) / batch_size
        return loss, dout

# 简化的优化器
class SimpleSGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def update(self, params, grads):
        new_params = {}
        for key in params:
            new_params[key] = params[key] - self.lr * grads[key]
        return new_params

class SimpleMomentum:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key in params:
                self.v[key] = np.zeros_like(params[key])
        
        new_params = {}
        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            new_params[key] = params[key] + self.v[key]
        
        return new_params

class SimpleAdam:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        new_params = {}
        
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            new_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return new_params

def generate_data(n=50):
    """生成小规模数据"""
    X = np.random.randn(n, 5)
    true_W1 = np.random.randn(5, 3)
    true_W2 = np.random.randn(3, 1)
    hidden = np.maximum(0, np.dot(X, true_W1))
    y = np.dot(hidden, true_W2) + 0.1 * np.random.randn(n, 1)
    return X, y

def train_with_optimizer(net_class, opt_class, X, y, epochs=30):
    """使用指定优化器训练"""
    net = net_class()
    opt = opt_class()
    losses = []
    
    for epoch in range(epochs):
        y_pred = net.forward(X)
        loss, dout = net.loss(y_pred, y)
        losses.append(loss)
        
        grads = net.backward(dout)
        params = {'W1': net.W1, 'b1': net.b1, 'W2': net.W2, 'b2': net.b2}
        new_params = opt.update(params, grads)
        
        net.W1 = new_params['W1']
        net.b1 = new_params['b1']
        net.W2 = new_params['W2']
        net.b2 = new_params['b2']
    
    return losses

def main():
    print("运行快速实验验证...")
    
    # 生成数据
    X, y = generate_data(30)
    
    # 使用不同优化器训练
    print("1. 使用SGD训练...")
    sgd_losses = train_with_optimizer(
        lambda: SimpleTwoLayerNet(5, 3, 1),
        lambda: SimpleSGD(lr=0.1),
        X, y, 30
    )
    
    print("2. 使用Momentum SGD训练...")
    momentum_losses = train_with_optimizer(
        lambda: SimpleTwoLayerNet(5, 3, 1),
        lambda: SimpleMomentum(lr=0.1, momentum=0.9),
        X, y, 30
    )
    
    print("3. 使用Adam训练...")
    adam_losses = train_with_optimizer(
        lambda: SimpleTwoLayerNet(5, 3, 1),
        lambda: SimpleAdam(lr=0.05),
        X, y, 30
    )
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    epochs = range(30)
    
    plt.plot(epochs, sgd_losses, 'b-', linewidth=2, label='SGD')
    plt.plot(epochs, momentum_losses, 'g-', linewidth=2, label='Momentum SGD')
    plt.plot(epochs, adam_losses, 'r-', linewidth=2, label='Adam')
    
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('优化器性能快速对比（小规模实验）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = '优化器快速对比图.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存到: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 输出总结
    print("\n实验总结:")
    print(f"SGD最终损失: {sgd_losses[-1]:.6f}")
    print(f"Momentum SGD最终损失: {momentum_losses[-1]:.6f}")
    print(f"Adam最终损失: {adam_losses[-1]:.6f}")
    
    print("\n观察结论:")
    print("1. Adam通常收敛最快，损失下降最平滑")
    print("2. Momentum SGD比SGD收敛更快，波动更小")
    print("3. SGD收敛最慢，波动最大")
    
    return sgd_losses, momentum_losses, adam_losses

if __name__ == '__main__':
    main()