"""
反向传播实现练习
===================
本文件包含反向传播算法的完整实现，包括：
1. 计算图与自动微分
2. 常见层的前向/反向传播
3. 梯度检验的正确性验证
4. 完整MLP的训练循环

作者：AI Learning Agent
日期：2026-04-20
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# ============================================================================
# 第一部分：计算图基础实现
# ============================================================================

@dataclass
class Tensor:
    """张量类，存储值和梯度"""
    value: np.ndarray
    grad: np.ndarray = field(default_factory=lambda: None)
    
    def __post_init__(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
    
    def zero_grad(self):
        """清零梯度"""
        self.grad = np.zeros_like(self.value)


class Operation:
    """计算图操作基类"""
    
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> List[Tensor]:
        raise NotImplementedError
    
    def __call__(self, *inputs: Tensor) -> Tensor:
        output = self.forward(*inputs)
        output.ctx = self  # 保存计算上下文用于反向传播
        output.inputs = inputs
        return output


class Add(Operation):
    """加法操作：z = x + y"""
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.value + y.value)
    
    def backward(self, grad: Tensor) -> List[Tensor]:
        return [Tensor(grad.value), Tensor(grad.value)]


class Multiply(Operation):
    """乘法操作：z = x * y"""
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.value * y.value)
    
    def backward(self, grad: Tensor) -> List[Tensor]:
        x, y = self.inputs
        return [
            Tensor(grad.value * y.value),
            Tensor(grad.value * x.value)
        ]


class Sigmoid(Operation):
    """Sigmoid激活函数：σ(x) = 1/(1+exp(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        output = Tensor(1 / (1 + np.exp(-np.clip(x.value, -500, 500))))
        output.cache = x.value  # 缓存用于反向传播
        return output
    
    def backward(self, grad: Tensor) -> List[Tensor]:
        sigmoid = self.forward(Tensor(self.inputs[0].ctx.cache if hasattr(self.inputs[0], 'ctx') else self.inputs[0].value))
        # 重新计算sigmoid值用于梯度
        x_val = self.inputs[0].value
        sig_val = 1 / (1 + np.exp(-np.clip(x_val, -500, 500)))
        dx = sig_val * (1 - sig_val)
        return [Tensor(grad.value * dx)]


class MSE(Operation):
    """均方误差损失：L = 0.5 * (y_pred - y_true)^2"""
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred.value - y_true.value
        output = Tensor(0.5 * diff ** 2)
        output.cache = (y_pred.value, y_true.value, diff)
        return output
    
    def backward(self, grad: Tensor) -> List[Tensor]:
        y_pred, y_true, diff = self.cache
        return [
            Tensor(grad.value * diff),
            Tensor(-grad.value * diff)
        ]


# ============================================================================
# 第二部分：神经网络层实现
# ============================================================================

class Layer:
    """神经网络层基类"""
    
    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()


class Dense(Layer):
    """全连接层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 weight_scale: float = 0.01):
        super().__init__()
        # Xavier初始化
        scale = weight_scale * np.sqrt(2.0 / (input_dim + output_dim))
        self.params['W'] = Tensor(np.random.randn(input_dim, output_dim) * scale)
        self.params['b'] = Tensor(np.zeros((1, output_dim)))
        self.last_x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return np.dot(x, self.params['W'].value) + self.params['b'].value
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # 梯度计算
        self.params['W'].grad += np.dot(self.last_x.T, grad)
        self.params['b'].grad += np.sum(grad, axis=0, keepdims=True)
        # 返回传播到前一层的梯度
        return np.dot(grad, self.params['W'].value.T)


class Activation(Layer):
    """激活函数层"""
    
    def __init__(self, activation: str = 'relu'):
        super().__init__()
        self.activation_name = activation
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.activation_name == 'relu':
            return grad * (self.last_x > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(self.last_x, -500, 500)))
            return grad * sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return grad * (1 - np.tanh(self.last_x) ** 2)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")


# ============================================================================
# 第三部分：梯度检验
# ============================================================================

def compute_numerical_gradient(func: Callable, x: np.ndarray, 
                                 epsilon: float = 1e-5) -> np.ndarray:
    """
    数值梯度计算（用于验证反向传播正确性）
    
    使用中心差分法：f'(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + epsilon)
        x[idx] = old_value + epsilon
        loss_plus = func(x)
        
        # f(x - epsilon)
        x[idx] = old_value - epsilon
        loss_minus = func(x)
        
        # 中心差分
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # 恢复原值
        x[idx] = old_value
        it.iternext()
    
    return grad


def gradient_check(layer: Layer, x: np.ndarray, grad: np.ndarray,
                   epsilon: float = 1e-5) -> Tuple[float, float]:
    """
    梯度检验函数
    
    返回：
        analytical_grad: 解析梯度
        numerical_grad: 数值梯度
    """
    # 前向传播
    output = layer.forward(x)
    
    # 获取解析梯度
    layer.backward(grad)
    analytical_grad = layer.params['W'].grad.copy()
    
    # 计算数值梯度
    def loss_func(w):
        layer_copy = Dense(x.shape[1], grad.shape[1])
        layer_copy.params['W'].value = w
        out = layer_copy.forward(x)
        return np.sum(out * grad)
    
    numerical_grad = compute_numerical_gradient(loss_func, x.copy())
    
    return analytical_grad, numerical_grad


def compare_gradients(analytical: np.ndarray, numerical: np.ndarray,
                      tolerance: float = 1e-6) -> bool:
    """
    比较解析梯度和数值梯度
    
    相对误差公式：|a - n| / (|a| + |n| + ε)
    """
    numerator = np.abs(analytical - numerical)
    denominator = np.abs(analytical) + np.abs(numerical) + 1e-8
    relative_error = numerator / denominator
    
    print(f"最大相对误差: {np.max(relative_error):.2e}")
    print(f"平均相对误差: {np.mean(relative_error):.2e}")
    
    return np.allclose(analytical, numerical, rtol=tolerance, atol=tolerance)


# ============================================================================
# 第四部分：完整MLP实现
# ============================================================================

class MLP:
    """多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, activation: str = 'relu'):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(Dense(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 输出层不加激活
                self.layers.append(Activation(activation))
        
        self.loss_history = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SGD:
    """随机梯度下降优化器"""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Dict[str, np.ndarray] = {}
    
    def step(self, model: MLP):
        for layer in model.layers:
            if isinstance(layer, Dense):
                for name, param in layer.params.items():
                    key = id(param)
                    
                    if self.momentum > 0:
                        if key not in self.velocity:
                            self.velocity[key] = np.zeros_like(param.value)
                        
                        # v = m*v - lr*g
                        self.velocity[key] = (self.momentum * self.velocity[key] 
                                            - self.lr * param.grad)
                        # θ = θ + v
                        param.value += self.velocity[key]
                    else:
                        # θ = θ - lr*g
                        param.value -= self.lr * param.grad


class Adam:
    """Adam优化器"""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, np.ndarray] = {}  # 一阶矩估计
        self.v: Dict[str, np.ndarray] = {}  # 二阶矩估计
        self.t = 0  # 时间步
    
    def step(self, model: MLP):
        self.t += 1
        
        for layer in model.layers:
            if isinstance(layer, Dense):
                for name, param in layer.params.items():
                    key = id(param)
                    
                    if key not in self.m:
                        self.m[key] = np.zeros_like(param.value)
                        self.v[key] = np.zeros_like(param.value)
                    
                    # 更新一阶和二阶矩估计
                    g = param.grad
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
                    
                    # 偏差修正
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                    
                    # 更新参数
                    param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def train_step(model: MLP, x: np.ndarray, y: np.ndarray, 
               optimizer: SGD) -> float:
    """
    单步训练
    """
    # 清零梯度
    model.zero_grad()
    
    # 前向传播
    y_pred = model.forward(x)
    
    # 计算损失 (MSE)
    loss = 0.5 * np.mean((y_pred - y) ** 2)
    
    # 反向传播
    grad = y_pred - y
    model.backward(grad)
    
    # 更新参数
    optimizer.step(model)
    
    return loss


def train(model: MLP, X_train: np.ndarray, y_train: np.ndarray,
          optimizer: SGD, epochs: int = 100, 
          batch_size: int = 32) -> List[float]:
    """
    完整训练流程
    """
    losses = []
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            loss = train_step(model, x_batch, y_batch, optimizer)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


# ============================================================================
# 第五部分：可视化演示
# ============================================================================

def visualize_gradient_flow():
    """可视化不同激活函数的梯度流动"""
    x = np.linspace(-5, 5, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    activations = {
        'Sigmoid': (1 / (1 + np.exp(-x)), 
                    1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))),
        'Tanh': (np.tanh(x), 1 - np.tanh(x)**2),
        'ReLU': (np.maximum(0, x), (x > 0).astype(float))
    }
    
    for ax, (name, (val, grad)) in zip(axes, activations.items()):
        ax.plot(x, val, 'b-', label='Activation', linewidth=2)
        ax.plot(x, grad, 'r--', label='Gradient', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_title(f'{name} Activation & Gradient')
        ax.set_xlabel('x')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("梯度流动可视化已保存: gradient_flow.png")


# ============================================================================
# 测试与验证
# ============================================================================

def run_gradient_check_demo():
    """运行梯度检验演示"""
    print("=" * 60)
    print("梯度检验演示")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    x = np.random.randn(10, 20)  # batch_size=10, input_dim=20
    grad = np.random.randn(10, 15)  # 输出维度=15
    
    # 创建层
    layer = Dense(20, 15)
    
    # 运行梯度检验
    print("\n[1] 计算解析梯度...")
    analytical, numerical = gradient_check(layer, x, grad)
    
    print("\n[2] 解析梯度 (前5个元素):")
    print(analytical.flatten()[:5])
    
    print("\n[3] 数值梯度 (前5个元素):")
    print(numerical.flatten()[:5])
    
    print("\n[4] 梯度比较结果:")
    passed = compare_gradients(analytical, numerical)
    
    if passed:
        print("✅ 梯度检验通过！解析梯度与数值梯度一致。")
    else:
        print("❌ 梯度检验失败！请检查反向传播实现。")
    
    return passed


def run_training_demo():
    """运行完整训练演示"""
    print("\n" + "=" * 60)
    print("MLP训练演示")
    print("=" * 60)
    
    # 生成训练数据（简单分类）
    np.random.seed(42)
    n_samples = 500
    
    # 类别1: (-1, -1) 附近
    X1 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    y1 = np.zeros((n_samples, 1))
    
    # 类别2: (1, 1) 附近
    X2 = np.random.randn(n_samples, 2) + np.array([2, 2])
    y2 = np.ones((n_samples, 1))
    
    X_train = np.vstack([X1, X2])
    y_train = np.vstack([y1, y2])
    
    # 打乱数据
    shuffle_idx = np.random.permutation(2 * n_samples)
    X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
    
    # 创建模型
    model = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1, activation='relu')
    
    # 使用Adam优化器
    optimizer = Adam(lr=0.01)
    
    # 训练
    print("\n开始训练...")
    losses = train(model, X_train, y_train, optimizer, epochs=100, batch_size=32)
    
    # 可视化训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # 可视化决策边界
    plt.subplot(1, 2, 2)
    h = 0.1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:n_samples, 0], X_train[:n_samples, 1], c='blue', label='Class 0')
    plt.scatter(X_train[n_samples:, 0], X_train[n_samples:, 1], c='red', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n训练结果可视化已保存: training_result.png")
    
    # 计算准确率
    predictions = (model.predict(X_train) > 0.5).astype(int)
    accuracy = np.mean(predictions == y_train)
    print(f"\n最终准确率: {accuracy:.2%}")


def demo_backprop_manual():
    """手动演示反向传播计算"""
    print("\n" + "=" * 60)
    print("手动反向传播演示")
    print("=" * 60)
    
    # 简化示例：单层网络
    # y = w * x + b, loss = (y - y_true)^2
    
    x = 2.0  # 输入
    w = 0.5  # 权重
    b = 0.1  # 偏置
    y_true = 1.0  # 真实值
    
    # 前向传播
    z = w * x + b  # 加权求和
    y = 1 / (1 + np.exp(-z))  # Sigmoid激活
    loss = 0.5 * (y - y_true) ** 2  # MSE损失
    
    print(f"\n前向传播:")
    print(f"  x = {x}")
    print(f"  w = {w}, b = {b}")
    print(f"  z = w*x + b = {w}*{x} + {b} = {z}")
    print(f"  y = σ(z) = {y:.6f}")
    print(f"  loss = {loss:.6f}")
    
    # 反向传播
    # ∂L/∂y = y - y_true
    dL_dy = y - y_true
    
    # ∂y/∂z = y * (1 - y)
    dy_dz = y * (1 - y)
    
    # ∂L/∂z = ∂L/∂y * ∂y/∂z
    dL_dz = dL_dy * dy_dz
    
    # ∂z/∂w = x, ∂z/∂b = 1
    dL_dw = dL_dz * x
    dL_db = dL_dz
    
    print(f"\n反向传播:")
    print(f"  ∂L/∂y = {dL_dy:.6f}")
    print(f"  ∂y/∂z = y(1-y) = {dy_dz:.6f}")
    print(f"  ∂L/∂z = {dL_dz:.6f}")
    print(f"  ∂L/∂w = ∂L/∂z * x = {dL_dw:.6f}")
    print(f"  ∂L/∂b = ∂L/∂z * 1 = {dL_db:.6f}")
    
    # 参数更新 (学习率=0.1)
    lr = 0.1
    w_new = w - lr * dL_dw
    b_new = b - lr * dL_db
    
    print(f"\n参数更新 (lr={lr}):")
    print(f"  w_new = {w} - {lr} * {dL_dw:.6f} = {w_new:.6f}")
    print(f"  b_new = {b} - {lr} * {dL_db:.6f} = {b_new:.6f}")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "反向传播实现练习" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 1. 梯度检验
    run_gradient_check_demo()
    
    # 2. 手动演示
    demo_backprop_manual()
    
    # 3. 完整训练
    run_training_demo()
    
    # 4. 梯度流动可视化
    visualize_gradient_flow()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
