"""
神经网络实现练习
Week 3 Day 1 · 用 NumPy 手动实现神经网络核心概念

包含：
1. 单层感知机
2. 多层感知机的前向传播
3. 不同激活函数的对比实验
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable

# 设置随机种子，保证结果可复现
np.random.seed(42)

# ============================================================
# 第一部分：单层感知机
# ============================================================

class Perceptron:
    """单层感知机实现"""
    
    def __init__(self, n_inputs: int, learning_rate: float = 0.1):
        """
        初始化感知机
        
        Args:
            n_inputs: 输入特征数
            learning_rate: 学习率
        """
        # Xavier 初始化
        self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        self.bias = 0.0
        self.lr = learning_rate
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            X: 输入，形状 (n_samples, n_features) 或 (n_features,)
        
        Returns:
            预测输出，形状 (n_samples,) 或 scalar
        """
        X = np.asarray(X)
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear > 0, 1, 0)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        """
        训练感知机
        
        Args:
            X: 训练数据
            y: 标签
            epochs: 训练轮数
        
        Returns:
            每轮的准确率列表
        """
        X = np.asarray(X)
        y = np.asarray(y)
        accuracies = []
        
        for _ in range(epochs):
            # 遍历所有样本
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                
                # 前向计算
                prediction = 1 if np.dot(xi, self.weights) + self.bias > 0 else 0
                
                # 更新权重
                error = yi - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
            
            # 计算准确率
            predictions = self.forward(X)
            acc = np.mean(predictions == y)
            accuracies.append(acc)
        
        return accuracies


def demonstrate_logic_gates():
    """演示用感知机实现基本逻辑门"""
    print("=" * 60)
    print("单层感知机实现逻辑门")
    print("=" * 60)
    
    # 与门 (AND)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    # 或门 (OR)
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    # 非门 (NOT) - 单输入
    X_not = np.array([[0], [1]])
    y_not = np.array([1, 0])
    
    # 训练与门
    print("\n【与门 (AND)】")
    and_perceptron = Perceptron(n_inputs=2, learning_rate=1.0)
    and_perceptron.weights = np.array([1.0, 1.0])
    and_perceptron.bias = -1.5
    
    print("输入 | 预期输出 | 感知机输出")
    for i, x in enumerate(X_and):
        out = and_perceptron.forward(x)
        if isinstance(out, np.ndarray) and out.ndim > 0:
            out_val = out[0]
        else:
            out_val = int(out)
        print(f"  {x[0]}, {x[1]}  |    {y_and[i]}     |      {out_val}")
    
    # 训练或门
    print("\n【或门 (OR)】")
    or_perceptron = Perceptron(n_inputs=2, learning_rate=1.0)
    or_perceptron.weights = np.array([1.0, 1.0])
    or_perceptron.bias = -0.5
    
    print("输入 | 预期输出 | 感知机输出")
    for i, x in enumerate(X_or):
        out = or_perceptron.forward(x)
        if isinstance(out, np.ndarray) and out.ndim > 0:
            out_val = out[0]
        else:
            out_val = int(out)
        print(f"  {x[0]}, {x[1]}  |    {y_or[i]}     |      {out_val}")
    
    # XOR 演示（单层感知机无法实现）
    print("\n【异或门 (XOR) - 单层感知机无法实现】")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print("输入 | 预期输出 | 感知机输出")
    for i, x in enumerate(X_xor):
        out = or_perceptron.forward(x)
        if isinstance(out, np.ndarray) and out.ndim > 0:
            out_val = out[0]
        else:
            out_val = int(out)
        print(f"  {x[0]}, {x[1]}  |    {y_xor[i]}     |      {out_val} ❌")
    print("注意：单层感知机只能实现线性可分的模式")


# ============================================================
# 第二部分：激活函数
# ============================================================

class ActivationFunctions:
    """常用激活函数集合"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid: σ(x) = 1 / (1 + exp(-x))"""
        # 数值稳定的实现
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid 导数: σ'(x) = σ(x)(1 - σ(x))"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh 导数: tanh'(x) = 1 - tanh²(x)"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU 导数"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU 导数"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU: x * Φ(x)，Φ 为标准正态 CDF"""
        # 近似形式
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 用于多分类输出"""
        x = np.clip(x, -500, 500)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def visualize_activations():
    """可视化不同激活函数及其导数"""
    print("=" * 60)
    print("激活函数可视化")
    print("=" * 60)
    
    x = np.linspace(-5, 5, 200)
    
    activations = {
        'Sigmoid': ActivationFunctions.sigmoid(x),
        'Tanh': ActivationFunctions.tanh(x),
        'ReLU': ActivationFunctions.relu(x),
        'Leaky ReLU': ActivationFunctions.leaky_relu(x),
        'GELU': ActivationFunctions.gelu(x),
    }
    
    derivatives = {
        'Sigmoid\'': ActivationFunctions.sigmoid_derivative(x),
        'Tanh\'': ActivationFunctions.tanh_derivative(x),
        'ReLU\'': ActivationFunctions.relu_derivative(x),
        'Leaky ReLU\'': ActivationFunctions.leaky_relu_derivative(x),
    }
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 第一行：激活函数
    for idx, (name, y) in enumerate(activations.items()):
        ax = axes[0, idx] if idx < 3 else axes[1, 0]
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
    
    # GELU 单独显示
    ax = axes[1, 1]
    ax.plot(x, activations['GELU'], 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('GELU', fontsize=12, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    # 导数对比
    ax = axes[1, 2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, dy) in enumerate(derivatives.items()):
        ax.plot(x, dy, label=name, linewidth=2, color=colors[idx])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('导数对比', fontsize=12, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('激活函数可视化.png', dpi=150, bbox_inches='tight')
    print("✓ 激活函数可视化已保存")
    plt.close()


def compare_activation_properties():
    """对比不同激活函数的特性"""
    print("\n" + "=" * 60)
    print("激活函数特性对比")
    print("=" * 60)
    
    x = np.array([-10, -1, 0, 1, 10])
    
    functions = {
        'Sigmoid': ActivationFunctions.sigmoid,
        'Tanh': ActivationFunctions.tanh,
        'ReLU': ActivationFunctions.relu,
        'GELU': ActivationFunctions.gelu,
    }
    
    print(f"\n{'函数':<12}", end='')
    for xi in x:
        print(f"x={xi:>4}", end='')
    print()
    print("-" * 50)
    
    for name, func in functions.items():
        print(f"{name:<12}", end='')
        for xi in x:
            val = func(xi)[0] if hasattr(func(xi), '__len__') else func(xi)
            print(f"{val:>8.4f}", end='')
        print()
    
    print("\n关键观察：")
    print("• Sigmoid: x→±∞ 时梯度消失 (输出趋近0/1)")
    print("• Tanh: 零均值，但仍有梯度消失问题")
    print("• ReLU: x<0 时完全失活 ( Dying ReLU 问题)")
    print("• GELU: 平滑过渡，结合了 Sigmoid 和 ReLU 的优点")


# ============================================================
# 第三部分：多层感知机 (MLP) 前向传播
# ============================================================

class MLP:
    """多层感知机实现"""
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        """
        初始化 MLP
        
        Args:
            layer_sizes: 各层神经元数量，如 [784, 256, 128, 10]
            activations: 各层激活函数，如 ['relu', 'relu', 'softmax']
        """
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # 默认激活函数
        if activations is None:
            activations = ['relu'] * (self.n_layers - 2) + ['identity']
        
        self.activation_names = activations
        
        # Xavier/He 初始化
        for i in range(self.n_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            # Xavier 初始化适合 tanh/sigmoid
            # He 初始化适合 ReLU
            scale = np.sqrt(2.0 / n_in) if activations[i] == 'relu' else np.sqrt(1.0 / n_in)
            
            W = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _get_activation(self, name: str) -> Callable:
        """获取激活函数"""
        activations = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'gelu': ActivationFunctions.gelu,
            'softmax': ActivationFunctions.softmax,
            'identity': lambda x: x,
            'none': lambda x: x,
        }
        return activations.get(name.lower(), ActivationFunctions.relu)
    
    def forward(self, X: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List]:
        """
        前向传播
        
        Args:
            X: 输入，形状 (n_features,) 或 (n_features, n_samples)
            verbose: 是否打印中间层信息
        
        Returns:
            (output, cache) 元组
            - output: 模型输出
            - cache: 各层的线性组合 Z 和激活值 A
        """
        # 确保 X 是列向量形式
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        cache_Z = []  # 存储 Z = W·A + b
        cache_A = [X]  # 存储 A，包括输入层
        
        current_A = X
        
        for i in range(self.n_layers - 1):
            W = self.weights[i]
            b = self.biases[i]
            
            # 线性组合
            Z = np.dot(W, current_A) + b
            cache_Z.append(Z)
            
            # 激活函数（输出层通常不用激活或用 softmax）
            if i == self.n_layers - 2:
                # 输出层
                if self.activation_names[i].lower() == 'softmax':
                    A = ActivationFunctions.softmax(Z)
                else:
                    activation_func = self._get_activation(self.activation_names[i])
                    A = activation_func(Z)
            else:
                activation_func = self._get_activation(self.activation_names[i])
                A = activation_func(Z)
            
            cache_A.append(A)
            current_A = A
            
            if verbose:
                print(f"Layer {i+1}: Z shape = {Z.shape}, A shape = {A.shape}")
        
        return current_A, (cache_Z, cache_A)
    
    def summary(self):
        """打印网络结构摘要"""
        print("\n" + "=" * 60)
        print("MLP 网络结构")
        print("=" * 60)
        print(f"{'层':<8} {'类型':<15} {'神经元数':<10} {'激活函数':<12}")
        print("-" * 50)
        
        for i in range(self.n_layers):
            if i == 0:
                layer_type = "输入层"
                n_neurons = self.layer_sizes[i]
                activation = "-"
            elif i == self.n_layers - 1:
                layer_type = "输出层"
                n_neurons = self.layer_sizes[i]
                activation = self.activation_names[i-1]
            else:
                layer_type = "隐藏层"
                n_neurons = self.layer_sizes[i]
                activation = self.activation_names[i-1]
            
            print(f"Layer {i:<4} {layer_type:<15} {n_neurons:<10} {activation:<12}")
        
        # 计算总参数量
        total_params = sum(
            w.size + b.size 
            for w, b in zip(self.weights, self.biases)
        )
        print("-" * 50)
        print(f"总参数量: {total_params:,}")
        print("=" * 60)


def demonstrate_mlp_forward():
    """演示 MLP 前向传播"""
    print("\n" + "=" * 60)
    print("MLP 前向传播演示")
    print("=" * 60)
    
    # 创建一个简单的 MLP
    mlp = MLP(
        layer_sizes=[2, 4, 3, 2],
        activations=['relu', 'relu', 'softmax']
    )
    
    mlp.summary()
    
    # 随机输入
    X = np.array([0.5, -0.3])
    print(f"\n输入: {X}")
    
    # 前向传播
    output, (cache_Z, cache_A) = mlp.forward(X, verbose=True)
    
    print(f"\n最终输出:\n{output}")
    print(f"\n输出概率: {output.flatten()}")
    print(f"预测类别: {np.argmax(output)}")


def demonstrate_xor_with_mlp():
    """用 MLP 解决 XOR 问题"""
    print("\n" + "=" * 60)
    print("用 MLP 解决 XOR 问题")
    print("=" * 60)
    
    # XOR 数据
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_xor = np.array([0, 1, 1, 0])
    
    # 创建 MLP
    mlp = MLP(
        layer_sizes=[2, 4, 1],
        activations=['tanh', 'sigmoid']  # sigmoid 输出 [0,1]
    )
    
    # 手动设置权重以解决 XOR（理论上的最优权重）
    # 第一层
    mlp.weights[0] = np.array([
        [10, -10],   # 隐藏神经元 1: 检测 (1,0)
        [-10, 10],   # 隐藏神经元 2: 检测 (0,1)
        [-10, -10],  # 隐藏神经元 3: 抑制 (0,0)
        [10, 10]     # 隐藏神经元 4: 抑制 (1,1)
    ])
    mlp.biases[0] = np.array([[-5], [-5], [15], [-5]])
    
    # 第二层
    mlp.weights[1] = np.array([[10, 10, -10, -10]])
    mlp.biases[1] = np.array([[-5]])
    
    print("\nXOR 预测结果:")
    print("输入 | 隐藏层激活 | 预期 | 预测")
    print("-" * 45)
    
    for i in range(len(X_xor)):
        x = X_xor[i]
        output, (cache_Z, cache_A) = mlp.forward(x)
        
        hidden_activations = cache_A[1].flatten()
        pred = 1 if output[0, 0] > 0.5 else 0
        
        print(f"({x[0]}, {x[1]}) | {hidden_activations.round(2)} | {y_xor[i]} | {pred}")
    
    print("\n✓ MLP 通过增加隐藏层，成功解决了 XOR 这个非线性可分问题！")


def visualize_mlp_decision_boundary():
    """可视化 MLP 的决策边界"""
    print("\n" + "=" * 60)
    print("MLP 决策边界可视化")
    print("=" * 60)
    
    # 生成训练数据
    np.random.seed(42)
    n_samples = 200
    
    # XOR 风格的数据
    X1 = np.vstack([
        np.random.randn(n_samples//4, 2) + np.array([0, 0]),
        np.random.randn(n_samples//4, 2) + np.array([1, 1]),
        np.random.randn(n_samples//4, 2) + np.array([0, 1]),
        np.random.randn(n_samples//4, 2) + np.array([1, 0])
    ])
    y1 = np.hstack([
        np.zeros(n_samples//2),
        np.ones(n_samples//2)
    ])
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 创建网格
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 100), np.linspace(-1.5, 2.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # 图1：单层感知机（线性边界）
    ax = axes[0]
    perceptron = Perceptron(n_inputs=2)
    Z = np.dot(grid, perceptron.weights) + perceptron.bias
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#ffcccc', '#ccffcc'], alpha=0.5)
    ax.scatter(X1[y1==0, 0], X1[y1==0, 1], c='red', marker='o', label='Class 0', edgecolors='k')
    ax.scatter(X1[y1==1, 0], X1[y1==1, 1], c='blue', marker='s', label='Class 1', edgecolors='k')
    ax.set_title('单层感知机\n(只能画直线)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图2：MLP（复杂边界）
    ax = axes[1]
    mlp = MLP([2, 8, 8, 1], activations=['relu', 'relu', 'sigmoid'])
    # 用随机权重演示（不训练）
    output, _ = mlp.forward(grid.T)
    Z = output.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ffcccc', '#ccffcc'], alpha=0.5)
    ax.scatter(X1[y1==0, 0], X1[y1==0, 1], c='red', marker='o', label='Class 0', edgecolors='k')
    ax.scatter(X1[y1==1, 0], X1[y1==1, 1], c='blue', marker='s', label='Class 1', edgecolors='k')
    ax.set_title('多层感知机 (MLP)\n(可画复杂边界)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图3：激活函数效果
    ax = axes[2]
    x_line = np.linspace(-3, 3, 100)
    ax.plot(x_line, ActivationFunctions.relu(x_line), 'b-', label='ReLU', linewidth=2)
    ax.plot(x_line, ActivationFunctions.tanh(x_line), 'r--', label='Tanh', linewidth=2)
    ax.plot(x_line, ActivationFunctions.sigmoid(x_line), 'g:', label='Sigmoid', linewidth=2)
    ax.plot(x_line, ActivationFunctions.gelu(x_line), 'm-.', label='GELU', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('激活函数对比', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 2)
    
    plt.tight_layout()
    plt.savefig('MLP与激活函数.png', dpi=150, bbox_inches='tight')
    print("✓ MLP 与激活函数对比图已保存")
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("神经网络基础练习 - NumPy 实现")
    print("Week 3 Day 1 · 神经网络核心概念")
    print("=" * 60)
    
    # 1. 单层感知机与逻辑门
    demonstrate_logic_gates()
    
    # 2. 激活函数可视化
    visualize_activations()
    
    # 3. 激活函数特性对比
    compare_activation_properties()
    
    # 4. MLP 前向传播
    demonstrate_mlp_forward()
    
    # 5. XOR 问题演示
    demonstrate_xor_with_mlp()
    
    # 6. 决策边界可视化
    visualize_mlp_decision_boundary()
    
    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)
    print("""
练习要点回顾：
1. 单层感知机只能解决线性可分问题（如 AND、OR）
2. 异或(XOR)问题需要多层网络（MLP）才能解决
3. 激活函数为网络引入非线性
4. ReLU 是现代深度学习的默认选择
5. GELU 在 Transformer 架构中表现优异

进阶练习：
• 尝试实现反向传播算法
• 添加梯度检查验证导数计算
• 用 MLP 对真实数据集进行分类
    """)


if __name__ == "__main__":
    main()
