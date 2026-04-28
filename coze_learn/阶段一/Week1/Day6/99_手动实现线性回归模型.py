#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Day 6: 手动实现线性回归模型
AI学习计划 - 第一阶段：基础夯实 - 机器学习项目实战

本文件从零开始手动实现线性回归模型，包含完整的梯度下降优化过程。
通过代码实践深入理解线性回归的数学原理和优化算法的运作机制。

学习目标：
1. 理解线性回归模型的数学公式与损失函数
2. 掌握梯度下降算法的实现细节
3. 学习如何可视化训练过程与模型性能
4. 实现学习率调整机制，理解其对训练的影响
5. 对比不同优化策略的效果

时间投入：约2-3小时
前置要求：已安装NumPy、Matplotlib（若未安装：pip install numpy matplotlib）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Week 1 Day 6: 手动实现线性回归模型")
print("=" * 60)

# ============================================================================
# 第一部分：生成模拟数据
# ============================================================================
print("\n第一部分：生成模拟数据")
print("-" * 50)

def generate_linear_data(n_samples=200, noise_level=0.5, random_seed=42):
    """
    生成线性回归的模拟数据
    
    参数：
    n_samples: 样本数量
    noise_level: 噪声水平
    random_seed: 随机种子
    
    返回：
    X: 特征矩阵 (n_samples, 1)
    y: 目标值 (n_samples,)
    true_slope: 真实斜率
    true_intercept: 真实截距
    """
    np.random.seed(random_seed)
    
    # 生成特征X（在0-10之间均匀分布）
    X = np.random.rand(n_samples, 1) * 10
    
    # 真实参数：斜率=2.5，截距=1.0
    true_slope = 2.5
    true_intercept = 1.0
    
    # 生成目标值（带噪声）
    y = true_intercept + true_slope * X.flatten()
    noise = np.random.randn(n_samples) * noise_level
    y += noise
    
    return X, y, true_slope, true_intercept

# 生成数据
X, y, true_slope, true_intercept = generate_linear_data(n_samples=200, noise_level=1.0)

print(f"数据形状: X{X.shape}, y{y.shape}")
print(f"真实参数: 截距={true_intercept:.2f}, 斜率={true_slope:.2f}")
print(f"特征范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"目标范围: [{y.min():.2f}, {y.max():.2f}]")

# ============================================================================
# 第二部分：数据预处理
# ============================================================================
print("\n第二部分：数据预处理")
print("-" * 50)

# 添加偏置项（在特征矩阵前添加一列1）
def add_bias_term(X):
    """
    为特征矩阵添加偏置项（一列1）
    线性回归模型：y = w0*1 + w1*x1 + ...
    其中w0是截距，对应偏置项
    """
    # np.c_用于按列连接数组
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return X_with_bias

X_with_bias = add_bias_term(X)
print(f"添加偏置项后的特征矩阵形状: {X_with_bias.shape}")
print(f"第一行示例: {X_with_bias[0]} (偏置项=1, 特征值={X[0][0]:.2f})")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_with_bias, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]}个样本")
print(f"测试集: {X_test.shape[0]}个样本")

# ============================================================================
# 第三部分：线性回归模型实现
# ============================================================================
print("\n第三部分：线性回归模型实现")
print("-" * 50)

class LinearRegressionManual:
    """
    手动实现的线性回归模型
    
    核心方法：
    1. fit: 使用梯度下降训练模型
    2. predict: 使用训练好的模型进行预测
    3. compute_cost: 计算均方误差损失
    4. compute_gradient: 计算损失函数梯度
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化线性回归模型
        
        参数：
        learning_rate: 学习率（步长）
        n_iterations: 梯度下降迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # 模型参数 [w0, w1, ...]
        self.cost_history = []  # 记录每次迭代的损失值
        self.weight_history = []  # 记录参数变化轨迹（仅前两个参数）
    
    def initialize_weights(self, n_features):
        """
        初始化权重参数
        
        参数：
        n_features: 特征数量（包括偏置项）
        
        返回：
        weights: 初始化后的权重向量
        """
        # 使用小随机数初始化权重，打破对称性
        # 注意：线性回归的损失函数是凸函数，任何初始化都会收敛到全局最优
        self.weights = np.random.randn(n_features) * 0.01
        return self.weights
    
    def predict(self, X):
        """
        使用当前权重进行预测
        
        参数：
        X: 特征矩阵（已包含偏置项）
        
        返回：
        y_pred: 预测值
        """
        # 线性模型：y_pred = X * w
        y_pred = np.dot(X, self.weights)
        return y_pred
    
    def compute_cost(self, X, y):
        """
        计算均方误差损失
        
        参数：
        X: 特征矩阵
        y: 真实标签
        
        返回：
        cost: 均方误差损失值
        """
        m = X.shape[0]  # 样本数量
        y_pred = self.predict(X)
        
        # 均方误差公式: (1/2m) * Σ(y_pred - y)^2
        # 1/2是为了梯度计算时的简化
        error = y_pred - y
        cost = (1 / (2 * m)) * np.dot(error, error)
        return cost
    
    def compute_gradient(self, X, y):
        """
        计算损失函数的梯度
        
        参数：
        X: 特征矩阵
        y: 真实标签
        
        返回：
        gradient: 梯度向量
        """
        m = X.shape[0]  # 样本数量
        y_pred = self.predict(X)
        error = y_pred - y
        
        # 梯度公式: (1/m) * X^T * (Xw - y)
        gradient = (1 / m) * np.dot(X.T, error)
        return gradient
    
    def fit(self, X, y, verbose=False):
        """
        使用梯度下降训练线性回归模型
        
        参数：
        X: 特征矩阵（已包含偏置项）
        y: 真实标签
        verbose: 是否打印训练过程
        
        返回：
        self: 训练好的模型
        """
        # 获取特征数量
        n_features = X.shape[1]
        
        # 初始化权重
        self.initialize_weights(n_features)
        
        # 清空历史记录
        self.cost_history = []
        self.weight_history = []
        
        print(f"开始梯度下降训练...")
        print(f"初始权重: {self.weights}")
        
        # 梯度下降主循环
        for i in range(self.n_iterations):
            # 计算当前损失
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # 记录权重变化（仅前两个权重）
            if n_features <= 2:
                self.weight_history.append(self.weights.copy())
            
            # 计算梯度
            gradient = self.compute_gradient(X, y)
            
            # 更新权重
            self.weights -= self.learning_rate * gradient
            
            # 每100次迭代打印一次进度
            if verbose and i % 100 == 0:
                print(f"迭代 {i:4d}: 损失 = {cost:.6f}, 梯度范数 = {np.linalg.norm(gradient):.6f}")
        
        # 计算最终损失
        final_cost = self.compute_cost(X, y)
        print(f"训练完成！最终损失: {final_cost:.6f}")
        print(f"最终权重: {self.weights}")
        
        return self
    
    def fit_with_learning_rate_decay(self, X, y, decay_rate=0.95, verbose=False):
        """
        使用学习率衰减的梯度下降训练模型
        
        参数：
        X: 特征矩阵
        y: 真实标签
        decay_rate: 学习率衰减率
        verbose: 是否打印训练过程
        
        返回：
        self: 训练好的模型
        """
        n_features = X.shape[1]
        self.initialize_weights(n_features)
        
        self.cost_history = []
        self.weight_history = []
        
        current_lr = self.learning_rate
        
        print(f"开始带学习率衰减的梯度下降训练...")
        print(f"初始学习率: {current_lr}")
        
        for i in range(self.n_iterations):
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            if n_features <= 2:
                self.weight_history.append(self.weights.copy())
            
            gradient = self.compute_gradient(X, y)
            
            # 更新权重
            self.weights -= current_lr * gradient
            
            # 学习率衰减
            if i % 100 == 0 and i > 0:
                current_lr *= decay_rate
            
            if verbose and i % 100 == 0:
                print(f"迭代 {i:4d}: 损失 = {cost:.6f}, 学习率 = {current_lr:.6f}")
        
        final_cost = self.compute_cost(X, y)
        print(f"训练完成！最终损失: {final_cost:.6f}")
        print(f"最终权重: {self.weights}")
        
        return self

# ============================================================================
# 第四部分：模型训练与可视化
# ============================================================================
print("\n第四部分：模型训练与可视化")
print("-" * 50)

# 训练第一个模型（固定学习率）
print("\n1. 固定学习率训练")
model1 = LinearRegressionManual(learning_rate=0.01, n_iterations=1000)
model1.fit(X_train, y_train, verbose=True)

# 训练第二个模型（带学习率衰减）
print("\n2. 带学习率衰减的训练")
model2 = LinearRegressionManual(learning_rate=0.05, n_iterations=1000)
model2.fit_with_learning_rate_decay(X_train, y_train, decay_rate=0.98, verbose=True)

# ============================================================================
# 第五部分：结果可视化
# ============================================================================
print("\n第五部分：结果可视化")
print("-" * 50)

def plot_training_results(model, X_train, y_train, X_test, y_test, model_name="模型"):
    """
    绘制模型训练结果
    
    参数：
    model: 训练好的模型
    X_train: 训练特征
    y_train: 训练标签
    X_test: 测试特征
    y_test: 测试标签
    model_name: 模型名称（用于标题）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 损失函数收敛曲线
    axes[0, 0].plot(model.cost_history)
    axes[0, 0].set_title(f'{model_name} - 损失函数收敛曲线')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('均方误差损失')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 对损失曲线取对数，更容易观察收敛情况
    axes[0, 1].plot(np.log(model.cost_history))
    axes[0, 1].set_title(f'{model_name} - 损失函数（对数尺度）')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('log(损失)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. 回归拟合图
    # 生成预测值
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 提取原始特征（去掉偏置项）
    X_train_original = X_train[:, 1] if X_train.shape[1] > 1 else X_train.flatten()
    X_test_original = X_test[:, 1] if X_test.shape[1] > 1 else X_test.flatten()
    
    axes[1, 0].scatter(X_train_original, y_train, alpha=0.6, label='训练数据', color='blue')
    axes[1, 0].scatter(X_test_original, y_test, alpha=0.6, label='测试数据', color='green')
    
    # 绘制回归线
    # 生成用于绘制回归线的X值
    x_line = np.linspace(X_train_original.min(), X_train_original.max(), 100)
    # 添加偏置项
    x_line_with_bias = np.c_[np.ones(100), x_line.reshape(-1, 1)]
    y_line = model.predict(x_line_with_bias)
    
    axes[1, 0].plot(x_line, y_line, 'r-', linewidth=2, label='回归线')
    axes[1, 0].set_title(f'{model_name} - 回归拟合效果')
    axes[1, 0].set_xlabel('特征X')
    axes[1, 0].set_ylabel('目标y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. 参数空间轨迹（仅当有2个参数时）
    if len(model.weight_history) > 0:
        weights_array = np.array(model.weight_history)
        axes[1, 1].plot(weights_array[:, 0], weights_array[:, 1], 'b-', alpha=0.6)
        axes[1, 1].scatter(weights_array[:, 0], weights_array[:, 1], c=range(len(weights_array)), 
                          cmap='viridis', alpha=0.8, s=20)
        axes[1, 1].scatter([weights_array[-1, 0]], [weights_array[-1, 1]], 
                          color='red', s=100, marker='*', label='最终参数')
        axes[1, 1].set_title(f'{model_name} - 参数空间轨迹')
        axes[1, 1].set_xlabel('截距 w0')
        axes[1, 1].set_ylabel('斜率 w1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '参数空间轨迹仅适用于\n二维特征空间', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'{model_name} - 参数空间轨迹')
    
    plt.tight_layout()
    plt.show()
    
    # 打印模型性能
    train_cost = model.compute_cost(X_train, y_train)
    test_cost = model.compute_cost(X_test, y_test)
    
    print(f"{model_name}性能:")
    print(f"  训练集损失: {train_cost:.6f}")
    print(f"  测试集损失: {test_cost:.6f}")
    print(f"  参数值: {model.weights}")
    
    # 计算R²分数
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"  训练集R²: {r2_train:.4f}")
    print(f"  测试集R²: {r2_test:.4f}")

# 绘制第一个模型的结果
plot_training_results(model1, X_train, y_train, X_test, y_test, "固定学习率模型")

# 绘制第二个模型的结果
plot_training_results(model2, X_train, y_train, X_test, y_test, "学习率衰减模型")

# ============================================================================
# 第六部分：学习率对比实验
# ============================================================================
print("\n第六部分：学习率对比实验")
print("-" * 50)

def compare_learning_rates(X_train, y_train, learning_rates=[0.001, 0.01, 0.1, 0.5]):
    """
    比较不同学习率对训练的影响
    
    参数：
    X_train: 训练特征
    y_train: 训练标签
    learning_rates: 要比较的学习率列表
    """
    plt.figure(figsize=(10, 6))
    
    for lr in learning_rates:
        model = LinearRegressionManual(learning_rate=lr, n_iterations=500)
        model.fit(X_train, y_train, verbose=False)
        
        # 绘制损失曲线
        plt.plot(model.cost_history, label=f'学习率={lr}')
    
    plt.title('不同学习率下的损失收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 打印不同学习率的最终损失
    print("\n不同学习率训练结果对比:")
    print("学习率\t最终损失\t收敛速度")
    print("-" * 40)
    
    for lr in learning_rates:
        model = LinearRegressionManual(learning_rate=lr, n_iterations=500)
        model.fit(X_train, y_train, verbose=False)
        final_cost = model.cost_history[-1]
        
        # 估计收敛速度：损失下降到初始值10%所需的迭代次数
        initial_cost = model.cost_history[0]
        target_cost = initial_cost * 0.1
        
        # 找到第一个低于目标损失的迭代
        convergence_iter = None
        for i, cost in enumerate(model.cost_history):
            if cost <= target_cost:
                convergence_iter = i
                break
        
        conv_speed = f"{convergence_iter}" if convergence_iter else "未收敛"
        print(f"{lr}\t{final_cost:.6f}\t{conv_speed}次迭代")

# 执行学习率对比实验
compare_learning_rates(X_train, y_train)

# ============================================================================
# 第七部分：模型保存与应用
# ============================================================================
print("\n第七部分：模型保存与应用")
print("-" * 50)

def save_model(model, filename):
    """
    保存模型参数到文件
    
    参数：
    model: 训练好的模型
    filename: 保存文件名
    """
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'weights': model.weights,
            'learning_rate': model.learning_rate,
            'n_iterations': model.n_iterations
        }, f)
    
    print(f"模型已保存到 {filename}")

def load_model(filename):
    """
    从文件加载模型参数
    
    参数：
    filename: 模型文件名
    
    返回：
    model: 加载的模型
    """
    import pickle
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    model = LinearRegressionManual(
        learning_rate=data['learning_rate'],
        n_iterations=data['n_iterations']
    )
    model.weights = data['weights']
    
    print(f"模型已从 {filename} 加载")
    return model

# 保存最佳模型
save_model(model1, 'outputs/阶段一/Week1/Day6/linear_regression_model.pkl')

# 示例：加载模型并预测
print("\n示例：加载模型并进行预测")
loaded_model = load_model('outputs/阶段一/Week1/Day6/linear_regression_model.pkl')

# 对测试集进行预测
y_pred = loaded_model.predict(X_test)
print(f"前5个测试样本预测值: {y_pred[:5]}")
print(f"前5个测试样本真实值: {y_test[:5]}")

# ============================================================================
# 第八部分：总结与思考题
# ============================================================================
print("\n" + "=" * 60)
print("总结与思考题")
print("=" * 60)

print("""
🎯 今日学习成果总结：

1. **数学原理掌握**：深入理解了线性回归模型、均方误差损失函数和梯度下降算法
2. **代码实现能力**：从零实现了完整的线性回归模型，包括梯度下降优化
3. **可视化分析**：通过损失曲线、回归拟合图等可视化工具分析模型性能
4. **超参数调优**：探索了学习率对训练过程的影响，实现了学习率衰减机制

🤔 关键思考题：

1. **梯度下降的收敛性**：为什么线性回归使用梯度下降总能找到全局最优解？
2. **学习率的选择**：学习率过大或过小分别会导致什么问题？如何选择合适的学习率？
3. **特征缩放的重要性**：如果不进行特征缩放，梯度下降会遇到什么困难？
4. **批量梯度下降 vs 随机梯度下降**：两者各有什么优缺点？分别适用于什么场景？

🔧 进一步探索建议：

1. **添加正则化**：实现L1/L2正则化，防止过拟合
2. **实现随机梯度下降**：每次使用一个样本更新权重，观察收敛速度变化
3. **多特征扩展**：将当前单特征模型扩展到多特征场景
4. **多项式回归**：通过特征工程实现非线性关系的拟合

📈 下一阶段学习：

完成今日实战后，你已掌握了机器学习的第一个核心模型！下一阶段将学习：
- 逻辑回归（分类问题）
- 决策树与随机森林
- 支持向量机
- 神经网络基础

请记得填写今日执行卡片，记录你的学习收获与疑问！
""")

print("\n" + "=" * 60)
print("代码执行完成！")
print("=" * 60)