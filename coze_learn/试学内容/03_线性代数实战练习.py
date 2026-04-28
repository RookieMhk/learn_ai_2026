#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性代数实战练习 - NumPy实现与可视化
AI学习计划 - 第一阶段：基础夯实 - 线性代数核心

本文件包含线性代数在机器学习中应用的实战练习，通过代码加深理解。
建议在Jupyter Notebook或Python环境中运行。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("线性代数实战练习 - NumPy实现与可视化")
print("=" * 60)

# ============================================================================
# 1. 向量基础操作
# ============================================================================
print("\n1. 向量基础操作")
print("-" * 40)

# 创建向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")

# 向量运算
print(f"\n向量加法: v1 + v2 = {v1 + v2}")
print(f"向量点积: v1 · v2 = {np.dot(v1, v2)}")
print(f"向量范数 (L2): ||v1|| = {np.linalg.norm(v1):.4f}")
print(f"向量夹角余弦值: cosθ = {np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)):.4f}")

# ============================================================================
# 2. 矩阵基础操作
# ============================================================================
print("\n\n2. 矩阵基础操作")
print("-" * 40)

# 创建矩阵
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
print(f"矩阵 A:\n{A}")
print(f"\n矩阵 B:\n{B}")

# 矩阵运算
print(f"\n矩阵加法:\n{A + B}")
print(f"\n矩阵乘法:\n{np.matmul(A, B)}")
print(f"\n矩阵转置 Aᵀ:\n{A.T}")
print(f"\n矩阵行列式: det(A) = {np.linalg.det(A):.4f}")

# ============================================================================
# 3. 特征值与特征向量
# ============================================================================
print("\n\n3. 特征值与特征向量")
print("-" * 40)

# 创建一个对称矩阵（特征值均为实数）
C = np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]])
print(f"对称矩阵 C:\n{C}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(C)
print(f"\n特征值: {eigenvalues}")
print(f"\n特征向量 (每列对应一个特征值):\n{eigenvectors}")

# 验证特征值定义：Cv = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i] * v
    Cv = np.dot(C, v)
    error = np.linalg.norm(Cv - lambda_v)
    print(f"特征值 λ{i+1} = {eigenvalues[i]:.4f}, 验证误差: {error:.6f}")

# ============================================================================
# 4. 奇异值分解 (SVD)
# ============================================================================
print("\n\n4. 奇异值分解 (SVD)")
print("-" * 40)

# 创建一个矩形矩阵
D = np.array([[1, 2, 3], [4, 5, 6]])
print(f"矩阵 D (2×3):\n{D}")

# SVD分解
U, Sigma, Vt = np.linalg.svd(D, full_matrices=False)
print(f"\nU矩阵 (2×2):\n{U}")
print(f"\n奇异值 Σ: {Sigma}")
print(f"\nVᵀ矩阵 (2×3):\n{Vt}")

# 重构矩阵
Sigma_matrix = np.diag(Sigma)
D_reconstructed = np.dot(U, np.dot(Sigma_matrix, Vt))
print(f"\n重构矩阵 D':\n{D_reconstructed}")
print(f"重构误差: {np.linalg.norm(D - D_reconstructed):.6f}")

# ============================================================================
# 5. 线性回归的矩阵解法
# ============================================================================
print("\n\n5. 线性回归的矩阵解法")
print("-" * 40)

# 生成模拟数据
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)
y = 3 + 4 * X + np.random.randn(n_samples, 1)

# 添加偏置项
X_b = np.c_[np.ones((n_samples, 1)), X]  # [1, X]

# 正规方程解：θ = (XᵀX)⁻¹Xᵀy
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"线性回归参数: θ₀ = {theta[0][0]:.4f}, θ₁ = {theta[1][0]:.4f}")
print(f"真实参数: θ₀ = 3, θ₁ = 4")

# ============================================================================
# 6. 主成分分析 (PCA) 可视化
# ============================================================================
print("\n\n6. 主成分分析 (PCA) 可视化")
print("-" * 40)

# 生成二维数据
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
X_pca = np.random.multivariate_normal(mean, cov, 100)

# 中心化数据
X_centered = X_pca - np.mean(X_pca, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)

# 计算特征值和特征向量
eigvals, eigvecs = np.linalg.eig(cov_matrix)

print(f"协方差矩阵:\n{cov_matrix}")
print(f"\n特征值: {eigvals}")
print(f"特征向量:\n{eigvecs}")

# ============================================================================
# 7. 注意力机制模拟
# ============================================================================
print("\n\n7. 注意力机制模拟")
print("-" * 40)

# 模拟Transformer中的自注意力
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """缩放点积注意力"""
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

# 创建模拟的查询、键、值矩阵
np.random.seed(42)
n_tokens = 5
d_model = 4

Q = np.random.randn(n_tokens, d_model)
K = np.random.randn(n_tokens, d_model)
V = np.random.randn(n_tokens, d_model)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"查询矩阵 Q (5×4):\n{Q}")
print(f"\n注意力权重矩阵:\n{attention_weights}")
print(f"\n注意力输出 (5×4):\n{output}")

# ============================================================================
# 8. 可视化部分
# ============================================================================
print("\n\n8. 生成可视化图表...")
print("-" * 40)

# 创建图表
fig = plt.figure(figsize=(15, 10))

# 子图1：向量可视化
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
origin = [0, 0, 0]
ax1.quiver(*origin, *v1, color='r', label='v1')
ax1.quiver(*origin, *v2, color='b', label='v2')
ax1.set_xlim([0, 7])
ax1.set_ylim([0, 7])
ax1.set_zlim([0, 7])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('向量可视化')
ax1.legend()

# 子图2：线性回归拟合
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(X, y, alpha=0.5, label='数据点')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta
ax2.plot(X_new, y_predict, 'r-', label=f'y = {theta[0][0]:.2f} + {theta[1][0]:.2f}x')
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('线性回归拟合')
ax2.legend()

# 子图3：PCA主成分
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label='数据点')
# 绘制特征向量
for i in range(len(eigvals)):
    eigvec = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2
    ax3.arrow(0, 0, eigvec[0], eigvec[1], 
              head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
ax3.set_xlabel('特征1')
ax3.set_ylabel('特征2')
ax3.set_title('PCA主成分分析')
ax3.legend(['数据点', '主成分方向'])
ax3.axis('equal')

# 子图4：注意力权重热图
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='YlOrRd', 
            square=True, cbar_kws={"shrink": 0.8})
ax4.set_xlabel('键位置')
ax4.set_ylabel('查询位置')
ax4.set_title('注意力权重热图')

# 子图5：奇异值重要性
ax5 = fig.add_subplot(2, 3, 5)
cumulative_variance = np.cumsum(Sigma) / np.sum(Sigma)
ax5.bar(range(1, len(Sigma) + 1), Sigma, alpha=0.7, label='奇异值')
ax5.plot(range(1, len(Sigma) + 1), cumulative_variance * Sigma[0], 
         'ro-', label='累积方差比例')
ax5.set_xlabel('奇异值序号')
ax5.set_ylabel('奇异值大小')
ax5.set_title('奇异值分解 - 重要性分析')
ax5.legend()

# 子图6：特征值分布
ax6 = fig.add_subplot(2, 3, 6)
angles = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(angles)
circle_y = np.sin(angles)
ax6.plot(circle_x, circle_y, 'k--', alpha=0.3)
ax6.scatter(np.real(eigenvalues), np.imag(eigenvalues), 
            c=range(len(eigenvalues)), cmap='viridis', s=100)
ax6.set_xlabel('实部')
ax6.set_ylabel('虚部')
ax6.set_title('特征值复数分布')
ax6.axis('equal')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/试学内容/线性代数可视化结果.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存到 'outputs/试学内容/线性代数可视化结果.png'")

# ============================================================================
# 9. 综合练习题目
# ============================================================================
print("\n\n9. 综合练习题目")
print("-" * 40)
print("请尝试完成以下练习题目：")
print("")
print("题目1：矩阵秩的计算")
print("  给定矩阵 E = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]")
print("  请计算矩阵E的秩，并解释其几何意义")
print("")
print("题目2：正交投影")
print("  给定向量 u = [1, 2, 3], v = [4, 5, 6]")
print("  计算向量u在v方向上的正交投影")
print("")
print("题目3：伪逆矩阵")
print("  对于矩形矩阵 F = [[1, 2], [3, 4], [5, 6]]")
print("  计算其伪逆矩阵，并验证 Moore-Penrose 条件")
print("")
print("题目4：QR分解")
print("  对矩阵 G = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]")
print("  进行QR分解，并验证分解的正确性")
print("")
print("题目5：在LLM中的应用思考")
print("  1. 解释特征值分解在Transformer中有什么潜在应用？")
print("  2. 奇异值分解如何用于模型压缩和加速推理？")
print("  3. 线性代数概念在Mamba架构中起到了什么作用？")

# ============================================================================
# 10. 参考资料与下一步学习
# ============================================================================
print("\n\n10. 参考资料与下一步学习")
print("-" * 40)
print("推荐学习资源：")
print("  1. 《深度学习》(花书) - 第2章 线性代数")
print("  2. 吴恩达机器学习课程 - 线性代数复习")
print("  3. NumPy官方文档 - 线性代数模块")
print("")
print("下一步学习建议：")
print("  1. 概率统计基础：贝叶斯定理、概率分布")
print("  2. 微积分核心：梯度下降、反向传播")
print("  3. Python数据处理：Pandas、Matplotlib进阶")

print("\n" + "=" * 60)
print("练习完成！建议将代码复制到Jupyter Notebook中逐步运行")
print("理解每个操作背后的数学原理和AI应用场景")
print("=" * 60)

# 显示图表
plt.show()

print("\n提示：如果要保存可视化结果，请取消 plt.show() 的注释")
print("或将最后一行改为：")
print("plt.savefig('线性代数可视化结果.png', dpi=300, bbox_inches='tight')")
print("plt.close()")