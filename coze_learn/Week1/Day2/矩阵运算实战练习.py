#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Day 2: NumPy矩阵运算实战练习
AI学习计划 - 第一阶段：基础夯实 - 线性代数核心

本文件包含NumPy矩阵运算的实战练习，重点对比手动实现与NumPy内置函数的性能差异。
通过手动实现加深对算法原理的理解，通过性能对比认识NumPy优化的重要性。

学习目标：
1. 掌握矩阵乘法的手动实现（三重循环）
2. 理解特征值计算的幂迭代法原理
3. 实现简化版SVD分解算法
4. 对比手动实现与NumPy内置函数的性能差异
5. 理解向量化运算的优势

时间投入：约2-3小时
前置要求：已安装NumPy、Matplotlib（若未安装：pip install numpy matplotlib）
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Week 1 Day 2: NumPy矩阵运算实战练习")
print("=" * 60)

# ============================================================================
# 第一部分：矩阵乘法手动实现与性能对比
# ============================================================================
print("\n第一部分：矩阵乘法手动实现与性能对比")
print("-" * 50)

def manual_matrix_multiply(A, B):
    """
    手动实现矩阵乘法（三重循环）
    输入：A (m×n), B (n×p)
    输出：C (m×p) = A × B
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "矩阵维度不匹配"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val
    return C

def numpy_matrix_multiply(A, B):
    """NumPy内置矩阵乘法（作为对比基准）"""
    return np.matmul(A, B)

# 测试不同规模矩阵的性能
matrix_sizes = [10, 50, 100, 200]  # 为避免时间过长，最大200×200
manual_times = []
numpy_times = []

print("\n矩阵乘法性能对比实验：")
print("大小\t手动时间(s)\tNumPy时间(s)\t加速比")

for size in matrix_sizes:
    # 生成随机矩阵
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    
    # 手动实现计时
    start = time.time()
    C_manual = manual_matrix_multiply(A, B)
    manual_time = time.time() - start
    
    # NumPy实现计时
    start = time.time()
    C_numpy = numpy_matrix_multiply(A, B)
    numpy_time = time.time() - start
    
    # 验证结果一致性
    error = np.max(np.abs(C_manual - C_numpy))
    if error > 1e-8:
        print(f"警告：大小{size}的结果误差较大: {error:.6f}")
    
    manual_times.append(manual_time)
    numpy_times.append(numpy_time)
    
    speedup = manual_time / numpy_time if numpy_time > 0 else float('inf')
    print(f"{size}×{size}\t{manual_time:.4f}\t\t{numpy_time:.6f}\t\t{speedup:.1f}x")

# ============================================================================
# 第二部分：特征值计算的幂迭代法实现
# ============================================================================
print("\n\n第二部分：特征值计算的幂迭代法实现")
print("-" * 50)

def power_iteration(A, num_iterations=100, epsilon=1e-10):
    """
    幂迭代法计算矩阵的绝对值最大特征值及对应特征向量
    适用于对称矩阵的主特征值计算
    """
    n = A.shape[0]
    # 随机初始化向量
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    
    eigenvalues = []
    
    for i in range(num_iterations):
        # 计算 A × b_k
        b_k1 = np.dot(A, b_k)
        
        # 计算当前特征值估计（Rayleigh商）
        eigenvalue = np.dot(b_k, b_k1)
        eigenvalues.append(eigenvalue)
        
        # 归一化
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        
        # 检查收敛
        if i > 0 and abs(eigenvalues[-1] - eigenvalues[-2]) < epsilon:
            break
    
    return eigenvalue, b_k, eigenvalues

# 创建一个对称矩阵
np.random.seed(42)
n = 8
M = np.random.randn(n, n)
M_symmetric = (M + M.T) / 2  # 使其对称

print(f"对称矩阵 M (8×8):")
print(M_symmetric)

# 使用幂迭代法计算主特征值
max_eigenvalue, max_eigenvector, history = power_iteration(M_symmetric, num_iterations=50)
print(f"\n幂迭代法结果:")
print(f"最大特征值: {max_eigenvalue:.6f}")
print(f"对应特征向量: {max_eigenvector}")

# 使用NumPy eig计算所有特征值对比
eigenvalues_np, eigenvectors_np = np.linalg.eig(M_symmetric)
max_eigenvalue_np = np.max(np.abs(eigenvalues_np))
idx = np.argmax(np.abs(eigenvalues_np))
max_eigenvector_np = eigenvectors_np[:, idx]

print(f"\nNumPy eig结果:")
print(f"最大特征值: {max_eigenvalue_np:.6f}")
print(f"对应特征向量: {max_eigenvector_np}")

# 计算误差
eigenvalue_error = abs(max_eigenvalue - max_eigenvalue_np)
vector_error = np.linalg.norm(max_eigenvector - max_eigenvector_np)
print(f"\n特征值误差: {eigenvalue_error:.6f}")
print(f"特征向量误差: {vector_error:.6f}")

# ============================================================================
# 第三部分：简化版SVD分解实现
# ============================================================================
print("\n\n第三部分：简化版SVD分解实现")
print("-" * 50)

def simple_svd(A, k=2):
    """
    简化版SVD分解（通过特征值分解实现）
    仅计算前k个奇异值和向量，用于理解SVD原理
    """
    m, n = A.shape
    
    # 计算 AᵀA 和 AAᵀ
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)
    
    # 计算特征值和特征向量
    eigenvalues_ATA, eigenvectors_ATA = np.linalg.eig(ATA)
    eigenvalues_AAT, eigenvectors_AAT = np.linalg.eig(AAT)
    
    # 按特征值大小排序（取前k个）
    idx_ATA = np.argsort(eigenvalues_ATA)[::-1][:k]
    idx_AAT = np.argsort(eigenvalues_AAT)[::-1][:k]
    
    # 奇异值是特征值的平方根
    singular_values = np.sqrt(np.abs(eigenvalues_ATA[idx_ATA]))
    
    # 构建U, Σ, Vt
    U = eigenvectors_AAT[:, idx_AAT]
    Sigma = np.diag(singular_values)
    Vt = eigenvectors_ATA[:, idx_ATA].T
    
    return U, Sigma, Vt

# 测试矩阵
X = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

print(f"测试矩阵 X (4×3):")
print(X)

# 使用简化版SVD
U_simple, Sigma_simple, Vt_simple = simple_svd(X, k=2)
print(f"\n简化版SVD结果:")
print(f"U (前2列):\n{U_simple}")
print(f"奇异值 Σ:\n{Sigma_simple}")
print(f"Vᵀ (前2行):\n{Vt_simple}")

# 使用NumPy SVD对比
U_np, Sigma_np, Vt_np = np.linalg.svd(X, full_matrices=False)
print(f"\nNumPy SVD结果:")
print(f"U:\n{U_np[:, :2]}")
print(f"奇异值 Σ: {Sigma_np[:2]}")
print(f"Vᵀ:\n{Vt_np[:2, :]}")

# 重构矩阵
X_reconstructed_simple = np.dot(U_simple, np.dot(Sigma_simple, Vt_simple))
X_reconstructed_np = np.dot(U_np[:, :2], np.dot(np.diag(Sigma_np[:2]), Vt_np[:2, :]))

print(f"\n重构误差:")
print(f"简化版SVD重构误差: {np.linalg.norm(X - X_reconstructed_simple):.6f}")
print(f"NumPy SVD重构误差: {np.linalg.norm(X[:, :2] - X_reconstructed_np):.6f}")

# ============================================================================
# 第四部分：性能对比可视化
# ============================================================================
print("\n\n第四部分：性能对比可视化")
print("-" * 50)

# 创建性能对比图表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 矩阵乘法性能对比
ax1 = axes[0, 0]
ax1.plot(matrix_sizes, manual_times, 'o-', label='手动实现', linewidth=2)
ax1.plot(matrix_sizes, numpy_times, 's-', label='NumPy内置', linewidth=2)
ax1.set_xlabel('矩阵大小 (n×n)')
ax1.set_ylabel('时间 (秒)')
ax1.set_title('矩阵乘法性能对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 加速比柱状图
ax2 = axes[0, 1]
speedups = [manual_times[i]/numpy_times[i] for i in range(len(matrix_sizes))]
bars = ax2.bar(range(len(matrix_sizes)), speedups, color=['#d97757', '#6a9bcc', '#788c5d', '#c4a35a'])
ax2.set_xlabel('矩阵大小')
ax2.set_ylabel('加速比 (手动/NumPy)')
ax2.set_title('NumPy向量化加速效果')
ax2.set_xticks(range(len(matrix_sizes)))
ax2.set_xticklabels([f'{s}×{s}' for s in matrix_sizes])
# 在柱子上添加数值
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{speedup:.1f}x', ha='center', va='bottom')

# 3. 幂迭代法收敛过程
ax3 = axes[1, 0]
ax3.plot(history, 'o-', linewidth=2)
ax3.axhline(y=max_eigenvalue_np, color='r', linestyle='--', label=f'NumPy值: {max_eigenvalue_np:.4f}')
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('特征值估计')
ax3.set_title('幂迭代法收敛过程')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 奇异值比较
ax4 = axes[1, 1]
x_pos = np.arange(2)
width = 0.35
ax4.bar(x_pos - width/2, Sigma_simple.diagonal(), width, label='简化版SVD', alpha=0.8)
ax4.bar(x_pos + width/2, Sigma_np[:2], width, label='NumPy SVD', alpha=0.8)
ax4.set_xlabel('奇异值序号')
ax4.set_ylabel('奇异值大小')
ax4.set_title('奇异值计算对比')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['σ₁', 'σ₂'])
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day2/矩阵运算性能对比.png', dpi=300, bbox_inches='tight')
print("性能对比图表已保存到 'outputs/阶段一/Week1/Day2/矩阵运算性能对比.png'")

# ============================================================================
# 第五部分：综合练习题目
# ============================================================================
print("\n\n第五部分：综合练习题目")
print("-" * 50)
print("请尝试完成以下练习题目：")
print("")
print("题目1：矩阵乘法优化")
print("  尝试优化manual_matrix_multiply函数：")
print("  1. 使用局部变量减少属性访问")
print("  2. 改变循环顺序提高缓存命中率")
print("  3. 使用分块（Blocking）技术")
print("  测量优化后的性能提升")
print("")
print("题目2：QR算法实现")
print("  实现QR算法计算所有特征值：")
print("  1. 实现Gram-Schmidt正交化")
print("  2. 迭代进行QR分解：A = QR, A_new = RQ")
print("  3. 验证结果与np.linalg.eig的一致性")
print("")
print("题目3：SVD应用")
print("  使用SVD进行图像压缩：")
print("  1. 加载一张灰度图像（或使用随机矩阵模拟）")
print("  2. 对图像矩阵进行SVD分解")
print("  3. 使用前k个奇异值重构图像")
print("  4. 分析不同k值的压缩效果和信噪比")
print("")
print("题目4：在LLM中的实际应用思考")
print("  1. 矩阵乘法在Transformer的注意力机制中如何被优化？")
print("  2. 特征值分解在模型稳定性分析中有什么作用？")
print("  3. SVD如何用于大模型的知识蒸馏和模型压缩？")

# ============================================================================
# 第六部分：运行指导与总结
# ============================================================================
print("\n\n第六部分：运行指导与总结")
print("-" * 50)
print("运行指导：")
print("  1. 确保已安装NumPy和Matplotlib：pip install numpy matplotlib")
print("  2. 直接运行本文件：python 矩阵运算实战练习.py")
print("  3. 建议在Jupyter Notebook中分块运行，便于调试和观察")
print("")
print("关键收获：")
print("  1. 理解矩阵乘法的算法复杂度（O(n³)）")
print("  2. 认识NumPy向量化运算的巨大性能优势")
print("  3. 掌握特征值计算的基本迭代方法")
print("  4. 理解SVD分解的数学原理和实现方式")
print("")
print("下一步学习建议：")
print("  1. 深入学习NumPy的广播机制和高级索引")
print("  2. 学习SciPy中的稀疏矩阵运算")
print("  3. 了解GPU加速的矩阵运算（CuPy）")

print("\n" + "=" * 60)
print("Day 2练习完成！建议将代码复制到Jupyter Notebook中逐步运行")
print("深入理解每个算法背后的数学原理和性能优化思路")
print("=" * 60)

# 显示图表
plt.show()

print("\n提示：如果要保存可视化结果，请取消 plt.show() 的注释")
print("或将最后一行改为：")
print("plt.savefig('矩阵运算性能对比.png', dpi=300, bbox_inches='tight')")
print("plt.close()")