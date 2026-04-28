#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Day 2: 矩阵运算可视化代码
AI学习计划 - 第一阶段：基础夯实 - 线性代数核心

本文件提供矩阵运算相关的可视化实现，包括：
1. 特征向量方向可视化
2. 矩阵乘法过程示意图
3. 奇异值分解几何解释
4. 线性变换可视化

通过图形直观理解线性代数概念的几何意义。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Week 1 Day 2: 矩阵运算可视化代码")
print("=" * 60)

# ============================================================================
# 可视化1：特征向量方向可视化
# ============================================================================
def visualize_eigenvectors(A, title="特征向量方向可视化"):
    """
    可视化矩阵的特征向量方向及其变换效果
    
    参数：
        A: 2×2矩阵
        title: 图表标题
    """
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 创建标准单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    
    # 应用矩阵变换
    transformed_circle = A @ unit_circle
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：特征向量方向
    ax1 = axes[0]
    # 绘制单位圆
    ax1.plot(unit_circle[0, :], unit_circle[1, :], 'b-', alpha=0.5, label='单位圆')
    ax1.fill(unit_circle[0, :], unit_circle[1, :], 'b', alpha=0.1)
    
    # 绘制特征向量方向
    colors = ['r', 'g']
    for i in range(len(eigenvalues)):
        eigvec = eigenvectors[:, i]
        eigval = eigenvalues[i]
        
        # 归一化特征向量用于显示
        scale = 1.5
        ax1.arrow(0, 0, scale*eigvec[0], scale*eigvec[1], 
                 head_width=0.1, head_length=0.15, 
                 fc=colors[i], ec=colors[i], linewidth=3,
                 label=f'v{i+1} (λ={eigval:.2f})')
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('特征向量方向')
    ax1.axhline(y=0, color='k', alpha=0.3)
    ax1.axvline(x=0, color='k', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 子图2：矩阵变换效果
    ax2 = axes[1]
    # 绘制变换后的椭圆
    ax2.plot(transformed_circle[0, :], transformed_circle[1, :], 'r-', alpha=0.7, label='变换后椭圆')
    ax2.fill(transformed_circle[0, :], transformed_circle[1, :], 'r', alpha=0.1)
    
    # 绘制变换后的特征向量方向
    for i in range(len(eigenvalues)):
        eigvec = eigenvectors[:, i]
        eigval = eigenvalues[i]
        transformed_eigvec = A @ eigvec
        
        ax2.arrow(0, 0, transformed_eigvec[0], transformed_eigvec[1],
                 head_width=0.1, head_length=0.15,
                 fc=colors[i], ec=colors[i], linewidth=3,
                 label=f'A·v{i+1} (拉伸{abs(eigval):.2f}倍)')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('矩阵变换效果')
    ax2.axhline(y=0, color='k', alpha=0.3)
    ax2.axvline(x=0, color='k', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/阶段一/Week1/Day2/特征向量可视化.png', dpi=300, bbox_inches='tight')
    print("特征向量可视化已保存")
    plt.show()

# ============================================================================
# 可视化2：矩阵乘法过程示意图
# ============================================================================
def visualize_matrix_multiplication(A, B, C=None):
    """
    可视化矩阵乘法 A × B 的过程
    
    参数：
        A: m×n矩阵
        B: n×p矩阵
        C: 计算结果矩阵（可选）
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "矩阵维度不匹配"
    
    if C is None:
        C = np.matmul(A, B)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1：矩阵A
    ax1 = plt.subplot(131)
    ax1.set_title(f'矩阵 A ({m}×{n})', fontsize=12)
    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(-0.5, m-0.5)
    ax1.invert_yaxis()
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(m))
    ax1.grid(True, alpha=0.3)
    
    # 绘制矩阵A的单元格
    for i in range(m):
        for j in range(n):
            val = A[i, j]
            color = 'lightblue' if val >= 0 else 'lightcoral'
            rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                            facecolor=color, edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10)
    
    # 子图2：矩阵B
    ax2 = plt.subplot(132)
    ax2.set_title(f'矩阵 B ({n}×{p})', fontsize=12)
    ax2.set_xlim(-0.5, p-0.5)
    ax2.set_ylim(-0.5, n-0.5)
    ax2.invert_yaxis()
    ax2.set_xticks(range(p))
    ax2.set_yticks(range(n))
    ax2.grid(True, alpha=0.3)
    
    # 绘制矩阵B的单元格
    for i in range(n):
        for j in range(p):
            val = B[i, j]
            color = 'lightgreen' if val >= 0 else 'lightyellow'
            rect = Rectangle((j-0.5, i-0.5), 1, 1,
                            facecolor=color, edgecolor='black', alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10)
    
    # 子图3：结果矩阵C
    ax3 = plt.subplot(133)
    ax3.set_title(f'结果矩阵 C = A×B ({m}×{p})', fontsize=12)
    ax3.set_xlim(-0.5, p-0.5)
    ax3.set_ylim(-0.5, m-0.5)
    ax3.invert_yaxis()
    ax3.set_xticks(range(p))
    ax3.set_yticks(range(m))
    ax3.grid(True, alpha=0.3)
    
    # 绘制矩阵C的单元格
    for i in range(m):
        for j in range(p):
            val = C[i, j]
            # 颜色基于值的大小
            norm_val = (val - C.min()) / (C.max() - C.min() + 1e-8)
            color = plt.cm.RdYlBu(norm_val)
            rect = Rectangle((j-0.5, i-0.5), 1, 1,
                            facecolor=color, edgecolor='black', alpha=0.8)
            ax3.add_patch(rect)
            ax3.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10,
                    color='white' if norm_val > 0.7 else 'black')
    
    # 添加计算过程的文字说明
    plt.figtext(0.5, 0.02, 
                f'计算过程：C[i,j] = Σₖ A[i,k] × B[k,j] (k=0..{n-1})',
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.savefig('outputs/阶段一/Week1/Day2/矩阵乘法示意图.png', dpi=300, bbox_inches='tight')
    print("矩阵乘法示意图已保存")
    plt.show()

# ============================================================================
# 可视化3：奇异值分解几何解释
# ============================================================================
def visualize_svd_geometry(A, title="奇异值分解几何解释"):
    """
    可视化SVD分解的几何意义：旋转→缩放→旋转
    
    参数：
        A: 2×2矩阵
        title: 图表标题
    """
    # 计算SVD
    U, Sigma, Vt = np.linalg.svd(A)
    V = Vt.T
    
    # 创建标准单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    
    # 逐步变换
    # 1. 第一步旋转：Vᵀ
    rotated1 = V.T @ unit_circle
    # 2. 缩放：Σ
    scaled = np.diag(Sigma) @ rotated1
    # 3. 第二步旋转：U
    rotated2 = U @ scaled
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    titles = ['1. 原始单位圆', '2. 旋转 Vᵀ', '3. 缩放 Σ', '4. 旋转 U']
    data = [unit_circle, rotated1, scaled, rotated2]
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, ax in enumerate(axes.flat):
        circle_data = data[idx]
        ax.plot(circle_data[0, :], circle_data[1, :], '-', color=colors[idx], alpha=0.7, linewidth=2)
        ax.fill(circle_data[0, :], circle_data[1, :], colors[idx], alpha=0.1)
        
        # 绘制坐标轴方向
        ax.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
        ax.arrow(0, 0, 0, 1, head_width=0.05, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
        
        # 绘制奇异向量方向
        if idx == 0:
            # 原始坐标轴
            pass
        elif idx == 1:
            # V的列向量方向
            for i in range(2):
                vec = V[:, i]
                ax.arrow(0, 0, vec[0], vec[1], 
                        head_width=0.05, head_length=0.1,
                        fc='red', ec='red', linewidth=2,
                        label=f'v{i+1}')
        elif idx == 2:
            # 缩放后的轴
            for i in range(2):
                scale = Sigma[i]
                vec = np.array([scale if j == i else 0 for j in range(2)])
                ax.arrow(0, 0, vec[0], vec[1],
                        head_width=0.05, head_length=0.1,
                        fc='purple', ec='purple', linewidth=2,
                        label=f'σ{i+1}={scale:.2f}')
        elif idx == 3:
            # U的列向量方向
            for i in range(2):
                vec = U[:, i]
                ax.arrow(0, 0, vec[0], vec[1],
                        head_width=0.05, head_length=0.1,
                        fc='darkblue', ec='darkblue', linewidth=2,
                        label=f'u{i+1}')
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(titles[idx], fontsize=12)
        ax.axhline(y=0, color='k', alpha=0.3)
        ax.axvline(x=0, color='k', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if idx in [1, 3]:
            ax.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/阶段一/Week1/Day2/SVD几何解释.png', dpi=300, bbox_inches='tight')
    print("SVD几何解释可视化已保存")
    plt.show()

# ============================================================================
# 可视化4：线性变换动画
# ============================================================================
def create_linear_transform_animation(A, filename="linear_transform.gif"):
    """
    创建线性变换的动画，展示向量如何被矩阵变换
    
    参数：
        A: 2×2变换矩阵
        filename: 输出文件名
    """
    # 创建初始向量集合
    vectors = np.array([[1, 0], [0, 1], [0.5, 0.5], [-0.5, 0.3], [0.3, -0.7]])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 设置坐标轴
    for ax in [ax1, ax2]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(y=0, color='k', alpha=0.3)
        ax.axvline(x=0, color='k', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    ax1.set_title('原始向量')
    ax2.set_title('变换过程')
    
    # 绘制原始向量
    arrows1 = []
    for vec, color in zip(vectors, colors):
        arrow = ax1.arrow(0, 0, vec[0], vec[1], 
                         head_width=0.1, head_length=0.15,
                         fc=color, ec=color, linewidth=2,
                         alpha=0.7)
        arrows1.append(arrow)
    
    # 动画函数
    def update(frame):
        # 清除第二幅图的箭头
        for artist in ax2.collections + ax2.patches:
            artist.remove()
        
        t = frame / 30  # 30帧完成变换
        # 插值矩阵：I + t*(A-I)
        interp_matrix = np.eye(2) + t * (A - np.eye(2))
        
        transformed_vectors = interp_matrix @ vectors.T
        
        # 绘制变换后的向量
        for i, (vec, color) in enumerate(zip(transformed_vectors.T, colors)):
            ax2.arrow(0, 0, vec[0], vec[1],
                     head_width=0.1, head_length=0.15,
                     fc=color, ec=color, linewidth=2,
                     alpha=0.7)
            
            # 绘制轨迹点
            if frame > 0:
                prev_t = (frame-1) / 30
                prev_matrix = np.eye(2) + prev_t * (A - np.eye(2))
                prev_vec = prev_matrix @ vectors[i]
                ax2.plot([prev_vec[0], vec[0]], [prev_vec[1], vec[1]],
                        color=color, alpha=0.3, linewidth=1)
        
        ax2.set_title(f'线性变换过程 t={t:.2f}')
        return []
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=31, interval=50, blit=False)
    
    # 保存动画（需要安装pillow）
    try:
        anim.save(f'outputs/阶段一/Week1/Day2/{filename}', writer='pillow', fps=10)
        print(f"线性变换动画已保存为 {filename}")
    except ImportError:
        print("警告：未安装pillow，无法保存GIF动画")
        print("请运行: pip install pillow")
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 主函数：运行所有可视化
# ============================================================================
def main():
    print("开始生成矩阵运算可视化...")
    
    # 示例矩阵
    A_example = np.array([[2, 0.5], [0.5, 1]])
    B_example = np.array([[1, 2, 3], [4, 5, 6]])
    
    # 1. 特征向量可视化
    print("\n1. 生成特征向量方向可视化...")
    visualize_eigenvectors(A_example)
    
    # 2. 矩阵乘法示意图
    print("\n2. 生成矩阵乘法示意图...")
    A_mul = np.array([[1, 2], [3, 4], [5, 6]])
    B_mul = np.array([[2, 1, 0], [3, 4, 5]])
    visualize_matrix_multiplication(A_mul, B_mul)
    
    # 3. SVD几何解释
    print("\n3. 生成SVD几何解释...")
    visualize_svd_geometry(A_example)
    
    # 4. 线性变换动画（可选）
    print("\n4. 生成线性变换动画（可选）...")
    create_linear_transform_animation(A_example)
    
    print("\n" + "=" * 60)
    print("所有可视化代码已准备就绪！")
    print("=" * 60)
    print("\n使用说明：")
    print("1. 直接运行本文件：python 可视化代码.py")
    print("2. 在Jupyter Notebook中分块运行")
    print("3. 自定义矩阵参数观察不同变换效果")
    print("4. 生成的图片保存在 outputs/阶段一/Week1/Day2/ 目录")

if __name__ == "__main__":
    main()