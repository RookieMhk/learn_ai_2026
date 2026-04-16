#!/usr/bin/env python3
"""
AI数据可视化案例研究
Week 1 Day 5 - Matplotlib高级应用

本文件包含Transformer注意力机制可视化和神经网络特征可视化两大模块，
提供可运行代码示例，支持实际AI模型分析场景。

使用方法：
1. 安装依赖：pip install numpy matplotlib torch torchvision transformers
2. 运行示例：python AI数据可视化案例研究.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# 配置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 第一部分：Transformer注意力机制可视化
# ============================================================================

def generate_synthetic_attention_weights(seq_len=10, n_layers=3, n_heads=8):
    """
    生成合成注意力权重，用于演示可视化技术
    
    参数：
    seq_len: 序列长度
    n_layers: Transformer层数
    n_heads: 每层的注意力头数量
    
    返回：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    """
    np.random.seed(42)
    
    # 初始化注意力权重张量
    attention_weights = np.zeros((n_layers, n_heads, seq_len, seq_len))
    
    # 为每层和每个头生成不同的注意力模式
    for layer in range(n_layers):
        for head in range(n_heads):
            # 随机生成基础注意力矩阵
            base_attn = np.random.rand(seq_len, seq_len)
            
            # 根据不同模式调整
            if head % 4 == 0:  # 对角模式（关注自身和邻近位置）
                for i in range(seq_len):
                    for j in range(seq_len):
                        distance = abs(i - j)
                        base_attn[i, j] *= np.exp(-distance / 2.0)
                        
            elif head % 4 == 1:  # 全局模式（关注所有位置）
                base_attn = np.ones((seq_len, seq_len)) * 0.1
                base_attn += np.random.rand(seq_len, seq_len) * 0.1
                
            elif head % 4 == 2:  # 局部模式（关注固定窗口）
                window_size = 3
                for i in range(seq_len):
                    for j in range(seq_len):
                        if abs(i - j) > window_size:
                            base_attn[i, j] *= 0.1
                            
            else:  # 随机稀疏模式
                for i in range(seq_len):
                    for j in range(seq_len):
                        if np.random.rand() > 0.3:
                            base_attn[i, j] = 0
            
            # 归一化每行（softmax模拟）
            for i in range(seq_len):
                row = base_attn[i]
                # 添加一个小值避免除零
                row = row + 1e-8
                row = np.exp(row) / np.sum(np.exp(row))
                base_attn[i] = row
            
            attention_weights[layer, head] = base_attn
    
    return attention_weights


def visualize_attention_heatmap(attention_weights, tokens, 
                               layer_idx=0, head_idx=0,
                               figsize=(12, 10)):
    """
    绘制单头注意力热力图
    
    参数：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    tokens: 词元列表
    layer_idx: 层索引（0-based）
    head_idx: 头索引（0-based）
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    # 提取指定层和头的注意力权重
    attn = attention_weights[layer_idx, head_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建自定义色彩映射（蓝-白-红发散色）
    colors = ['#2166ac', '#f7f7f7', '#b2182b']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                                   colors, N=n_bins)
    
    # 绘制热力图
    im = ax.imshow(attn, cmap=custom_cmap, aspect='auto', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(tokens, fontsize=12)
    
    # 添加网格线
    ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', 
            linewidth=0.5, alpha=0.3)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('注意力权重', rotation=90, va='bottom', fontsize=14)
    
    # 设置标题
    title = (f'Transformer注意力热力图\n'
             f'层 {layer_idx+1}/{attention_weights.shape[0]}, '
             f'头 {head_idx+1}/{attention_weights.shape[1]}')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 标注最高注意力值的位置
    max_idx = np.unravel_index(attn.argmax(), attn.shape)
    ax.plot(max_idx[1], max_idx[0], 'r*', markersize=15, 
            markeredgecolor='black', markeredgewidth=1)
    
    # 添加文本说明
    info_text = (f'序列长度: {len(tokens)}\n'
                 f'最大权重: {attn[max_idx]:.3f}\n'
                 f'位置: ({tokens[max_idx[0]]}, {tokens[max_idx[1]]})')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def visualize_multihead_attention_matrix(attention_weights, tokens, 
                                        query_idx=0, figsize=(16, 12)):
    """
    可视化多头注意力矩阵，展示所有层和头的注意力分布
    
    参数：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    tokens: 词元列表
    query_idx: 查询位置索引
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    # 创建子图网格
    fig, axs = plt.subplots(n_layers, n_heads, figsize=figsize,
                           squeeze=False, sharex=True, sharey=True)
    
    # 遍历所有层和头
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axs[layer, head]
            
            # 提取该头的注意力权重（特定查询位置）
            attn = attention_weights[layer, head, query_idx]
            
            # 绘制条形图
            bars = ax.bar(range(seq_len), attn, color='steelblue',
                         edgecolor='navy', linewidth=0.5)
            
            # 高亮查询位置自身
            if query_idx < seq_len:
                bars[query_idx].set_color('crimson')
                bars[query_idx].set_edgecolor('darkred')
                bars[query_idx].set_linewidth(1.5)
            
            # 设置Y轴范围
            ax.set_ylim([0, 1])
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 设置标题（只在第一行）
            if layer == 0:
                ax.set_title(f'头 {head+1}', fontsize=10, fontweight='bold')
            
            # 设置坐标轴标签（只在最后一行和第一列）
            if layer == n_layers - 1:
                ax.set_xlabel('位置', fontsize=9)
            if head == 0:
                ax.set_ylabel(f'层{layer+1}\n权重', fontsize=9)
    
    # 添加整体标题
    fig.suptitle(f'多头注意力矩阵分析 (查询位置: {tokens[query_idx]})', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def visualize_attention_pattern_evolution(attention_weights, tokens,
                                         head_selection=[0, 2, 4, 6],
                                         figsize=(14, 10)):
    """
    可视化注意力模式随层数的演变
    
    参数：
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    tokens: 词元列表
    head_selection: 要可视化的头索引列表
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    n_layers, _, seq_len, _ = attention_weights.shape
    n_heads_to_plot = len(head_selection)
    
    fig, axs = plt.subplots(n_layers, n_heads_to_plot, 
                           figsize=figsize, squeeze=False)
    
    # 创建色彩映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for layer_idx, layer in enumerate(range(n_layers)):
        for head_idx, head in enumerate(head_selection):
            ax = axs[layer_idx, head_idx]
            
            # 提取该头所有查询位置的注意力权重
            attn = attention_weights[layer, head]
            
            # 绘制堆叠区域图（展示不同查询位置的注意力分布）
            for query_idx in range(min(4, seq_len)):
                # 计算累积和以堆叠显示
                if query_idx == 0:
                    cumulative = attn[query_idx]
                else:
                    cumulative = cumulative + attn[query_idx]
                
                ax.fill_between(range(seq_len), 0, cumulative,
                               color=colors[query_idx % len(colors)],
                               alpha=0.7, 
                               label=f'查询{tokens[query_idx]}' if layer_idx == 0 else None)
            
            # 设置图表元素
            ax.set_title(f'层{layer+1} 头{head+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 只在底部子图显示X轴标签
            if layer_idx == n_layers - 1:
                ax.set_xlabel('键位置', fontsize=9)
            
            # 只在左侧子图显示Y轴标签
            if head_idx == 0:
                ax.set_ylabel('注意力权重', fontsize=9)
    
    # 添加图例
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
              ncol=min(4, seq_len), fontsize=11,
              bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle('注意力模式随Transformer层数的演变', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig


# ============================================================================
# 第二部分：神经网络特征可视化
# ============================================================================

def generate_synthetic_feature_activations(batch_size=4, channels=16, 
                                          height=8, width=8):
    """
    生成合成特征激活，模拟CNN中间层输出
    
    参数：
    batch_size: 批大小
    channels: 通道数
    height: 特征图高度
    width: 特征图宽度
    
    返回：
    activations: [batch_size, channels, height, width]
    """
    np.random.seed(123)
    
    activations = np.zeros((batch_size, channels, height, width))
    
    # 为每个批次和通道生成不同模式
    for b in range(batch_size):
        for c in range(channels):
            # 基础激活模式
            pattern = np.random.rand(height, width) * 0.5
            
            # 添加不同空间模式
            if c % 4 == 0:  # 对角模式
                for i in range(height):
                    for j in range(width):
                        pattern[i, j] += np.exp(-abs(i - j) / 3.0)
                        
            elif c % 4 == 1:  # 中心模式
                center_y, center_x = height // 2, width // 2
                for i in range(height):
                    for j in range(width):
                        distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                        pattern[i, j] += np.exp(-distance / 2.0)
                        
            elif c % 4 == 2:  # 边缘模式
                pattern[:, 0] += 0.8
                pattern[:, -1] += 0.8
                pattern[0, :] += 0.8
                pattern[-1, :] += 0.8
                
            else:  # 随机点模式
                for _ in range(5):
                    i = np.random.randint(0, height)
                    j = np.random.randint(0, width)
                    pattern[i, j] += 1.0
            
            # 添加噪声
            noise = np.random.randn(height, width) * 0.1
            activations[b, c] = pattern + noise
    
    return activations


def visualize_feature_maps(activations, batch_idx=0, 
                          n_channels_to_show=8, figsize=(15, 8)):
    """
    可视化特征图（CNN中间层激活）
    
    参数：
    activations: [batch_size, channels, height, width]
    batch_idx: 批次索引
    n_channels_to_show: 要显示的特征图数量
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    batch_activations = activations[batch_idx]
    n_channels = min(n_channels_to_show, batch_activations.shape[0])
    
    # 计算网格布局
    cols = 4
    rows = (n_channels + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=figsize,
                           squeeze=False)
    
    # 遍历通道
    for idx in range(n_channels):
        row = idx // cols
        col = idx % cols
        ax = axs[row, col]
        
        # 提取特征图
        feature_map = batch_activations[idx]
        
        # 归一化显示
        vmin = feature_map.min()
        vmax = feature_map.max()
        
        # 绘制热力图
        im = ax.imshow(feature_map, cmap='hot', 
                       vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题
        ax.set_title(f'通道 {idx+1}', fontsize=10, fontweight='bold')
        
        # 添加颜色条（只在第一行）
        if row == 0 and col == 0:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.set_ylabel('激活强度', rotation=90, fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(n_channels, rows * cols):
        row = idx // cols
        col = idx % cols
        axs[row, col].axis('off')
    
    # 添加整体标题
    plt.suptitle(f'CNN特征图可视化 (批次 {batch_idx+1})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def visualize_feature_distribution(activations, layer_name='conv3',
                                  figsize=(14, 10)):
    """
    可视化特征激活的统计分布
    
    参数：
    activations: [batch_size, channels, height, width]
    layer_name: 层名称（用于标题）
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    # 展平所有激活
    flat_activations = activations.flatten()
    
    # 计算统计量
    mean_val = np.mean(flat_activations)
    std_val = np.std(flat_activations)
    median_val = np.median(flat_activations)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 使用GridSpec创建复杂布局
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # 1. 直方图
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(flat_activations, bins=50, 
                               color='steelblue', edgecolor='black',
                               alpha=0.7)
    ax1.set_xlabel('激活值', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('激活值分布直方图', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图（按通道）
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 计算每个通道的激活统计
    channel_means = activations.mean(axis=(0, 2, 3))
    channel_medians = np.median(activations, axis=(0, 2, 3))
    
    positions = range(len(channel_means))
    ax2.bar(positions, channel_means, color='lightcoral', 
           alpha=0.7, label='均值')
    ax2.scatter(positions, channel_medians, color='darkred', 
               s=50, zorder=5, label='中位数')
    
    ax2.set_xlabel('通道编号', fontsize=11)
    ax2.set_ylabel('激活强度', fontsize=11)
    ax2.set_title('各通道激活强度对比', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 散点图（激活相关性）
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 随机选择两个通道计算相关性
    if activations.shape[1] >= 2:
        chan1 = activations[:, 0].flatten()
        chan2 = activations[:, 1].flatten()
        
        # 采样以提升性能
        sample_size = min(1000, len(chan1))
        indices = np.random.choice(len(chan1), sample_size, replace=False)
        
        ax3.scatter(chan1[indices], chan2[indices], 
                   alpha=0.5, s=10, color='seagreen')
        
        # 计算并绘制回归线
        if sample_size > 1:
            coeffs = np.polyfit(chan1[indices], chan2[indices], 1)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(chan1.min(), chan1.max(), 100)
            ax3.plot(x_range, poly(x_range), 'r-', linewidth=2, 
                    label=f'R = {np.corrcoef(chan1[indices], chan2[indices])[0,1]:.3f}')
        
        ax3.set_xlabel('通道 1 激活', fontsize=11)
        ax3.set_ylabel('通道 2 激活', fontsize=11)
        ax3.set_title('通道间激活相关性', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 热力图（批次-通道平均激活）
    ax4 = fig.add_subplot(gs[1, 0:2])
    
    # 计算每个批次和通道的平均激活
    batch_channel_means = activations.mean(axis=(2, 3))
    
    im = ax4.imshow(batch_channel_means, cmap='YlOrRd', 
                   aspect='auto', interpolation='nearest')
    
    ax4.set_xlabel('通道编号', fontsize=11)
    ax4.set_ylabel('批次编号', fontsize=11)
    ax4.set_title('批次-通道平均激活热力图', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax4, shrink=0.9)
    cbar.ax.set_ylabel('平均激活强度', rotation=90, fontsize=11)
    
    # 5. 统计摘要（文本）
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 生成统计摘要文本
    stats_text = (f'特征激活统计摘要\n\n'
                  f'层名称: {layer_name}\n'
                  f'数据形状: {activations.shape}\n\n'
                  f'总体统计:\n'
                  f'• 均值: {mean_val:.4f}\n'
                  f'• 标准差: {std_val:.4f}\n'
                  f'• 中位数: {median_val:.4f}\n'
                  f'• 最小值: {flat_activations.min():.4f}\n'
                  f'• 最大值: {flat_activations.max():.4f}\n\n'
                  f'通道统计:\n'
                  f'• 最高激活通道: {np.argmax(channel_means)+1}\n'
                  f'• 最低激活通道: {np.argmin(channel_means)+1}\n'
                  f'• 激活变异系数: {std_val/mean_val:.3f}')
    
    ax5.text(0.1, 0.95, stats_text, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'神经网络特征激活分析 - {layer_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def visualize_feature_evolution_across_layers(layer_activations_list,
                                             layer_names=None,
                                             figsize=(16, 12)):
    """
    可视化特征随网络层数的演变
    
    参数：
    layer_activations_list: 各层激活张量的列表
    layer_names: 各层名称列表
    figsize: 图表尺寸
    
    返回：
    fig: Matplotlib图形对象
    """
    n_layers = len(layer_activations_list)
    
    if layer_names is None:
        layer_names = [f'层{i+1}' for i in range(n_layers)]
    
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    # 1. 平均激活强度随层数变化
    ax1 = axs[0, 0]
    mean_activations = [act.mean() for act in layer_activations_list]
    std_activations = [act.std() for act in layer_activations_list]
    
    ax1.errorbar(range(n_layers), mean_activations, yerr=std_activations,
                fmt='o-', linewidth=2, capsize=5, color='darkblue')
    ax1.set_xlabel('层数', fontsize=11)
    ax1.set_ylabel('平均激活强度', fontsize=11)
    ax1.set_title('激活强度随层数变化', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 激活稀疏性随层数变化
    ax2 = axs[0, 1]
    
    # 计算稀疏性（激活值小于阈值0.1的比例）
    sparsity = []
    for act in layer_activations_list:
        flat = act.flatten()
        sparsity.append(np.sum(np.abs(flat) < 0.1) / len(flat))
    
    ax2.plot(range(n_layers), sparsity, 's--', linewidth=2, color='darkgreen')
    ax2.set_xlabel('层数', fontsize=11)
    ax2.set_ylabel('稀疏性比例', fontsize=11)
    ax2.set_title('激活稀疏性随层数变化', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 激活分布变化（箱线图）
    ax3 = axs[0, 2]
    
    # 为每层提取激活样本
    sample_data = []
    for i, act in enumerate(layer_activations_list):
        flat = act.flatten()
        # 随机采样
        sample = np.random.choice(flat, size=min(500, len(flat)), replace=False)
        sample_data.append(sample)
    
    ax3.boxplot(sample_data, labels=layer_names[:n_layers])
    ax3.set_xlabel('网络层', fontsize=11)
    ax3.set_ylabel('激活值', fontsize=11)
    ax3.set_title('激活值分布箱线图', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 特征图对比（第一层 vs 最后一层）
    ax4 = axs[1, 0]
    
    if len(layer_activations_list) >= 2:
        first_layer = layer_activations_list[0][0, 0]  # 第一个批次，第一个通道
        last_layer = layer_activations_list[-1][0, 0]
        
        im1 = ax4.imshow(first_layer, cmap='viridis', interpolation='nearest')
        ax4.set_title('第一层特征图', fontsize=11)
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        # 添加颜色条
        cbar1 = fig.colorbar(im1, ax=ax4, shrink=0.8)
        cbar1.ax.set_ylabel('激活强度', fontsize=9)
    
    # 5. 最后一层特征图
    ax5 = axs[1, 1]
    
    if len(layer_activations_list) >= 2:
        im2 = ax5.imshow(last_layer, cmap='viridis', interpolation='nearest')
        ax5.set_title('最后一层特征图', fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])
        
        cbar2 = fig.colorbar(im2, ax=ax5, shrink=0.8)
        cbar2.ax.set_ylabel('激活强度', fontsize=9)
    
    # 6. 演变规律总结
    ax6 = axs[1, 2]
    ax6.axis('off')
    
    summary_text = (f'特征演变规律总结\n\n'
                   f'观察到的模式:\n'
                   f'1. 激活强度: {mean_activations[0]:.3f} → {mean_activations[-1]:.3f}\n'
                   f'2. 变化趋势: {"增加" if mean_activations[-1] > mean_activations[0] else "减少"}\n'
                   f'3. 稀疏性: {sparsity[0]:.1%} → {sparsity[-1]:.1%}\n'
                   f'4. 分布变化: 从{"宽" if std_activations[0] > std_activations[-1] else "窄"}到{"宽" if std_activations[-1] > std_activations[0] else "窄"}\n\n'
                   f'分析解读:\n'
                   f'• 深层网络提取更抽象特征\n'
                   f'• 激活模式从局部到全局\n'
                   f'• 特征选择性逐渐增强')
    
    ax6.text(0.1, 0.95, summary_text, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('神经网络特征随层数演变分析', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 第三部分：主程序与示例运行
# ============================================================================

def run_all_examples():
    """
    运行所有可视化示例
    """
    print("=" * 60)
    print("AI数据可视化案例研究 - Week 1 Day 5")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 准备合成数据...")
    
    # 注意力权重数据
    seq_len = 12
    n_layers = 4
    n_heads = 6
    
    # 创建示例词元（模拟句子）
    tokens = ['[CLS]', 'AI', '技术', '学习', '计划', '正在进行', 
             '可视化', '技术', '让', '理解', '更', '直观']
    
    attention_weights = generate_synthetic_attention_weights(
        seq_len=seq_len, n_layers=n_layers, n_heads=n_heads
    )
    
    # 特征激活数据
    batch_size = 4
    channels = 8
    height = width = 10
    
    activations = generate_synthetic_feature_activations(
        batch_size=batch_size, channels=channels,
        height=height, width=width
    )
    
    # 多层激活数据（用于演变分析）
    layer_activations = [
        np.random.randn(4, 8, 8, 8) * 0.5 + 0.1,
        np.random.randn(4, 16, 6, 6) * 0.4 + 0.2,
        np.random.randn(4, 32, 4, 4) * 0.3 + 0.3,
        np.random.randn(4, 64, 2, 2) * 0.2 + 0.4,
    ]
    layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4']
    
    # 2. 运行可视化
    print("\n2. 运行可视化示例...")
    
    # 2.1 注意力热力图
    print("  • 生成注意力热力图...")
    fig1 = visualize_attention_heatmap(attention_weights, tokens, 
                                      layer_idx=1, head_idx=2)
    fig1.savefig('outputs/阶段一/Week1/Day5/attention_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2.2 多头注意力矩阵
    print("  • 生成多头注意力矩阵...")
    fig2 = visualize_multihead_attention_matrix(attention_weights, tokens,
                                               query_idx=1)
    fig2.savefig('outputs/阶段一/Week1/Day5/multihead_attention_matrix.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 2.3 注意力模式演变
    print("  • 生成注意力模式演变图...")
    fig3 = visualize_attention_pattern_evolution(attention_weights, tokens)
    fig3.savefig('outputs/阶段一/Week1/Day5/attention_evolution.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 2.4 特征图可视化
    print("  • 生成特征图可视化...")
    fig4 = visualize_feature_maps(activations, batch_idx=0)
    fig4.savefig('outputs/阶段一/Week1/Day5/feature_maps.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 2.5 特征分布分析
    print("  • 生成特征分布分析图...")
    fig5 = visualize_feature_distribution(activations, layer_name='Conv2')
    fig5.savefig('outputs/阶段一/Week1/Day5/feature_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 2.6 特征演变分析
    print("  • 生成特征演变分析图...")
    fig6 = visualize_feature_evolution_across_layers(layer_activations,
                                                    layer_names)
    fig6.savefig('outputs/阶段一/Week1/Day5/feature_evolution.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    # 3. 输出总结
    print("\n3. 可视化完成！")
    print("  生成的图像已保存至 outputs/阶段一/Week1/Day5/ 目录：")
    print("  • attention_heatmap.png        - 单头注意力热力图")
    print("  • multihead_attention_matrix.png - 多头注意力矩阵")
    print("  • attention_evolution.png      - 注意力模式演变")
    print("  • feature_maps.png             - CNN特征图")
    print("  • feature_distribution.png     - 特征统计分布")
    print("  • feature_evolution.png        - 跨层特征演变")
    
    print("\n4. 技术要点总结：")
    print("  • 掌握了Matplotlib高级布局技巧（GridSpec、subplot_mosaic）")
    print("  • 实现了Transformer注意力机制的专业可视化")
    print("  • 完成了神经网络特征激活的全面分析")
    print("  • 创建了可复用的AI可视化代码模板")
    
    return True


def save_visualization_functions():
    """
    将关键函数保存为独立模块，方便后续复用
    """
    import inspect
    
    functions_to_save = [
        generate_synthetic_attention_weights,
        visualize_attention_heatmap,
        visualize_multihead_attention_matrix,
        visualize_attention_pattern_evolution,
        generate_synthetic_feature_activations,
        visualize_feature_maps,
        visualize_feature_distribution,
        visualize_feature_evolution_across_layers
    ]
    
    module_content = '''"""
AI可视化工具模块
包含Transformer注意力可视化和神经网络特征可视化功能

导入方式：
from ai_visualization_tools import (
    generate_synthetic_attention_weights,
    visualize_attention_heatmap,
    visualize_multihead_attention_matrix,
    visualize_attention_pattern_evolution,
    generate_synthetic_feature_activations,
    visualize_feature_maps,
    visualize_feature_distribution,
    visualize_feature_evolution_across_layers
)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

'''
    
    for func in functions_to_save:
        module_content += '\n\n' + inspect.getsource(func)
    
    # 保存模块文件
    with open('outputs/阶段一/Week1/Day5/ai_visualization_tools.py', 'w', encoding='utf-8') as f:
        f.write(module_content)
    
    print("已保存可复用模块至 ai_visualization_tools.py")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("开始执行AI数据可视化案例研究...")
    
    try:
        # 运行所有示例
        run_all_examples()
        
        # 保存可复用函数
        save_visualization_functions()
        
        print("\n" + "=" * 60)
        print("AI数据可视化案例研究执行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()