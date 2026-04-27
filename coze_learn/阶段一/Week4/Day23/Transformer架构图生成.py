#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer架构图生成器
绘制标准的编码器-解码器结构，包含多头注意力、前馈网络等组件
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def draw_transformer_architecture():
    """绘制Transformer架构图"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(8, 11.5, 'Transformer架构图', fontsize=24, ha='center', weight='bold')
    ax.text(8, 11, '编码器-解码器结构 | 多头自注意力机制', fontsize=16, ha='center', alpha=0.7)
    
    # ========================================================================
    # 1. 整体框架
    # ========================================================================
    
    # 编码器区域
    encoder_rect = patches.Rectangle((1, 1), 6, 9, linewidth=2, 
                                     edgecolor='#2E86AB', facecolor='#F0F8FF', alpha=0.8)
    ax.add_patch(encoder_rect)
    ax.text(4, 10.2, '编码器 (Encoder)', fontsize=18, ha='center', weight='bold', color='#2E86AB')
    
    # 解码器区域
    decoder_rect = patches.Rectangle((9, 1), 6, 9, linewidth=2, 
                                     edgecolor='#A23B72', facecolor='#FFF0F5', alpha=0.8)
    ax.add_patch(decoder_rect)
    ax.text(12, 10.2, '解码器 (Decoder)', fontsize=18, ha='center', weight='bold', color='#A23B72')
    
    # ========================================================================
    # 2. 编码器详细结构
    # ========================================================================
    
    # 编码器输入
    ax.text(1.5, 9.5, '输入嵌入', fontsize=12, ha='left', weight='bold')
    ax.text(1.5, 9.2, '+ 位置编码', fontsize=10, ha='left', alpha=0.8)
    
    # 编码器层（重复N次）
    encoder_layers_y = [8.0, 6.5, 5.0, 3.5]
    for i, y in enumerate(encoder_layers_y):
        # 编码器层框
        layer_rect = patches.Rectangle((2, y-0.8), 4, 1.2, linewidth=1.5,
                                       edgecolor='#2E86AB', facecolor='white', alpha=0.9)
        ax.add_patch(layer_rect)
        
        # 层标签
        ax.text(2.1, y-0.3, f'编码器层 {i+1}', fontsize=11, ha='left', weight='bold')
        
        # 子层1：多头自注意力
        attn_rect = patches.Rectangle((2.2, y-0.6), 1.6, 0.4, linewidth=1,
                                       edgecolor='#3D5A80', facecolor='#E0F7FA', alpha=0.8)
        ax.add_patch(attn_rect)
        ax.text(2.3, y-0.45, '多头注意力', fontsize=9, ha='left')
        
        # 子层2：前馈网络
        ffn_rect = patches.Rectangle((3.9, y-0.6), 1.9, 0.4, linewidth=1,
                                       edgecolor='#3D5A80', facecolor='#E8F5E8', alpha=0.8)
        ax.add_patch(ffn_rect)
        ax.text(4.0, y-0.45, '前馈网络', fontsize=9, ha='left')
        
        # 残差连接和层归一化
        # 自注意力后的Add & Norm
        ax.plot([2.2, 2.2], [y-0.8, y-0.2], '--', color='#FF6B6B', alpha=0.6, linewidth=1)
        ax.text(2.05, y-0.5, 'Add & Norm', fontsize=8, rotation=90, alpha=0.7)
        
        # 前馈网络后的Add & Norm
        ax.plot([5.8, 5.8], [y-0.8, y-0.2], '--', color='#FF6B6B', alpha=0.6, linewidth=1)
        ax.text(5.65, y-0.5, 'Add & Norm', fontsize=8, rotation=90, alpha=0.7)
        
        # 层间连接
        if i < len(encoder_layers_y) - 1:
            ax.plot([4, 4], [y-0.8, encoder_layers_y[i+1]+0.4], '-', 
                    color='#2E86AB', alpha=0.5, linewidth=1)
    
    # 编码器输出
    ax.text(4, 2.2, '编码器输出', fontsize=12, ha='center', weight='bold')
    
    # ========================================================================
    # 3. 解码器详细结构
    # ========================================================================
    
    # 解码器输入（来自编码器输出和之前生成的token）
    ax.text(9.5, 9.5, '输出嵌入', fontsize=12, ha='left', weight='bold')
    ax.text(9.5, 9.2, '+ 位置编码', fontsize=10, ha='left', alpha=0.8)
    ax.text(9.5, 8.9, '+ 编码器输出', fontsize=10, ha='left', alpha=0.8)
    
    # 解码器层（重复N次）
    decoder_layers_y = [8.0, 6.5, 5.0, 3.5]
    for i, y in enumerate(decoder_layers_y):
        # 解码器层框
        layer_rect = patches.Rectangle((10, y-0.8), 4, 1.2, linewidth=1.5,
                                       edgecolor='#A23B72', facecolor='white', alpha=0.9)
        ax.add_patch(layer_rect)
        
        # 层标签
        ax.text(10.1, y-0.3, f'解码器层 {i+1}', fontsize=11, ha='left', weight='bold')
        
        # 子层1：掩码多头自注意力（因果注意力）
        masked_attn_rect = patches.Rectangle((10.2, y-0.6), 1.8, 0.4, linewidth=1,
                                       edgecolor='#8A4F7D', facecolor='#FFE4E1', alpha=0.8)
        ax.add_patch(masked_attn_rect)
        ax.text(10.3, y-0.45, '掩码多头注意力', fontsize=9, ha='left')
        
        # 子层2：编码器-解码器注意力
        cross_attn_rect = patches.Rectangle((12.1, y-0.6), 1.8, 0.4, linewidth=1,
                                       edgecolor='#8A4F7D', facecolor='#FFF0F5', alpha=0.8)
        ax.add_patch(cross_attn_rect)
        ax.text(12.2, y-0.45, '交叉注意力', fontsize=9, ha='left')
        
        # 子层3：前馈网络
        ffn_rect = patches.Rectangle((14.0, y-0.6), 1.8, 0.4, linewidth=1,
                                       edgecolor='#8A4F7D', facecolor='#F0FFF0', alpha=0.8)
        ax.add_patch(ffn_rect)
        ax.text(14.1, y-0.45, '前馈网络', fontsize=9, ha='left')
        
        # 残差连接和层归一化
        # 掩码注意力后的Add & Norm
        ax.plot([10.2, 10.2], [y-0.8, y-0.2], '--', color='#4ECDC4', alpha=0.6, linewidth=1)
        ax.text(10.05, y-0.5, 'Add & Norm', fontsize=8, rotation=90, alpha=0.7)
        
        # 交叉注意力后的Add & Norm
        ax.plot([13.9, 13.9], [y-0.8, y-0.2], '--', color='#4ECDC4', alpha=0.6, linewidth=1)
        ax.text(13.75, y-0.5, 'Add & Norm', fontsize=8, rotation=90, alpha=0.7)
        
        # 前馈网络后的Add & Norm
        ax.plot([15.8, 15.8], [y-0.8, y-0.2], '--', color='#4ECDC4', alpha=0.6, linewidth=1)
        ax.text(15.65, y-0.5, 'Add & Norm', fontsize=8, rotation=90, alpha=0.7)
        
        # 层间连接
        if i < len(decoder_layers_y) - 1:
            ax.plot([12, 12], [y-0.8, decoder_layers_y[i+1]+0.4], '-', 
                    color='#A23B72', alpha=0.5, linewidth=1)
    
    # 解码器输出
    ax.text(12, 2.2, '解码器输出', fontsize=12, ha='center', weight='bold')
    ax.text(12, 1.9, '→ 线性投影 → Softmax', fontsize=10, ha='center', alpha=0.8)
    
    # ========================================================================
    # 4. 连接线：编码器输出到解码器的交叉注意力
    # ========================================================================
    
    # 编码器输出到解码器交叉注意力的连接
    for y in decoder_layers_y:
        # 从编码器输出到解码器交叉注意力层的箭头
        ax.annotate('', xy=(10, y-0.5), xytext=(8, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#FF9F1C', 
                                   linewidth=1.5, alpha=0.7, linestyle='-'))
    
    # ========================================================================
    # 5. 多头注意力机制详解（右侧详细图）
    # ========================================================================
    
    # 绘制多头注意力机制示意图
    attention_box_x, attention_box_y = 1, 0.5
    attention_width, attention_height = 14, 0.8
    
    # 注意力机制框
    attention_rect = patches.Rectangle(
        (attention_box_x, attention_box_y), attention_width, attention_height,
        linewidth=2, edgecolor='#3D5A80', facecolor='#F8F9FA', alpha=0.9
    )
    ax.add_patch(attention_rect)
    
    # 注意力机制标题
    ax.text(attention_box_x + attention_width/2, attention_box_y + attention_height/2,
            '多头注意力机制详解', fontsize=14, ha='center', va='center', weight='bold')
    
    # 绘制多头注意力流程
    # 1. 线性投影
    ax.text(2.5, 0.7, '线性投影', fontsize=10, ha='center', color='#2E86AB')
    ax.text(2.5, 0.6, 'Q, K, V', fontsize=9, ha='center', alpha=0.8)
    
    # 2. 分割多头
    ax.text(4.5, 0.7, '分割多头', fontsize=10, ha='center', color='#2E86AB')
    ax.text(4.5, 0.6, 'h个注意力头', fontsize=9, ha='center', alpha=0.8)
    
    # 3. 缩放点积注意力
    ax.text(6.5, 0.7, '缩放点积', fontsize=10, ha='center', color='#2E86AB')
    ax.text(6.5, 0.6, 'QKᵀ/√dₖ', fontsize=9, ha='center', alpha=0.8)
    
    # 4. Softmax
    ax.text(8.5, 0.7, 'Softmax', fontsize=10, ha='center', color='#2E86AB')
    ax.text(8.5, 0.6, '归一化权重', fontsize=9, ha='center', alpha=0.8)
    
    # 5. 加权求和
    ax.text(10.5, 0.7, '加权求和', fontsize=10, ha='center', color='#2E86AB')
    ax.text(10.5, 0.6, 'Attention(Q,K,V)', fontsize=9, ha='center', alpha=0.8)
    
    # 6. 合并多头
    ax.text(12.5, 0.7, '合并多头', fontsize=10, ha='center', color='#2E86AB')
    ax.text(12.5, 0.6, 'Concat + 线性投影', fontsize=9, ha='center', alpha=0.8)
    
    # 连接线
    for i in range(5):
        x_start = 3.0 + i * 2.0
        x_end = 4.0 + i * 2.0
        ax.plot([x_start, x_end], [0.65, 0.65], '-', color='#6C757D', alpha=0.5, linewidth=1)
    
    # ========================================================================
    # 6. 图例
    # ========================================================================
    
    # 创建图例元素
    legend_elements = [
        patches.Patch(facecolor='#F0F8FF', edgecolor='#2E86AB', alpha=0.8, label='编码器区域'),
        patches.Patch(facecolor='#FFF0F5', edgecolor='#A23B72', alpha=0.8, label='解码器区域'),
        patches.Patch(facecolor='#E0F7FA', edgecolor='#3D5A80', alpha=0.8, label='注意力子层'),
        patches.Patch(facecolor='#E8F5E8', edgecolor='#3D5A80', alpha=0.8, label='前馈网络子层'),
        mlines.Line2D([], [], color='#FF6B6B', linestyle='--', alpha=0.6, label='Add & Norm 操作'),
        mlines.Line2D([], [], color='#FF9F1C', linestyle='-', alpha=0.7, label='编码器-解码器连接'),
    ]
    
    # 添加图例
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # 7. 添加关键参数说明
    # ========================================================================
    
    # 参数说明框
    param_box = patches.Rectangle((0.5, 10.5), 15, 1.2, linewidth=1,
                                  edgecolor='#6C757D', facecolor='#F8F9FA', alpha=0.8)
    ax.add_patch(param_box)
    
    # 关键参数
    param_text = [
        "关键参数: d_model=512, nhead=8, num_layers=6, dim_feedforward=2048",
        "注意力计算: Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V",
        "位置编码: PE(pos,2i)=sin(pos/10000^{2i/d}), PE(pos,2i+1)=cos(pos/10000^{2i/d})",
        "归一化: LayerNorm(x) = γ ⊙ (x-μ)/σ + β"
    ]
    
    for i, text in enumerate(param_text):
        ax.text(1, 11.2 - i*0.3, text, fontsize=9, ha='left', alpha=0.8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('transformer_architecture_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Transformer架构图已生成并保存为 'transformer_architecture_detailed.png'")
    print("图像尺寸: 16x12 英寸，分辨率: 300 DPI")
    
    return fig

def draw_simplified_transformer():
    """绘制简化版Transformer架构图（用于文档嵌入）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(5, 7.5, 'Transformer简化架构', fontsize=20, ha='center', weight='bold')
    
    # 编码器堆叠
    encoder_layers = 6
    for i in range(encoder_layers):
        y = 6.0 - i * 0.8
        # 编码器层
        rect = patches.Rectangle((2, y-0.3), 2.5, 0.6, linewidth=1.5,
                                 edgecolor='#2E86AB', facecolor='#F0F8FF', alpha=0.8)
        ax.add_patch(rect)
        ax.text(2.1, y, f'编码器层 {i+1}', fontsize=10, ha='left', va='center')
        
        # 内部结构示意
        # 注意力
        attn_circle = patches.Circle((3.5, y-0.1), 0.08, facecolor='#3D5A80', alpha=0.8)
        ax.add_patch(attn_circle)
        
        # 前馈网络
        ffn_circle = patches.Circle((3.8, y-0.1), 0.08, facecolor='#4CAF50', alpha=0.8)
        ax.add_patch(ffn_circle)
        
        # 层间连接
        if i < encoder_layers - 1:
            ax.plot([3.5, 3.5], [y-0.3, y-0.9], '-', color='#2E86AB', alpha=0.5, linewidth=1)
    
    # 解码器堆叠
    decoder_layers = 6
    for i in range(decoder_layers):
        y = 6.0 - i * 0.8
        # 解码器层
        rect = patches.Rectangle((5.5, y-0.3), 2.5, 0.6, linewidth=1.5,
                                 edgecolor='#A23B72', facecolor='#FFF0F5', alpha=0.8)
        ax.add_patch(rect)
        ax.text(5.6, y, f'解码器层 {i+1}', fontsize=10, ha='left', va='center')
        
        # 内部结构示意
        # 掩码注意力
        masked_circle = patches.Circle((6.5, y-0.1), 0.08, facecolor='#8A4F7D', alpha=0.8)
        ax.add_patch(masked_circle)
        
        # 交叉注意力
        cross_circle = patches.Circle((6.8, y-0.1), 0.08, facecolor='#FF9800', alpha=0.8)
        ax.add_patch(cross_circle)
        
        # 前馈网络
        ffn_circle = patches.Circle((7.1, y-0.1), 0.08, facecolor='#4CAF50', alpha=0.8)
        ax.add_patch(ffn_circle)
        
        # 层间连接
        if i < decoder_layers - 1:
            ax.plot([6.8, 6.8], [y-0.3, y-0.9], '-', color='#A23B72', alpha=0.5, linewidth=1)
    
    # 输入输出
    ax.text(1.5, 6.5, '输入', fontsize=12, ha='center', weight='bold')
    ax.text(8.5, 6.5, '输出', fontsize=12, ha='center', weight='bold')
    
    # 注意力机制示意
    ax.text(5, 2.5, '注意力机制', fontsize=14, ha='center', weight='bold')
    
    # 绘制注意力计算流程
    x_pos = [3, 4, 5, 6, 7]
    labels = ['Q', 'K', 'V', 'QKᵀ/√dₖ', 'Softmax', '加权求和']
    
    for i, x in enumerate(x_pos):
        circle = patches.Circle((x, 2), 0.15, facecolor='#FFC107', alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, 2, labels[i], fontsize=10, ha='center', va='center', weight='bold')
        
        if i < len(x_pos) - 1:
            ax.arrow(x+0.15, 2, 0.7, 0, head_width=0.05, head_length=0.05, 
                    fc='#6C757D', ec='#6C757D', alpha=0.6)
    
    # 添加说明
    ax.text(5, 1.5, '多头注意力: 分割为多个子空间并行计算', fontsize=10, ha='center', alpha=0.8)
    ax.text(5, 1.3, '位置编码: 注入序列顺序信息', fontsize=10, ha='center', alpha=0.8)
    ax.text(5, 1.1, '前馈网络: 每个位置独立非线性变换', fontsize=10, ha='center', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('transformer_simplified.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("简化版Transformer架构图已保存为 'transformer_simplified.png'")
    
    return fig

def draw_attention_mechanism_detail():
    """绘制注意力机制详细图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 标题
    ax.text(6, 5.5, '自注意力机制详细图解', fontsize=20, ha='center', weight='bold')
    
    # 1. 输入表示
    ax.text(1, 5, '输入序列', fontsize=14, ha='center', weight='bold')
    ax.text(1, 4.7, 'X ∈ ℝ^{n×d}', fontsize=11, ha='center', alpha=0.8)
    
    # 绘制输入矩阵示意
    for i in range(5):
        for j in range(3):
            rect = patches.Rectangle((0.5+j*0.3, 4.0-i*0.3), 0.25, 0.25,
                                     facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.7)
            ax.add_patch(rect)
    
    # 2. 线性投影
    ax.text(3, 5, '线性投影', fontsize=14, ha='center', weight='bold')
    ax.text(3, 4.7, 'Q = XW_Q, K = XW_K, V = XW_V', fontsize=11, ha='center', alpha=0.8)
    
    # 绘制投影过程
    ax.arrow(1.8, 4.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='#4CAF50', ec='#4CAF50')
    
    # 3. 注意力计算
    ax.text(6, 5, '注意力计算', fontsize=14, ha='center', weight='bold')
    ax.text(6, 4.7, 'Attention(Q,K,V)=softmax(QKᵀ/√d)V', fontsize=11, ha='center', alpha=0.8)
    
    # 绘制注意力矩阵
    attention_matrix = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.05, 0.1, 0.7, 0.1, 0.05],
        [0.03, 0.05, 0.1, 0.75, 0.07],
        [0.02, 0.05, 0.05, 0.07, 0.81]
    ])
    
    # 绘制热力图
    im = ax.imshow(attention_matrix, cmap='YlOrRd', extent=[4.5, 7.5, 2.5, 4.5], alpha=0.8)
    
    # 4. 输出
    ax.text(9, 5, '输出', fontsize=14, ha='center', weight='bold')
    ax.text(9, 4.7, 'O = Attention(Q,K,V)', fontsize=11, ha='center', alpha=0.8)
    
    # 绘制输出矩阵
    for i in range(5):
        for j in range(3):
            rect = patches.Rectangle((8.5+j*0.3, 4.0-i*0.3), 0.25, 0.25,
                                     facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.7)
            ax.add_patch(rect)
    
    # 连接箭头
    ax.arrow(7.5, 3.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='#FF9800', ec='#FF9800')
    
    # 详细公式说明
    formula_box = patches.Rectangle((1, 1), 10, 1.5, linewidth=1,
                                    edgecolor='#6C757D', facecolor='#F5F5F5', alpha=0.9)
    ax.add_patch(formula_box)
    
    formulas = [
        "1. 查询、键、值投影: Q = XW_Q, K = XW_K, V = XW_V, W_Q, W_K, W_V ∈ ℝ^{d×dₖ}",
        "2. 注意力分数: S = QKᵀ ∈ ℝ^{n×n}, s_ij = q_i·k_j",
        "3. 缩放: S_scaled = S/√dₖ (防止梯度消失)",
        "4. Softmax: A = softmax(S_scaled), a_ij = exp(s_ij/√dₖ)/Σ_k exp(s_ik/√dₖ)",
        "5. 加权求和: O = AV, o_i = Σ_j a_ij v_j"
    ]
    
    for i, formula in enumerate(formulas):
        ax.text(1.5, 2.2 - i*0.25, formula, fontsize=9, ha='left', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('attention_mechanism_detail.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("注意力机制详细图已保存为 'attention_mechanism_detail.png'")
    
    return fig

def main():
    """主函数：生成所有架构图"""
    print("=" * 60)
    print("Transformer架构图生成器")
    print("=" * 60)
    
    print("\n1. 生成详细Transformer架构图...")
    fig1 = draw_transformer_architecture()
    
    print("\n2. 生成简化版Transformer架构图...")
    fig2 = draw_simplified_transformer()
    
    print("\n3. 生成注意力机制详细图...")
    fig3 = draw_attention_mechanism_detail()
    
    print("\n" + "=" * 60)
    print("所有架构图生成完成!")
    print("=" * 60)
    
    print("\n生成的文件:")
    print("1. transformer_architecture_detailed.png - 详细架构图")
    print("2. transformer_simplified.png - 简化架构图")
    print("3. attention_mechanism_detail.png - 注意力机制详解")
    
    print("\n说明:")
    print("• 详细架构图展示了完整的编码器-解码器结构")
    print("• 简化架构图适合嵌入文档和演示")
    print("• 注意力机制图详细解释了计算过程")
    
    return fig1, fig2, fig3

if __name__ == "__main__":
    main()