"""
注意力机制实现
Week 3 Day 5 实践代码

包含：
1. 手动实现自注意力机制
2. 多头注意力模块
3. 位置编码可视化
4. 完整Transformer编码器层

运行方式：python 注意力机制实现.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("长期计划/个性化学习计划/outputs/阶段二/Week3/Day5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 第一部分：基础自注意力机制实现
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    
    参数:
        Q: Query矩阵 (seq_len, d_k)
        K: Key矩阵 (seq_len, d_k)
        V: Value矩阵 (seq_len, d_v)
        mask: 掩码矩阵，可选
    
    返回:
        output: 注意力输出
        attention_weights: 注意力权重矩阵
    """
    d_k = Q.shape[-1]
    
    # 1. 计算Q和K的点积，得到相似度矩阵
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (seq_len, seq_len)
    
    # 2. 缩放，防止点积过大导致Softmax梯度消失
    scores = scores / math.sqrt(d_k)
    
    # 3. 应用掩码（可选，用于因果掩码）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax归一化，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和得到输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class SelfAttention(nn.Module):
    """
    手动实现的自注意力层
    """
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        # 可学习的投影矩阵
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.W_O = nn.Linear(d_v, d_model)
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 生成 Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, d_k)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 计算注意力
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 最终投影
        output = self.W_O(output)  # (batch, seq_len, d_model)
        
        return output, attention_weights


# ============================================================================
# 第二部分：多头注意力实现
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想：并行运行多个注意力头，每个头关注不同的语义关系
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 使用nn.Linear实现，与nn.MultiheadAttention接口一致
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        返回: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 重塑为多头形式: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算多头注意力
        if mask is not None:
            mask = mask.unsqueeze(1)  # 广播维度
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        
        return output, attention_weights


# ============================================================================
# 第三部分：位置编码实现
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（来自原论文）
    
    原理：
    - 偶数维度使用sin编码
    - 奇数维度使用cos编码
    - 不同频率的波叠加，形成唯一的位置标识
    """
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = self._create positional_encoding(max_seq_len, d_model)
        pe = pe.unsqueeze(0)  # 添加batch维度
        self.register_buffer('pe', pe)
    
    def _create_positional_encoding(self, seq_len, d_model):
        """生成位置编码矩阵"""
        PE = np.zeros((seq_len, d_model))
        
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        
        PE[:, 0::2] = np.sin(position * div_term)  # 偶数维度
        PE[:, 1::2] = np.cos(position * div_term)  # 奇数维度
        
        return torch.FloatTensor(PE)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码（2026年主流方案）
    
    将位置作为可学习的参数，让模型自己学习最优的位置表示
    """
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # 初始化位置ID
        self.register_buffer(
            'position_ids',
            torch.arange(max_seq_len).unsqueeze(0)
        )
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


def visualize_positional_encoding():
    """
    可视化位置编码
    """
    d_model = 128
    max_seq_len = 512
    
    # 生成位置编码
    PE = np.zeros((max_seq_len, d_model))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 热力图：位置编码全局视图
    ax1 = axes[0, 0]
    im = ax1.imshow(PE[:100, :64].T, cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Positional Encoding Heatmap (First 100 positions, First 64 dims)')
    plt.colorbar(im, ax=ax1)
    
    # 2. 不同维度的正弦/余弦曲线
    ax2 = axes[0, 1]
    dimensions = [0, 1, 2, 3, 63, 64]
    for dim in dimensions:
        label = f'dim={dim} ({"sin" if dim % 2 == 0 else "cos"})'
        ax2.plot(PE[:100, dim], label=label, linewidth=1.5)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Positional Encoding: Different Dimensions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 不同频率的周期展示
    ax3 = axes[1, 0]
    positions = np.arange(512)
    frequencies = [1, 10, 50, 100]
    for freq in frequencies:
        values = np.sin(2 * np.pi * freq * positions / 512)
        ax3.plot(positions, values, label=f'freq={freq}', alpha=0.7)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Value')
    ax3.set_title('Different Frequencies in Positional Encoding')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不同位置的编码相似度
    ax4 = axes[1, 1]
    # 计算位置之间的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(PE[:50])
    im4 = ax4.imshow(similarities, cmap='viridis', vmin=0, vmax=1)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    ax4.set_title('Cosine Similarity Between Positions')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'positional_encoding_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 位置编码可视化已保存至: {save_path}")
    return save_path


# ============================================================================
# 第四部分：Transformer编码器层
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    每层包含：
    1. 多头自注意力
    2. 残差连接 + 层归一化
    3. 前馈网络
    4. 残差连接 + 层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # 现代模型常用GELU替代ReLU
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class FeedForward(nn.Module):
    """
    前馈神经网络（FFN）
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    现代常用 SwiGLU 变体：
    FFN(x) = SwiGLU(x) = Swish(xW₁) ⊗ (xW₃)W₂
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


# ============================================================================
# 第五部分：可视化注意力权重
# ============================================================================

def visualize_attention_weights():
    """
    可视化多头注意力的权重分布
    """
    # 使用一个简单的示例
    seq_len = 20
    d_model = 64
    num_heads = 8
    
    # 创建随机输入
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, d_model)
    
    # 创建多头注意力层
    attention = MultiHeadAttention(d_model, num_heads)
    attention.eval()
    
    # 计算注意力
    with torch.no_grad():
        _, attention_weights = attention(x)
    
    # attention_weights: (batch, num_heads, seq_len, seq_len)
    attention_weights = attention_weights[0].numpy()  # 移除batch维度
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head in range(num_heads):
        ax = axes[head]
        im = ax.imshow(attention_weights[head], cmap='Blues', aspect='auto')
        ax.set_title(f'Head {head + 1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Multi-Head Attention Weights Visualization', fontsize=14)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'multihead_attention_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 多头注意力可视化已保存至: {save_path}")
    return save_path


def visualize_causal_mask():
    """
    可视化因果掩码的效果
    """
    seq_len = 10
    
    # 创建因果掩码（上三角为-inf）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # 计算注意力分数
    scores = torch.randn(seq_len, seq_len)
    scores = scores.masked_fill(mask, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始分数
    ax1 = axes[0]
    im1 = ax1.imshow(scores.numpy(), cmap='coolwarm', aspect='auto')
    ax1.set_title('Scores with Causal Mask (inf = masked)')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)
    
    # 掩码矩阵
    ax2 = axes[1]
    im2 = ax2.imshow(mask.float().numpy(), cmap='binary', aspect='auto')
    ax2.set_title('Causal Mask (1 = keep, 0 = mask)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2)
    
    # 注意力权重
    ax3 = axes[2]
    im3 = ax3.imshow(attention_weights.numpy(), cmap='Blues', aspect='auto')
    ax3.set_title('Attention Weights (After Softmax)')
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Position')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle('Causal Mask Effect in Decoder', fontsize=14)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'causal_mask_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 因果掩码可视化已保存至: {save_path}")
    return save_path


# ============================================================================
# 第六部分：完整示例 - 使用注意力机制处理文本
# ============================================================================

def text_attention_example():
    """
    完整示例：展示注意力机制如何处理文本
    """
    print("\n" + "="*60)
    print("注意力机制文本处理示例")
    print("="*60)
    
    # 示例句子
    sentence = "The cat sat on the mat because it was tired"
    words = sentence.split()
    print(f"\n输入句子: '{sentence}'")
    print(f"词数: {len(words)}")
    
    # 参数配置
    d_model = 32  # 词向量维度
    num_heads = 4  # 注意力头数
    seq_len = len(words)
    
    # 简单的词嵌入（实际应用中用预训练embeddings）
    torch.manual_seed(123)
    embeddings = torch.randn(seq_len, d_model)
    
    print(f"\n配置:")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 每头维度: {d_model // num_heads}")
    
    # 添加位置编码
    pe = PositionalEncoding(d_model, seq_len)
    embedded = pe.dropout(embeddings.unsqueeze(0))  # 添加batch维度
    
    # 计算多头注意力
    attention = MultiHeadAttention(d_model, num_heads)
    attention.eval()
    
    with torch.no_grad():
        output, weights = attention(embedded)
    
    # 分析 "it" 这个词的注意力分布
    it_idx = words.index("it")
    print(f"\n分析 '{words[it_idx]}' 的注意力分布:")
    print("-" * 40)
    
    # 获取 "it" 对所有词的注意力（取所有头的平均）
    it_attention = weights[0, :, it_idx, :].mean(dim=0).numpy()
    
    attention_pairs = list(zip(words, it_attention))
    attention_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for word, attn in attention_pairs[:5]:
        bar = "█" * int(attn * 40)
        print(f"  {word:>10}: {attn:.3f} {bar}")
    
    # 可视化完整注意力矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_weights = weights[0].mean(dim=0).numpy()
    
    im = ax.imshow(avg_weights, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    ax.set_xlabel('Key Words')
    ax.set_ylabel('Query Words')
    ax.set_title('Average Attention Weights Across All Heads')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'text_attention_example.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 文本注意力可视化已保存至: {save_path}")
    
    return save_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "="*60)
    print("Transformer 注意力机制实现")
    print("Week 3 Day 5 实践代码")
    print("="*60)
    
    # 1. 可视化位置编码
    print("\n[1/4] 生成位置编码可视化...")
    visualize_positional_encoding()
    
    # 2. 可视化多头注意力
    print("\n[2/4] 生成多头注意力可视化...")
    visualize_attention_weights()
    
    # 3. 可视化因果掩码
    print("\n[3/4] 生成因果掩码可视化...")
    visualize_causal_mask()
    
    # 4. 文本注意力示例
    print("\n[4/4] 运行文本注意力示例...")
    text_attention_example()
    
    print("\n" + "="*60)
    print("✅ 所有可视化已完成!")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("="*60)
    
    # 打印生成的文件列表
    generated_files = list(OUTPUT_DIR.glob('*.png'))
    print("\n生成的文件:")
    for f in generated_files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
