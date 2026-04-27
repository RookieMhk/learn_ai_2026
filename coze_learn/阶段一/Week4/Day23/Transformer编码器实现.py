#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer编码器完整实现
包含：多头自注意力、位置编码、前馈网络、层归一化、残差连接
提供可运行的示例和测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 位置编码（Positional Encoding）
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码：使用正弦和余弦函数编码序列位置信息
    支持学习的位置编码和预计算的位置编码两种模式
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, 
                 learnable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        if learnable:
            # 可学习的位置编码（如BERT）
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, mean=0.0, std=0.02)
        else:
            # 预计算的正弦/余弦位置编码（原始Transformer）
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * 
                (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model]
        """
        if self.learnable:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    def visualize(self, seq_len: int = 50, d_model_slice: int = 64):
        """
        可视化位置编码
        """
        if self.learnable:
            pe_data = self.pe[0, :seq_len, :d_model_slice].detach().cpu().numpy()
        else:
            pe_data = self.pe[0, :seq_len, :d_model_slice].cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.imshow(pe_data.T, aspect='auto', cmap='RdBu', interpolation='nearest')
        plt.colorbar(label='位置编码值')
        plt.xlabel('序列位置')
        plt.ylabel('特征维度（前64维）')
        plt.title(f'位置编码可视化 ({self.learnable and "可学习" or "正弦/余弦"})')
        plt.tight_layout()
        plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"位置编码可视化已保存到 positional_encoding.png")

# ============================================================================
# 2. 多头自注意力（Multi-Head Self-Attention）
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    支持因果掩码（用于解码器）和填充掩码
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            key_padding_mask: [batch_size, seq_len_k] (True表示需要mask的位置)
            attn_mask: [seq_len_q, seq_len_k] 或 [batch_size, nhead, seq_len_q, seq_len_k]
        返回:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, nhead, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 1. 线性投影并分割多头
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask
        
        # 4. 应用键填充掩码
        if key_padding_mask is not None:
            # 扩展维度以匹配注意力分数
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len_k]
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # 5. Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 7. 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 8. 输出投影
        output = self.W_o(context)
        
        return output, attention_weights
    
    def visualize_attention(self, 
                           attention_weights: torch.Tensor,
                           title: str = "注意力权重可视化"):
        """
        可视化注意力权重
        """
        attn = attention_weights[0].detach().cpu().numpy()  # [nhead, seq_len_q, seq_len_k]
        nhead = attn.shape[0]
        
        fig, axes = plt.subplots(1, nhead, figsize=(4*nhead, 4))
        if nhead == 1:
            axes = [axes]
        
        for i in range(nhead):
            ax = axes[i]
            im = ax.imshow(attn[i], cmap='viridis', aspect='auto')
            ax.set_title(f'头 {i+1}')
            ax.set_xlabel('键位置')
            ax.set_ylabel('查询位置')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"注意力权重可视化已保存到 attention_weights.png")

# ============================================================================
# 3. 前馈网络（Feed-Forward Network）
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    位置感知前馈网络
    每个位置独立进行相同的非线性变换
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数选择
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================================================================
# 4. Transformer编码器层（Encoder Layer）
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含：多头自注意力 + 前馈网络，每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'gelu',
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        # 自注意力子层
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # 前馈网络子层
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout, activation)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 使用Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            src: [batch_size, seq_len, d_model]
            src_mask: [seq_len, seq_len] 或 [batch_size, nhead, seq_len, seq_len]
            src_key_padding_mask: [batch_size, seq_len] (True表示需要mask的位置)
        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, nhead, seq_len, seq_len]
        """
        # 1. 自注意力子层（Pre-LayerNorm）
        residual = src
        src_norm = self.norm1(src)
        attn_output, attention_weights = self.self_attn(
            query=src_norm,
            key=src_norm,
            value=src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = residual + self.dropout1(attn_output)
        
        # 2. 前馈网络子层（Pre-LayerNorm）
        residual = src
        src_norm = self.norm2(src)
        ffn_output = self.ffn(src_norm)
        src = residual + self.dropout2(ffn_output)
        
        return src, attention_weights

# ============================================================================
# 5. Transformer编码器（完整编码器）
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    包含：词嵌入、位置编码、多个编码器层
    """
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000,
                 padding_idx: int = 0,
                 learnable_pos_encoding: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        # 1. 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        
        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout,
            learnable=learnable_pos_encoding
        )
        
        # 3. 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])
        
        # 4. 最终归一化层
        self.norm = nn.LayerNorm(d_model)
        
        # 5. 输出投影（可选，用于分类任务）
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 词嵌入初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx].zero_()
        
        # 输出投影初始化
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, 
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            src: [batch_size, seq_len] (token indices)
            src_mask: [seq_len, seq_len] 或 [batch_size, nhead, seq_len, seq_len]
            src_key_padding_mask: [batch_size, seq_len] (True表示padding位置)
        返回:
            output: [batch_size, seq_len, d_model]
            all_attention_weights: List of [batch_size, nhead, seq_len, seq_len]
        """
        # 1. 词嵌入（乘以sqrt(d_model)以平衡梯度）
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        
        # 2. 添加位置编码
        embedded = self.pos_encoding(embedded)
        
        # 3. 通过编码器层
        all_attention_weights = []
        output = embedded
        
        for layer in self.layers:
            output, attention_weights = layer(
                output, src_mask, src_key_padding_mask
            )
            all_attention_weights.append(attention_weights)
        
        # 4. 最终归一化
        output = self.norm(output)
        
        # 5. 输出投影（可选）
        logits = self.output_projection(output)
        
        return output, logits, all_attention_weights
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """获取词嵌入（包含位置编码）"""
        embedded = self.embedding(token_ids) * math.sqrt(self.d_model)
        return self.pos_encoding(embedded)

# ============================================================================
# 6. 实用函数和测试
# ============================================================================

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建填充掩码
    True表示padding位置（需要被mask）
    """
    return (seq == pad_idx)

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果掩码（下三角矩阵）
    True表示未来位置（需要被mask）
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask

def test_transformer_encoder():
    """测试Transformer编码器"""
    print("=" * 60)
    print("Transformer编码器测试")
    print("=" * 60)
    
    # 超参数
    batch_size = 4
    seq_len = 32
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 6
    
    # 创建模型
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 创建输入数据
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # 创建填充掩码（假设前10个位置是真实token，后22个是padding）
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    src_key_padding_mask[:, 20:] = True  # 后12个位置是padding
    
    # 前向传播
    output, logits, attention_weights = model(src, src_key_padding_mask=src_key_padding_mask)
    
    # 打印信息
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Logits形状: {logits.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    print(f"每个注意力权重形状: {attention_weights[0].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试梯度
    loss = logits.mean()
    loss.backward()
    
    # 检查梯度
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
    
    print(f"梯度范数: {grad_norm:.4f}")
    print("测试通过!")
    
    return model, output, attention_weights

def visualize_model_architecture(model: nn.Module):
    """可视化模型架构"""
    print("\n" + "=" * 60)
    print("模型架构信息")
    print("=" * 60)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 计算各层参数量
    print("\n各层参数量:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:40} | {param_count:12,}")
    
    print(f"\n总可训练参数量: {total_params:,}")
    
    # 绘制参数量分布
    layer_names = []
    param_counts = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() > 1000:  # 只显示参数量较大的层
            layer_names.append(name.split('.')[-1])
            param_counts.append(param.numel())
    
    # 限制显示数量
    if len(layer_names) > 10:
        layer_names = layer_names[:10]
        param_counts = param_counts[:10]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(layer_names)), param_counts)
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.ylabel('参数量')
    plt.title('模型各层参数量分布')
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(param_counts),
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"模型架构可视化已保存到 model_architecture.png")

def benchmark_inference_speed(model: nn.Module, device: torch.device):
    """基准测试推理速度"""
    print("\n" + "=" * 60)
    print("推理速度基准测试")
    print("=" * 60)
    
    import time
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512, 1024]
    batch_size = 2
    vocab_size = 10000
    
    results = []
    
    for seq_len in seq_lengths:
        # 创建输入
        src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        
        # 预热
        for _ in range(10):
            _ = model(src, src_key_padding_mask=src_key_padding_mask)
        
        # 基准测试
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        iterations = 50
        for _ in range(iterations):
            output, _, _ = model(src, src_key_padding_mask=src_key_padding_mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # 毫秒
        
        # 估计计算复杂度
        theoretical_complexity = seq_len ** 2
        
        results.append({
            "序列长度": seq_len,
            "平均推理时间(ms)": f"{avg_time:.2f}",
            "理论复杂度(O(n²))": f"{theoretical_complexity:,}"
        })
    
    # 打印结果
    print(f"{'序列长度':<10} | {'平均推理时间(ms)':<20} | {'理论复杂度(O(n²))':<20}")
    print("-" * 60)
    for r in results:
        print(f"{r['序列长度']:<10} | {r['平均推理时间(ms)']:<20} | {r['理论复杂度(O(n²))']:<20}")
    
    # 绘制速度曲线
    seq_lens = [r['序列长度'] for r in results]
    times = [float(r['平均推理时间(ms)'].split()[0]) for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('序列长度')
    plt.ylabel('推理时间 (ms)')
    plt.title('Transformer编码器推理速度 vs 序列长度')
    plt.grid(True, alpha=0.3)
    
    # 添加二次函数拟合线（理论复杂度）
    x_fit = np.linspace(min(seq_lens), max(seq_lens), 100)
    # 归一化拟合
    coeff = np.polyfit(seq_lens, times, 2)
    y_fit = np.polyval(coeff, x_fit)
    plt.plot(x_fit, y_fit, 'r--', alpha=0.7, label='二次拟合')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('inference_speed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"推理速度曲线已保存到 inference_speed.png")
    
    return results

# ============================================================================
# 7. 主函数：运行完整测试
# ============================================================================

def main():
    """主函数：运行完整测试"""
    print("Transformer编码器完整实现")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 测试Transformer编码器
    print("\n1. 测试Transformer编码器...")
    model, output, attention_weights = test_transformer_encoder()
    model = model.to(device)
    
    # 2. 可视化模型架构
    print("\n2. 可视化模型架构...")
    visualize_model_architecture(model)
    
    # 3. 可视化位置编码
    print("\n3. 可视化位置编码...")
    pos_encoder = PositionalEncoding(d_model=512, max_len=100)
    pos_encoder.visualize(seq_len=50)
    
    # 4. 可视化注意力权重（示例）
    print("\n4. 可视化注意力权重...")
    if len(attention_weights) > 0:
        # 使用第一个样本的第一个注意力头
        sample_attn = attention_weights[0][0:1]  # [1, nhead, seq_len, seq_len]
        
        # 创建MultiHeadAttention实例用于可视化
        attn_module = MultiHeadAttention(d_model=512, nhead=8)
        attn_module.visualize_attention(
            sample_attn,
            title="示例注意力权重（第一个样本）"
        )
    
    # 5. 基准测试推理速度
    print("\n5. 基准测试推理速度...")
    if device.type == 'cuda':
        benchmark_inference_speed(model, device)
    else:
        print("GPU不可用，跳过推理速度测试")
    
    # 6. 保存模型示例
    print("\n6. 保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': 10000,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1
        }
    }, 'transformer_encoder_example.pth')
    print("模型已保存到 transformer_encoder_example.pth")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    
    # 总结
    print("\n总结:")
    print("1. 成功实现了完整的Transformer编码器")
    print("2. 包含位置编码、多头注意力、前馈网络等核心组件")
    print("3. 提供了可视化工具和基准测试")
    print("4. 模型支持填充掩码和注意力掩码")
    print("5. 代码完全可运行，适合学习和实验")

# ============================================================================
# 8. 运行主函数
# ============================================================================

if __name__ == "__main__":
    main()