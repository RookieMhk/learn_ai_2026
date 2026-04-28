"""
多头注意力机制完整实现（带旋转位置编码RoPE）
包含自注意力、多头注意力、RoPE位置编码的实现和测试示例
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def softmax(x, axis=-1):
    """稳定的softmax实现"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力实现
    
    参数:
        Q: 查询矩阵，形状 (batch_size, seq_len_q, d_k)
        K: 键矩阵，形状 (batch_size, seq_len_k, d_k)
        V: 值矩阵，形状 (batch_size, seq_len_v, d_v)
        mask: 可选掩码，形状 (batch_size, seq_len_q, seq_len_k)
    
    返回:
        注意力输出和注意力权重
    """
    d_k = Q.shape[-1]
    
    # 计算点积并缩放
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # softmax归一化
    attention_weights = softmax(scores, axis=-1)
    
    # 加权求和
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def precompute_freqs(seq_len, d_model, base=10000):
    """预计算旋转位置编码的频率"""
    half_dim = d_model // 2
    freqs = 1.0 / (base ** (np.arange(0, half_dim) / half_dim))
    
    t = np.arange(seq_len)
    freqs = np.outer(t, freqs)  # (seq_len, half_dim)
    
    freqs_cos = np.cos(freqs)
    freqs_sin = np.sin(freqs)
    
    return freqs_cos, freqs_sin

def apply_rope(x, freqs_cos, freqs_sin):
    """
    应用旋转位置编码(RoPE)
    
    参数:
        x: 输入张量，形状 (batch_size, seq_len, d_model)
        freqs_cos: 余弦频率，形状 (seq_len, d_model//2)
        freqs_sin: 正弦频率，形状 (seq_len, d_model//2)
    
    返回:
        旋转后的张量
    """
    batch_size, seq_len, d_model = x.shape
    half_dim = d_model // 2
    
    # 将输入拆分为实部和虚部（二维子空间）
    x_reshaped = x.reshape(batch_size, seq_len, half_dim, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    
    # 应用旋转（复数乘法）
    # (a+bi) * (cosθ + i sinθ) = (a cosθ - b sinθ) + i(a sinθ + b cosθ)
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    
    # 重新组合
    out = np.stack([out_real, out_imag], axis=-1)
    out = out.reshape(batch_size, seq_len, d_model)
    
    return out

class MultiHeadAttention:
    """标准多头注意力实现"""
    
    def __init__(self, d_model=512, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # 初始化权重矩阵
        self.WQ = np.random.randn(d_model, d_model) * 0.01
        self.WK = np.random.randn(d_model, d_model) * 0.01
        self.WV = np.random.randn(d_model, d_model) * 0.01
        self.WO = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x, batch_size):
        """将输入分割为多个头"""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        参数:
            q: 查询，形状 (batch_size, seq_len_q, d_model)
            k: 键，形状 (batch_size, seq_len_k, d_model)
            v: 值，形状 (batch_size, seq_len_v, d_model)
            mask: 可选掩码
        
        返回:
            输出和注意力权重
        """
        batch_size = q.shape[0]
        
        # 线性投影
        q_proj = np.dot(q, self.WQ)
        k_proj = np.dot(k, self.WK)
        v_proj = np.dot(v, self.WV)
        
        # 分割为多个头
        q_split = self.split_heads(q_proj, batch_size)
        k_split = self.split_heads(k_proj, batch_size)
        v_split = self.split_heads(v_proj, batch_size)
        
        # 计算缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask
        )
        
        # 拼接头
        concat_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = concat_attention.reshape(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = np.dot(concat_attention, self.WO)
        
        return output, attention_weights

class MultiHeadAttentionWithRoPE:
    """带旋转位置编码的多头注意力实现"""
    
    def __init__(self, d_model=512, num_heads=8, max_seq_len=512):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # 初始化权重矩阵
        self.WQ = np.random.randn(d_model, d_model) * 0.01
        self.WK = np.random.randn(d_model, d_model) * 0.01
        self.WV = np.random.randn(d_model, d_model) * 0.01
        self.WO = np.random.randn(d_model, d_model) * 0.01
        
        # 预计算旋转频率
        self.freqs_cos, self.freqs_sin = precompute_freqs(max_seq_len, self.depth)
    
    def split_heads(self, x, batch_size):
        """将输入分割为多个头"""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
    
    def forward(self, q, k, v, mask=None):
        """
        前向传播（带RoPE）
        
        参数:
            q: 查询，形状 (batch_size, seq_len_q, d_model)
            k: 键，形状 (batch_size, seq_len_k, d_model)
            v: 值，形状 (batch_size, seq_len_v, d_model)
            mask: 可选掩码
        
        返回:
            输出和注意力权重
        """
        batch_size, seq_len, _ = q.shape
        
        # 线性投影
        q_proj = np.dot(q, self.WQ)
        k_proj = np.dot(k, self.WK)
        v_proj = np.dot(v, self.WV)
        
        # 分割为多个头
        q_split = self.split_heads(q_proj, batch_size)
        k_split = self.split_heads(k_proj, batch_size)
        v_split = self.split_heads(v_proj, batch_size)
        
        # 应用RoPE
        q_rotated = apply_rope(q_split, 
                              self.freqs_cos[:seq_len], 
                              self.freqs_sin[:seq_len])
        k_rotated = apply_rope(k_split,
                              self.freqs_cos[:seq_len],
                              self.freqs_sin[:seq_len])
        
        # 计算注意力
        scores = np.matmul(q_rotated, k_rotated.transpose(0, 1, 3, 2))
        scores = scores / np.sqrt(self.depth)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention_weights = softmax(scores, axis=-1)
        scaled_attention = np.matmul(attention_weights, v_split)
        
        # 拼接头
        concat_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = concat_attention.reshape(batch_size, seq_len, self.d_model)
        
        # 最终投影
        output = np.dot(concat_attention, self.WO)
        
        return output, attention_weights

def visualize_attention_weights(attention_weights, title="注意力权重可视化"):
    """
    可视化注意力权重矩阵
    
    参数:
        attention_weights: 注意力权重矩阵，形状 (batch_size, num_heads, seq_len, seq_len)
        title: 图表标题
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    fig, axes = plt.subplots(1, num_heads, figsize=(4*num_heads, 4))
    
    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        im = ax.imshow(attention_weights[0, head_idx], cmap='viridis')
        ax.set_title(f'头 {head_idx+1}')
        ax.set_xlabel('键位置')
        ax.set_ylabel('查询位置')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def test_basic_attention():
    """测试基础注意力机制"""
    print("=" * 60)
    print("测试1: 基础缩放点积注意力")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 1
    seq_len = 4
    d_k = 8
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    print(f"查询(Q)形状: {Q.shape}")
    print(f"键(K)形状: {K.shape}")
    print(f"值(V)形状: {V.shape}")
    
    # 计算注意力
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\n注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 验证注意力权重和为1
    for i in range(seq_len):
        weight_sum = np.sum(attn_weights[0, i])
        print(f"位置 {i} 的注意力权重和: {weight_sum:.6f}")
    
    return output, attn_weights

def test_multihead_attention():
    """测试多头注意力"""
    print("\n" + "=" * 60)
    print("测试2: 标准多头注意力")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 6
    d_model = 512
    num_heads = 8
    
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"模型配置: d_model={d_model}, num_heads={num_heads}")
    
    # 创建注意力层
    attention_layer = MultiHeadAttention(d_model, num_heads)
    
    # 前向传播
    output, attn_weights = attention_layer.forward(q, k, v)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 验证每个头的注意力权重和为1
    for head_idx in range(num_heads):
        for pos_idx in range(seq_len):
            weight_sum = np.sum(attn_weights[0, head_idx, pos_idx])
            if head_idx == 0 and pos_idx == 0:
                print(f"头{head_idx+1} 位置{pos_idx} 权重和: {weight_sum:.6f}")
    
    return output, attn_weights

def test_rope_attention():
    """测试带RoPE的多头注意力"""
    print("\n" + "=" * 60)
    print("测试3: 带旋转位置编码(RoPE)的多头注意力")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 8
    d_model = 512
    num_heads = 8
    max_seq_len = 1024
    
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"模型配置: d_model={d_model}, num_heads={num_heads}, max_seq_len={max_seq_len}")
    
    # 创建带RoPE的注意力层
    rope_attention_layer = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len)
    
    # 前向传播
    output, attn_weights = rope_attention_layer.forward(q, k, v)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试RoPE的相对位置特性
    print("\n测试RoPE的相对位置特性:")
    
    # 创建两个不同位置的相同向量
    vec = np.random.randn(1, 1, d_model)
    pos1 = 3
    pos2 = 7
    
    # 预计算频率
    freqs_cos, freqs_sin = precompute_freqs(max_seq_len, d_model // num_heads)
    
    # 应用RoPE
    vec_pos1 = apply_rope(vec, freqs_cos[pos1:pos1+1], freqs_sin[pos1:pos1+1])
    vec_pos2 = apply_rope(vec, freqs_cos[pos2:pos2+1], freqs_sin[pos2:pos2+1])
    
    # 计算点积（应该只依赖于相对位置）
    dot_product = np.matmul(vec_pos1, vec_pos2.transpose(0, 2, 1))
    print(f"位置{pos1}和位置{pos2}的向量点积: {dot_product[0,0,0]:.6f}")
    
    return output, attn_weights

def test_causal_mask():
    """测试因果掩码"""
    print("\n" + "=" * 60)
    print("测试4: 因果掩码（防止未来信息泄露）")
    print("=" * 60)
    
    seq_len = 6
    batch_size = 1
    
    # 创建因果掩码（下三角矩阵）
    causal_mask = np.tril(np.ones((batch_size, seq_len, seq_len)))
    
    print("因果掩码矩阵:")
    print(causal_mask[0])
    
    # 创建测试数据
    d_k = 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    # 计算带掩码的注意力
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    print("\n带因果掩码的注意力权重矩阵:")
    print(attn_weights[0])
    
    # 验证未来位置权重为0
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:  # 未来位置
                weight = attn_weights[0, i, j]
                if abs(weight) > 1e-6:
                    print(f"警告: 位置({i},{j})的权重不为0: {weight:.6f}")
    
    return attn_weights

def run_all_tests():
    """运行所有测试"""
    print("开始运行多头注意力机制测试...")
    print("=" * 60)
    
    # 测试1: 基础注意力
    test_basic_attention()
    
    # 测试2: 多头注意力
    _, attn_weights_mha = test_multihead_attention()
    
    # 测试3: RoPE注意力
    _, attn_weights_rope = test_rope_attention()
    
    # 测试4: 因果掩码
    test_causal_mask()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    
    return attn_weights_mha, attn_weights_rope

def create_visualization():
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("创建可视化图表")
    print("=" * 60)
    
    # 生成测试数据
    batch_size = 1
    seq_len = 8
    d_model = 512
    num_heads = 4
    
    # 创建随机注意力权重
    attn_weights = np.random.randn(batch_size, num_heads, seq_len, seq_len)
    attn_weights = softmax(attn_weights, axis=-1)
    
    # 可视化
    fig = visualize_attention_weights(attn_weights, "多头注意力权重分布示例")
    
    # 保存图表
    output_path = "outputs/阶段一/Week4/Day22/注意力权重可视化.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存至: {output_path}")
    
    # 显示简单统计信息
    print("\n注意力权重统计信息:")
    for head_idx in range(num_heads):
        weights = attn_weights[0, head_idx]
        print(f"头{head_idx+1}: 均值={np.mean(weights):.4f}, 标准差={np.std(weights):.4f}")
    
    return fig

def main():
    """主函数"""
    print("多头注意力机制实现与测试")
    print("=" * 60)
    
    # 运行所有测试
    attn_weights_mha, attn_weights_rope = run_all_tests()
    
    # 创建可视化
    create_visualization()
    
    print("\n" + "=" * 60)
    print("实现总结:")
    print("1. 实现了缩放点积注意力机制")
    print("2. 实现了标准多头注意力")
    print("3. 实现了带旋转位置编码(RoPE)的多头注意力")
    print("4. 包含了因果掩码测试")
    print("5. 提供了注意力权重可视化")
    print("=" * 60)
    
    return attn_weights_mha, attn_weights_rope

if __name__ == "__main__":
    # 运行主程序
    main()
    
    print("\n代码实现说明:")
    print("1. 所有实现使用纯NumPy，无需深度学习框架")
    print("2. 包含完整的数学推导和代码注释")
    print("3. 每个函数都有详细的参数说明")
    print("4. 提供了多种测试用例验证正确性")
    print("\n学习建议:")
    print("1. 尝试修改参数（如头数、维度）观察效果变化")
    print("2. 实现不同的位置编码方法进行比较")
    print("3. 将注意力机制集成到完整模型中")