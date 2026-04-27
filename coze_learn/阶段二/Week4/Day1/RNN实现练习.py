"""
RNN实现练习 - Week 4 Day 1
================================
使用PyTorch实现LSTM文本分类，包含：
1. LSTM基础文本分类
2. 序列数据处理流程
3. 隐藏状态可视化

运行方式：
    python RNN实现练习.py

依赖安装：
    pip install torch torchvision numpy matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import re
import random

# 设置随机种子保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== 1. 数据准备 =====================

class TextDataset(Dataset):
    """简单的文本分类数据集"""
    
    def __init__(self, texts, labels, word_to_idx, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 文本转索引
        indices = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) 
                   for w in text.split()]
        # 截断或填充
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word_to_idx['<PAD>']] * (self.max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def build_vocab(texts, min_freq=2):
    """构建词汇表"""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    # 保留高频词
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


# ===================== 2. LSTM模型定义 =====================

class LSTMClassifier(nn.Module):
    """
    LSTM文本分类器
    
    Architecture:
        Embedding → LSTM → FC → Output
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 因为双向
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len) 输入序列
            
        Returns:
            output: (batch_size, num_classes) 分类 logits
            hidden_states: 所有时间步的隐藏状态（用于可视化）
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM输出
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_dim * 2) 所有时间步的输出
        # h_n: (num_layers * 2, batch, hidden_dim) 最终隐藏状态
        
        # 使用最后时间步的隐藏状态（双向拼接）
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim * 2)
        
        # 全连接层
        output = self.fc(hidden)
        output = self.dropout(output)
        output = self.output(output)
        
        return output, lstm_out
    
    def get_hidden_states(self, x):
        """获取隐藏状态用于可视化"""
        with torch.no_grad():
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return lstm_out


# ===================== 3. 训练函数 =====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


# ===================== 4. 隐藏状态可视化 =====================

def visualize_hidden_states(model, dataloader, device, num_samples=100):
    """
    可视化LSTM隐藏状态
    使用t-SNE将高维隐藏状态降到2D进行可视化
    """
    model.eval()
    all_hidden = []
    all_labels = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if len(all_labels) >= num_samples:
                break
            batch_x = batch_x.to(device)
            hidden_states = model.get_hidden_states(batch_x)
            
            # 取最后时间步的隐藏状态
            final_hidden = hidden_states[:, -1, :].cpu().numpy()
            all_hidden.append(final_hidden)
            all_labels.extend(batch_y.numpy().tolist())
    
    all_hidden = np.concatenate(all_hidden, axis=0)[:num_samples]
    all_labels = np.array(all_labels[:num_samples])
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=SEED)
    hidden_2d = tsne.fit_transform(all_hidden)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i in range(len(np.unique(all_labels))):
        mask = all_labels == i
        plt.scatter(hidden_2d[mask, 0], hidden_2d[mask, 1], 
                   c=colors[i % len(colors)], label=f'Class {i}', alpha=0.6)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('LSTM Hidden States Visualization (t-SNE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hidden_states_visualization.png', dpi=150)
    print("隐藏状态可视化已保存: hidden_states_visualization.png")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("训练曲线已保存: training_curves.png")
    plt.close()


# ===================== 5. 主程序 =====================

def generate_sample_data():
    """生成示例数据 - 情感分类"""
    # 简单模拟的文本分类数据
    texts = [
        # 正面评论
        "this movie is amazing and wonderful",
        "i love this film it is great",
        "best movie i have ever seen",
        "fantastic story and beautiful acting",
        "enjoyed every minute of this film",
        "brilliant directing and superb cast",
        "outstanding cinematography and music",
        "highly recommend this masterpiece",
        "touching story with deep meaning",
        "perfect execution of the plot",
        
        # 负面评论
        "this movie is terrible and boring",
        "i hate this film it is awful",
        "worst movie i have ever watched",
        "waste of time and money",
        "confusing story and bad acting",
        "poor script and weak direction",
        "disappointing outcome overall",
        "do not recommend this disaster",
        "annoying characters and dull plot",
        "completely lacking any merit",
    ]
    
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 正面
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 负面
    
    # 数据增强：添加一些变体
    new_texts = []
    new_labels = []
    for text, label in zip(texts, labels):
        new_texts.append(text)
        new_labels.append(label)
        # 添加重复版本
        new_texts.append(text.replace(' ', '  '))
        new_labels.append(label)
    
    return new_texts, new_labels


def main():
    # ===================== 配置 =====================
    CONFIG = {
        'embed_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 8,
        'epochs': 30,
        'max_len': 20,
    }
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ===================== 数据 =====================
    print("\n" + "="*50)
    print("1. 准备数据")
    print("="*50)
    
    texts, labels = generate_sample_data()
    word_to_idx = build_vocab(texts, min_freq=1)
    print(f"词汇表大小: {len(word_to_idx)}")
    
    # 划分训练集和验证集
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.8 * len(texts))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = TextDataset(
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        word_to_idx,
        max_len=CONFIG['max_len']
    )
    val_dataset = TextDataset(
        [texts[i] for i in val_idx],
        [labels[i] for i in val_idx],
        word_to_idx,
        max_len=CONFIG['max_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # ===================== 模型 =====================
    print("\n" + "="*50)
    print("2. 构建模型")
    print("="*50)
    
    model = LSTMClassifier(
        vocab_size=len(word_to_idx),
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=2,
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # ===================== 训练 =====================
    print("\n" + "="*50)
    print("3. 训练模型")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    
    # ===================== 可视化 =====================
    print("\n" + "="*50)
    print("4. 生成可视化")
    print("="*50)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 可视化隐藏状态
    visualize_hidden_states(model, val_loader, device)
    
    # ===================== 序列处理流程展示 =====================
    print("\n" + "="*50)
    print("5. 序列处理流程演示")
    print("="*50)
    
    # 展示一个句子的处理过程
    sample_text = "this movie is wonderful"
    sample_indices = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in sample_text.split()]
    print(f"\n原始文本: '{sample_text}'")
    print(f"词索引: {sample_indices}")
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        sample_tensor = torch.tensor([sample_indices], dtype=torch.long).to(device)
        output, hidden_seq = model(sample_tensor)
        
        print(f"\n模型输出 logits: {output.cpu().numpy()}")
        print(f"预测类别: {torch.argmax(output, dim=1).item()} (0=负面, 1=正面)")
        print(f"隐藏状态序列形状: {hidden_seq.shape}")
    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)


if __name__ == '__main__':
    main()
