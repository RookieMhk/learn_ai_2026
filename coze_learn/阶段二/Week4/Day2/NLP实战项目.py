"""
NLP实战项目 - Week 4 Day 2
包含：情感分类、词向量训练与可视化、命名实体识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ==================== 第1部分：数据准备 ====================

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    def tokenize(self, text):
        """简单分词"""
        # 转小写，去除标点
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        for word, freq in counter.items():
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"词汇表大小: {len(self.word2idx)}")
        return self.word2idx
    
    def encode(self, text, max_len=50):
        """文本转索引"""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                   for token in tokens[:max_len]]
        
        # padding
        if len(indices) < max_len:
            indices += [self.word2idx['<PAD>']] * (max_len - len(indices))
        
        return indices


class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, texts, labels, preprocessor, max_len=50):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.preprocessor.encode(text, self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ==================== 第2部分：情感分类模型 ====================

class TextClassifier(nn.Module):
    """基于CNN的文本分类器"""
    
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, filter_sizes=[3, 4, 5]):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch, num_filters, seq_len-fs+1)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        concat = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * 3)
        dropped = self.dropout(concat)
        output = self.fc(dropped)
        
        return output


def train_sentiment_classifier():
    """训练情感分类器"""
    print("=" * 50)
    print("第1部分：情感分类模型训练")
    print("=" * 50)
    
    # 示例数据
    train_texts = [
        "This movie is fantastic! I really enjoyed it.",
        "What a terrible film, waste of time.",
        "Great acting and beautiful cinematography.",
        "Boring and slow, did not like it at all.",
        "Amazing story, highly recommend!",
        "Awful experience, will not watch again.",
        "Love this movie, best one this year!",
        "Disappointing plot, bad script.",
        "Wonderful performance by all actors.",
        "Hate it, the worst movie ever."
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面
    
    test_texts = [
        "Absolutely brilliant!",
        "Very disappointing."
    ]
    test_labels = [1, 0]
    
    # 数据预处理
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(train_texts + test_texts)
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor)
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    vocab_size = len(preprocessor.word2idx)
    model = TextClassifier(vocab_size=vocab_size, embed_dim=64, num_classes=2)
    model.to(device)
    
    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    model.train()
    for epoch in range(30):
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # 测试
    print("\n测试结果:")
    model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            
            for i, text in enumerate(texts):
                pred = "正面" if preds[i].item() == 1 else "负面"
                true = "正面" if labels[i].item() == 1 else "负面"
                print(f"文本: {test_texts[i]}")
                print(f"预测: {pred}, 真实: {true}")
                print()
    
    print("情感分类完成!\n")


# ==================== 第3部分：词向量训练与可视化 ====================

def train_word_embeddings():
    """训练Word2Vec词向量"""
    print("=" * 50)
    print("第2部分：词向量训练与可视化")
    print("=" * 50)
    
    # 训练语料
    corpus = [
        ['king', 'queen', 'man', 'woman', 'prince', 'princess'],
        ['apple', 'fruit', 'banana', 'orange', 'grape'],
        ['dog', 'cat', 'animal', 'pet', 'bird'],
        ['machine', 'learning', 'deep', 'neural', 'network', 'ai'],
        ['happy', 'joy', 'sad', 'angry', 'emotion'],
        ['run', 'walk', 'jump', 'swim', 'exercise'],
    ]
    
    # 使用简单的CBOW模型实现
    class SimpleWord2Vec(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.target_embed = nn.Embedding(vocab_size, embed_dim)
            self.context_embed = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, target, context):
            target_vec = self.target_embed(target)
            context_vec = self.context_embed(context)
            
            # 点积相似度
            score = torch.sum(target_vec * context_vec, dim=1)
            return score
    
    # 构建简单词汇表
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    for sentence in corpus:
        for word in sentence:
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word
    
    vocab_size = len(word2idx)
    print(f"词汇表大小: {vocab_size}")
    
    # 训练词向量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleWord2Vec(vocab_size, embed_dim=16)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 准备训练数据（简化版）
    train_targets = []
    train_contexts = []
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in word2idx:
                train_targets.append(word2idx[word])
                # 随机选择上下文词
                context_idx = (i + 1) % len(sentence)
                train_contexts.append(word2idx[sentence[context_idx]])
    
    train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)
    train_contexts = torch.tensor(train_contexts, dtype=torch.long).to(device)
    
    print("\n训练词向量...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        for i in range(len(train_targets)):
            target = train_targets[i:i+1]
            context = train_contexts[i:i+1]
            
            optimizer.zero_grad()
            score = model(target, context)
            # 简化的损失函数
            loss = -torch.mean(score)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # 提取词向量
    embeddings = model.target_embed.weight.detach().cpu().numpy()
    print(f"\n词向量维度: {embeddings.shape}")
    
    # 可视化
    visualize_word_embeddings(embeddings, idx2word)
    
    print("词向量训练完成!\n")


def visualize_word_embeddings(embeddings, idx2word):
    """词向量可视化"""
    # 只可视化词汇
    words_to_show = [idx2word[i] for i in range(len(idx2word)) if i in idx2word]
    vectors_to_show = [embeddings[i] for i in range(len(idx2word)) if i in idx2word]
    
    if len(vectors_to_show) < 3:
        print("词汇太少，跳过可视化")
        return
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors_to_show)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words_to_show):
        plt.scatter(coords[i, 0], coords[i, 1], s=100, alpha=0.7)
        plt.annotate(word, (coords[i, 0] + 0.05, coords[i, 1] + 0.05), fontsize=12)
    
    plt.title('Word Embeddings Visualization (PCA)', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = '长期计划/个性化学习计划/outputs/阶段二/Week4/Day2/word_embeddings.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"词向量可视化已保存至: {save_path}")
    plt.close()


# ==================== 第4部分：命名实体识别 ====================

class NERModel(nn.Module):
    """简单的BiLSTM NER模型"""
    
    def __init__(self, vocab_size, tag_to_ix, embed_dim=64, hidden_dim=32):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, len(tag_to_ix))
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out)
        return out


def train_ner_model():
    """训练NER模型"""
    print("=" * 50)
    print("第3部分：命名实体识别模型")
    print("=" * 50)
    
    # 标签定义
    tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 
                 'B-LOC': 5, 'I-LOC': 6, '<PAD>': 7}
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # 训练数据（简化格式：文本和对应的标签）
    train_data = [
        {
            "text": "John works at Google",
            "tags": ["B-PER", "O", "O", "B-ORG"]
        },
        {
            "text": "Apple is in California",
            "tags": ["B-ORG", "O", "O", "B-LOC"]
        },
        {
            "text": "Mary loves Paris",
            "tags": ["B-PER", "O", "B-LOC"]
        }
    ]
    
    # 简单的词汇表
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix.get(w.lower(), to_ix['<UNK>']) for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    
    # 构建词汇表
    for data in train_data:
        for word in data["text"].split():
            if word.lower() not in word_to_ix:
                word_to_ix[word.lower()] = len(word_to_ix)
    
    print(f"词汇表大小: {len(word_to_ix)}")
    
    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModel(vocab_size=len(word_to_ix), tag_to_ix=tag_to_ix)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n训练NER模型...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        for data in train_data:
            words = data["text"].split()
            tags = data["tags"]
            
            # 准备数据
            input_seq = prepare_sequence(words, word_to_ix).unsqueeze(0).to(device)
            target_tags = [tag_to_ix[t] for t in tags]
            target = torch.tensor(target_tags, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_seq)  # (1, seq_len, num_tags)
            outputs = outputs.squeeze(0)  # (seq_len, num_tags)
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # 测试
    print("\nNER识别结果:")
    model.eval()
    test_sentences = [
        "John loves Microsoft",
        "Mary visits California"
    ]
    
    with torch.no_grad():
        for sentence in test_sentences:
            words = sentence.split()
            input_seq = prepare_sequence(words, word_to_ix).unsqueeze(0).to(device)
            outputs = model(input_seq)
            preds = torch.argmax(outputs, dim=2).squeeze(0)
            
            print(f"\n句子: {sentence}")
            for i, word in enumerate(words):
                tag = ix_to_tag[preds[i].item()]
                if tag != 'O':
                    print(f"  {word} -> {tag}")
    
    print("\nNER模型训练完成!\n")


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 NLP实战项目 - Week 4 Day 2")
    print("=" * 60 + "\n")
    
    # 第1部分：情感分类
    train_sentiment_classifier()
    
    # 第2部分：词向量训练
    train_word_embeddings()
    
    # 第3部分：NER模型
    train_ner_model()
    
    print("\n" + "=" * 60)
    print("✅ 所有NLP任务完成!")
    print("=" * 60)
