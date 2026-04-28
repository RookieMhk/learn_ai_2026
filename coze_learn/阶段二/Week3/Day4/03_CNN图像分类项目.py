"""
CNN图像分类实战项目
使用 PyTorch + CIFAR-10 数据集实现图像分类

项目结构：
1. 数据加载与预处理
2. CNN模型定义
3. 数据增强
4. 训练流程
5. 模型评估与可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

# 设置随机种子，保证结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================
# 配置参数
# ============================================================
class Config:
    # 数据路径
    DATA_DIR = './data/cifar10'
    
    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 模型参数
    NUM_CLASSES = 10
    
    # 设备选择
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CIFAR-10 类别名称
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def get_data_transforms():
    """
    定义数据增强策略
    
    训练集：随机裁剪、水平翻转、颜色抖动、归一化
    测试集：只进行归一化
    """
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪，周围补零
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.RandomRotation(15),               # 随机旋转 ±15度
        transforms.ColorJitter(brightness=0.2,        # 颜色抖动
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),                        # 转换为张量
        transforms.Normalize(                         # 归一化到 [-1, 1]
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 测试时不进行增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, test_transform


def load_cifar10():
    """
    加载 CIFAR-10 数据集
    """
    train_transform, test_transform = get_data_transforms()
    
    # 下载并加载训练集
    train_dataset = datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数: {Config.NUM_CLASSES}")
    
    return train_loader, test_loader


# ============================================================
# 2. CNN模型定义
# ============================================================
class SimpleCNN(nn.Module):
    """
    简单的CNN模型
    
    架构：Conv - Pool - Conv - Pool - Conv - FC - FC
    
    为什么这样设计？
    - 浅层：提取低级特征（边缘、纹理）
    - 深层：提取高级特征（形状、物体）
    - 逐步降维：H×W 减小，C（通道）增加
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积块
        # 输入: 3×32×32 → 输出: 32×16×16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块
        # 输入: 32×16×16 → 输出: 64×8×8
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块
        # 输入: 64×8×8 → 输出: 128×4×4
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        # 全局平均池化 + 全连接分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class ResNetClassifier(nn.Module):
    """
    使用预训练的 ResNet18 作为 backbone
    迁移学习：在 ImageNet 预训练权重基础上微调
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        # 加载预训练的 ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 修改最后一层，适配 CIFAR-10 的 10 个类别
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================
# 3. 训练与评估函数
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'    Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于混淆矩阵
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_preds, all_targets


def plot_training_history(history):
    """
    绘制训练历史曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['test_acc'], label='Test Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("训练曲线已保存至 training_history.png")


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("混淆矩阵已保存至 confusion_matrix.png")


def visualize_predictions(model, test_loader, device, num_images=10):
    """
    可视化预测结果
    """
    model.eval()
    
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 预测
    outputs = model(images)
    _, predicted = outputs.max(1)
    
    # 反归一化用于显示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_np = images.cpu() * std + mean
    images_np = np.clip(images_np.numpy(), 0, 1)
    
    # 显示结果
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(images_np[i].transpose(1, 2, 0))
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {Config.CLASSES[labels[i]]}\n'
                         f'Pred: {Config.CLASSES[predicted[i]]}',
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print("预测结果已保存至 predictions.png")


# ============================================================
# 4. 主训练流程
# ============================================================
def main():
    """
    主函数：完整的训练流程
    """
    print("=" * 60)
    print("CNN图像分类实战 - CIFAR-10")
    print("=" * 60)
    print(f"使用设备: {Config.DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 加载数据
    print("步骤1: 加载 CIFAR-10 数据集...")
    train_loader, test_loader = load_cifar10()
    print()
    
    # 2. 创建模型
    print("步骤2: 创建 CNN 模型...")
    model = SimpleCNN(num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # 如果使用 ResNet，可以切换：
    # model = ResNetClassifier(num_classes=Config.NUM_CLASSES, pretrained=True)
    
    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print()
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=Config.LEARNING_RATE,
                           weight_decay=Config.WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 4. 训练循环
    print("步骤3: 开始训练...")
    print("-" * 60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()
        
        print(f'Epoch {epoch+1}/{Config.EPOCHS}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # 评估
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, Config.DEVICE
        )
        
        # 学习率调整
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print(f'  用时: {epoch_time:.1f}s')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  ✓ 保存最佳模型，准确率: {test_acc:.2f}%')
        
        print()
    
    # 5. 最终评估
    print("=" * 60)
    print("训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("=" * 60)
    
    # 6. 绘制结果
    print("\n绘制训练曲线...")
    plot_training_history(history)
    
    print("\n加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, preds, targets = evaluate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f'最终测试准确率: {test_acc:.2f}%')
    
    # 混淆矩阵
    print("\n绘制混淆矩阵...")
    plot_confusion_matrix(targets, preds, Config.CLASSES)
    
    # 可视化预测
    print("\n可视化预测结果...")
    visualize_predictions(model, test_loader, Config.DEVICE)
    
    # 7. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'history': history
    }, 'cnn_classifier.pth')
    print("\n模型已保存至 cnn_classifier.pth")


# ============================================================
# 5. 独立模块：可视化函数
# ============================================================
def visualize_feature_maps(model, image, layer_names=None):
    """
    可视化卷积层的特征图
    
    参数:
        model: 训练好的模型
        image: 单张输入图像 (C×H×W)
        layer_names: 要可视化的层名称列表
    """
    model.eval()
    
    # 注册hook来获取中间层输出
    feature_maps = {}
    
    def hook_fn(module, input, output, name):
        feature_maps[name] = output.detach()
    
    # 注册hook
    hooks = []
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )
    
    # 前向传播
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    
    # 移除hook
    for hook in hooks:
        hook.remove()
    
    # 可视化
    for layer_name, fmap in feature_maps.items():
        if len(fmap.shape) == 4:
            n_channels = min(16, fmap.shape[1])  # 最多显示16个通道
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            axes = axes.flatten()
            
            for i in range(n_channels):
                fmap_np = fmap[0, i].cpu().numpy()
                axes[i].imshow(fmap_np, cmap='viridis')
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps: {layer_name}')
            plt.tight_layout()
            plt.savefig(f'feature_map_{layer_name}.png', dpi=100)
            plt.show()


def visualize_conv_filters(model, layer_name):
    """
    可视化卷积核
    
    参数:
        model: 训练好的模型
        layer_name: 卷积层名称
    """
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data
            
            # 显示前16个卷积核
            n_filters = min(16, weights.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            axes = axes.flatten()
            
            for i in range(n_filters):
                # 如果是RGB卷积核，取第一个输入通道
                if weights.shape[1] == 3:
                    w = weights[i].cpu().numpy()  # (3, H, W)
                    w = w.transpose(1, 2, 0)  # (H, W, 3)
                    # 归一化到 [0, 1]
                    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
                else:
                    w = weights[i, 0].cpu().numpy()
                
                axes[i].imshow(w, cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Filter {i}')
            
            plt.suptitle(f'Conv Filters: {layer_name}')
            plt.tight_layout()
            plt.savefig(f'conv_filters_{layer_name}.png', dpi=100)
            plt.show()


# ============================================================
# 6. 快捷启动
# ============================================================
if __name__ == '__main__':
    # 运行完整训练流程
    main()
    
    # 如果只需要测试训练好的模型，取消下面注释：
    # model = SimpleCNN(num_classes=10)
    # model.load_state_dict(torch.load('best_model.pth'))
    # visualize_feature_maps(model, test_image)
