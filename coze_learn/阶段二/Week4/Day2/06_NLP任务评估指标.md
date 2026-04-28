# NLP任务评估指标

> Week 4 Day 2 - 全面理解NLP模型性能度量

---

## 一、分类任务评估指标

### 1.1 准确率 (Accuracy)

**定义**：正确预测的数量占总预测数量的比例。

$$Accuracy = (TP + TN)/(TP + TN + FP + FN)$$

```python
from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.2f}")  # 0.83
```

**适用场景**：
- ✅ 数据类别分布均匀
- ✅ 简单二分类任务
- ❌ 类别不平衡时不可靠

---

### 1.2 混淆矩阵 (Confusion Matrix)

**定义**：展示预测结果的详细分布

```
                预测结果
              Positive  Negative
真实  Positive    TP        FN
值    Negative    FP        TN
```

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[2 0]
#  [1 3]]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()
```

**解读**：
- **TP (True Positive)**：实际正类，预测为正 → 正确
- **TN (True Negative)**：实际负类，预测为负 → 正确
- **FP (False Positive)**：实际负类，预测为正 → 错误
- **FN (False Negative)**：实际正类，预测为负 → 错误

---

### 1.3 精确率 (Precision)

**定义**：预测为正类中，真正为正的比例

$$Precision = TP/(TP + FP) = TP/所有预测为正的$$

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"精确率: {precision:.2f}")  # 0.80
```

**适用场景**：
- 假阳性代价高时（如垃圾邮件过滤，不希望把正常邮件标记为垃圾）
- 需要控制误报的情况

---

### 1.4 召回率 (Recall)

**定义**：实际正类中，被正确预测的比例

$$Recall = TP/(TP + FN) = TP/所有实际为正的$$

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"召回率: {recall:.2f}")  # 0.75
```

**适用场景**：
- 假阴性代价高时（如疾病诊断，不希望漏检）
- 需要确保不遗漏正例的情况

---

### 1.5 F1 Score

**定义**：精确率和召回率的调和平均

$$F1 = 2 × Precision × Recall/(Precision + Recall)$$

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1分数: {f1:.2f}")  # 0.77
```

**特点**：
- 当精确率和召回率差距大时，F1会较低
- 适合评估类别不平衡的任务

---

### 1.6 PR曲线与ROC曲线

**PR曲线（Precision-Recall Curve）**：

```python
from sklearn.metrics import precision_recall_curve, auc

# 获取预测概率
y_proba = [0.9, 0.3, 0.8, 0.6, 0.2, 0.85]

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
pr_auc = auc(recall, precision)

plt.plot(recall, precision)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title(f'PR曲线 (AUC={pr_auc:.2f})')
plt.show()
```

**ROC曲线（Receiver Operating Characteristic）**：

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = roc_auc_score(y_true, y_proba)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title(f'ROC曲线 (AUC={roc_auc:.2f})')
plt.show()
```

**对比**：

| 指标 | 适用场景 | 特点 |
|------|---------|------|
| PR曲线 | 类别不平衡（正例少） | 关注正例 |
| ROC曲线 | 类别较平衡 | 综合考虑正负例 |

---

## 二、序列标注评估指标

### 2.1 Token级 vs Sequence级

**Token级评估**：对每个token独立计算指标

**Sequence级评估**：考虑整个序列的匹配

```python
from seqeval.metrics import classification_report, f1_score as seq_f1

# BIO标注格式
y_true = [['O', 'B-PER', 'I-PER', 'O'],
          ['B-ORG', 'I-ORG', 'O', 'O']]
y_pred = [['O', 'B-PER', 'I-PER', 'O'],
          ['B-ORG', 'I-ORG', 'O', 'O']]

# 详细报告
print(classification_report(y_true, y_pred))

# F1分数
f1 = seq_f1(y_true, y_pred)
print(f"实体F1: {f1:.2f}")
```

**输出示例**：
```
              precision    recall  f1-score   support
        PER       1.00      1.00      1.00         2
        ORG       1.00      1.00      1.00         1
   micro avg       1.00      1.00      1.00         3
   macro avg       1.00      1.00      1.00         3
```

---

### 2.2 实体级别的精确率、召回率、F1

```python
def evaluate_ner(true_entities, pred_entities):
    """
    实体级别的评估
    true_entities: [('John', 'PER'), ('Google', 'ORG'), ...]
    pred_entities: [('John', 'PER'), ('Microsoft', 'ORG'), ...]
    """
    
    # 完全匹配才算正确
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    
    correct = len(true_set & pred_set)
    precision = correct / len(pred_set) if pred_set else 0
    recall = correct / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# 示例
true = [('John', 'PER'), ('Google', 'ORG')]
pred = [('John', 'PER'), ('Microsoft', 'ORG')]

p, r, f = evaluate_ner(true, pred)
print(f"P: {p:.2f}, R: {r:.2f}, F1: {f:.2f}")  # P: 0.50, R: 0.50, F1: 0.50
```

---

## 三、文本生成评估指标

### 3.1 BLEU (Bilingual Evaluation Understudy)

**定义**：衡量生成文本与参考文本的n-gram重叠程度

$$BLEU = BP × ≤ft(Σ_ₙ₌₁N wₙ pₙ)$$

其中BP是简短惩罚，$pₙ$是n-gram精确率

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['The', 'cat', 'is', 'on', 'the', 'mat']

# 计算BLEU
bleu = sentence_bleu(reference, candidate)
print(f"BLEU: {bleu:.4f}")  # 1.0 (完全匹配)

# 部分匹配
candidate2 = ['A', 'cat', 'is', 'on', 'the', 'mat']
bleu2 = sentence_bleu(reference, candidate2)
print(f"BLEU: {bleu2:.4f}")
```

**n-gram级别**：

| BLEU-1 | 单词级精确率 |
| BLEU-2 | 二元组精确率 |
| BLEU-3 | 三元组精确率 |
| BLEU-4 | 四元组精确率 |

```python
# 分别计算不同n-gram的BLEU
weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0)]
for w in weights:
    bleu = sentence_bleu(reference, candidate, weights=w)
    print(f"BLEU-{int(1/sum([x for x in w if x > 0]))}: {bleu:.4f}")
```

---

### 3.2 ROUGE (Recall-Oriented Understudy for Gushing Evaluation)

**定义**：衡量生成文本对参考文本的覆盖率

```python
from rouge import Rouge

reference = "The cat is on the mat"
candidate = "A cat is on the mat"

rouge = Rouge()
scores = rouge.get_scores(candidate, reference)
print(scores)
```

**ROUGE变体**：

| 指标 | 含义 |
|------|------|
| ROUGE-N | N-gram召回率 |
| ROUGE-L | 最长公共子序列 |
| ROUGE-W | 加权最长公共子序列 |
| ROUGE-S | 跳跃二元组 |

---

### 3.3 METEOR

**特点**：考虑了词干匹配和同义词匹配

```python
from nltk.translate.meteor_score import meteor_score

reference = ['The cat is on the mat']
candidate = 'A cat is on the mat'

meteor = meteor_score(reference, candidate)
print(f"METEOR: {meteor:.4f}")
```

---

### 3.4 Perplexity (困惑度)

**定义**：衡量语言模型对文本的预测能力

$$PPL = ≤ft(-1/NΣ_ᵢ₌₁NP(wᵢ|w_₁:ᵢ₋₁))$$

```python
import math

def calculate_perplexity(log_probs, sequence_length):
    """计算困惑度"""
    avg_log_prob = sum(log_probs) / sequence_length
    perplexity = math.exp(-avg_log_prob)
    return perplexity

# 示例
log_probs = [-0.5, -1.0, -0.8, -0.6]
ppl = calculate_perplexity(log_probs, len(log_probs))
print(f"Perplexity: {ppl:.2f}")  # 2.19
```

**解读**：
- 越低越好
- 理想情况PPL=1（完美预测）
- 实际中PPL通常在10-50之间

---

## 四、评估指标选择指南

### 4.1 任务类型 → 指标选择

| 任务 | 推荐指标 | 说明 |
|------|---------|------|
| 文本分类（平衡） | Accuracy | 类别均匀时最直观 |
| 文本分类（不平衡） | F1 / AUC-ROC | 关注少数类 |
| 垃圾邮件检测 | Precision | 不想误杀重要邮件 |
| 疾病诊断 | Recall | 不想漏检 |
| NER | Entity F1 | 实体级别评估 |
| 机器翻译 | BLEU | 行业标准 |
| 文本摘要 | ROUGE | 覆盖率导向 |
| 语言模型 | Perplexity | 预测能力 |

### 4.2 综合评估示例

```python
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def comprehensive_evaluation(y_true, y_pred, y_proba=None):
    """综合评估"""
    print("=" * 50)
    print("综合评估报告")
    print("=" * 50)
    
    # 基本指标
    print(f"\n准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}")
    print(f"精确率 (Precision): {precision_score(y_true, y_pred):.4f}")
    print(f"召回率 (Recall): {recall_score(y_true, y_pred):.4f}")
    print(f"F1分数 (F1): {f1_score(y_true, y_pred):.4f}")
    
    # 详细报告
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred))
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

# 使用示例
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

comprehensive_evaluation(y_true, y_pred)
```

---

## 五、注意事项

### 5.1 类别不平衡问题

```python
from sklearn.metrics import balanced_accuracy_score

# 传统准确率
print(f"准确率: {accuracy_score(y_true, y_pred)}")

# 平衡准确率（对不平衡数据更可靠）
print(f"平衡准确率: {balanced_accuracy_score(y_true, y_pred)}")
```

### 5.2 多标签分类

```python
from sklearn.metrics import roc_auc_score

# 每个样本有多个标签
y_true = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
y_pred = [[0.8, 0.2, 0.9], [0.3, 0.7, 0.6], [0.9, 0.8, 0.1]]

# 多标签AUC
auc = roc_auc_score(y_true, y_pred, average='macro')
print(f"Macro AUC: {auc:.4f}")
```

### 5.3 交叉验证

```python
from sklearn.model_selection import cross_val_score

model = ...
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1分数: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

> 📝 笔记区域
>
> 今天学到的评估指标：
>
> 
>
> 如何选择合适的指标？
>
> 
>
