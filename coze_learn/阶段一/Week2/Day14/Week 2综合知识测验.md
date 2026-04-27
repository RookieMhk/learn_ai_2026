# Week 2 综合知识测验

## 测验说明
- **题目数量**：20题（选择题15题，填空题3题，编程题2题）
- **覆盖范围**：监督学习、优化算法、模型评估、特征工程、算法复杂度、AI工具
- **建议用时**：45-60分钟
- **评分标准**：选择题每题5分，填空题每题5分，编程题每题10分，总分100分

---

## 一、选择题（每题5分，共75分）

### 1. 线性回归模型的基本假设是什么？
A. 特征与目标变量呈线性关系，误差项独立同分布且服从正态分布  
B. 特征与目标变量呈指数关系，误差项相互依赖  
C. 特征与目标变量呈对数关系，误差项服从均匀分布  
D. 特征与目标变量呈多项式关系，误差项具有自相关性  

**答案**：A  
**知识点**：监督学习 - 线性回归  
**解析**：线性回归的核心假设包括线性关系、误差项独立同分布(i.i.d)、正态分布、同方差性等，这些是OLS估计的无偏性和有效性的基础。

### 2. 逻辑回归输出的值域范围是多少？
A. (-∞, +∞)  
B. [0, 1]  
C. [-1, 1]  
D. (0, +∞)  

**答案**：B  
**知识点**：监督学习 - 逻辑回归  
**解析**：逻辑回归通过Sigmoid函数将线性组合映射到(0,1)区间，表示概率值，严格来说是开区间，但常表述为[0,1]。

### 3. 决策树使用基尼不纯度作为分割准则时，其计算公式是？
A. Gini(p) = 1 - Σ(p_i²)  
B. Gini(p) = Σ(p_i log p_i)  
C. Gini(p) = Σ(p_i(1-p_i))  
D. Gini(p) = -Σ(p_i log p_i)  

**答案**：A  
**知识点**：监督学习 - 决策树  
**解析**：基尼不纯度衡量从数据集中随机选取两个样本类别不一致的概率，值域[0,0.5]，值越小纯度越高。

### 4. 随机森林属于哪种集成学习策略？
A. Boosting  
B. Bagging  
C. Stacking  
D. Blending  

**答案**：B  
**知识点**：监督学习 - 随机森林  
**解析**：随机森林通过自助采样(Bootstrap)构建多个决策树，属于Bagging（Bootstrap Aggregating）方法。

### 5. 支持向量机中，软间隔允许什么？
A. 允许一些样本被误分类，以提高模型泛化能力  
B. 要求所有样本都被正确分类  
C. 只允许支持向量被误分类  
D. 不允许任何误分类  

**答案**：A  
**知识点**：监督学习 - 支持向量机  
**解析**：软间隔通过引入松弛变量和惩罚参数C，允许部分样本在间隔内或被误分类，避免过拟合。

### 6. 梯度下降算法中，学习率η过大会导致什么？
A. 收敛速度变慢  
B. 在最优解附近震荡  
C. 可能越过最优解导致发散  
D. 陷入局部最优  

**答案**：C  
**知识点**：优化算法 - 梯度下降  
**解析**：学习率过大时，参数更新步长过大，可能越过损失函数最小值，导致震荡甚至发散。

### 7. Adam优化器相比SGD的主要优势是什么？
A. 自动调整学习率，适应不同参数  
B. 不需要计算梯度  
C. 保证收敛到全局最优  
D. 不需要初始化参数  

**答案**：A  
**知识点**：优化算法 - Adam  
**解析**：Adam结合动量（一阶矩）和自适应学习率（二阶矩），为每个参数维护独立的学习率，适应不同特征尺度。

### 8. 二分类任务中，精确率的计算公式是？
A. TP / (TP + FP)  
B. TP / (TP + FN)  
C. (TP + TN) / (TP + TN + FP + FN)  
D. (TP - FP) / (TP + FP)  

**答案**：A  
**知识点**：模型评估 - 精确率  
**解析**：精确率关注预测为正的样本中实际为正的比例，衡量预测的准确性。

### 9. 召回率主要关注的是什么？
A. 预测为正的样本中有多少是真正的正例  
B. 实际为正的样本中有多少被预测为正  
C. 所有预测正确的样本比例  
D. 预测为负的样本中有多少是真正的负例  

**答案**：B  
**知识点**：模型评估 - 召回率  
**解析**：召回率关注实际为正的样本中被正确预测的比例，衡量模型发现正例的能力。

### 10. ROC曲线的横纵坐标分别是什么？
A. 横坐标：假正例率(FPR)，纵坐标：真正例率(TPR)  
B. 横坐标：真正例率(TPR)，纵坐标：假正例率(FPR)  
C. 横坐标：精确率，纵坐标：召回率  
D. 横坐标：召回率，纵坐标：精确率  

**答案**：A  
**知识点**：模型评估 - ROC曲线  
**解析**：ROC曲线以假正例率(FPR)为横轴，真正例率(TPR)为纵轴，反映分类器在不同阈值下的性能。

### 11. 特征标准化（Z-score标准化）的公式是？
A. z = (x - μ) / σ  
B. z = (x - min) / (max - min)  
C. z = x / max  
D. z = log(x + 1)  

**答案**：A  
**知识点**：特征工程 - 标准化  
**解析**：标准化使特征均值为0，标准差为1，适用于基于距离的算法如SVM、KNN。

### 12. 独热编码（One-Hot Encoding）的主要缺点是什么？
A. 增加特征维度，可能造成维度灾难  
B. 丢失类别间的顺序信息  
C. 对异常值敏感  
D. 计算复杂度高  

**答案**：A  
**知识点**：特征工程 - 独热编码  
**解析**：独热编码将分类变量转换为二进制向量，类别数多时会大幅增加特征维度，可能需配合特征选择。

### 13. 主成分分析（PCA）的主要目标是什么？
A. 最大化保留数据的方差  
B. 最小化重构误差  
C. 寻找数据聚类中心  
D. 最大化特征相关性  

**答案**：A  
**知识点**：特征工程 - PCA  
**解析**：PCA通过正交变换找到方差最大的投影方向，用较少的主成分解释大部分数据变异。

### 14. 算法时间复杂度O(n log n)通常对应哪种排序算法？
A. 冒泡排序  
B. 快速排序  
C. 选择排序  
D. 插入排序  

**答案**：B  
**知识点**：算法复杂度 - 时间复杂度  
**解析**：快速排序、归并排序等高效排序算法的平均时间复杂度为O(n log n)。

### 15. 动态规划解决问题需要满足什么条件？
A. 最优子结构和重叠子问题  
B. 贪心选择和最优子结构  
C. 分治和递归  
D. 迭代和递推  

**答案**：A  
**知识点**：算法复杂度 - 动态规划  
**解析**：动态规划要求问题具有最优子结构（整体最优解包含子问题最优解）和重叠子问题（子问题被重复计算）。

---

## 二、填空题（每题5分，共15分）

### 16. 在机器学习中，______损失函数用于回归任务，______损失函数用于二分类任务。

**答案**：均方误差（MSE）、交叉熵（Cross-Entropy）  
**知识点**：损失函数  
**解析**：MSE衡量预测值与真实值的平方差，适用于回归；交叉熵衡量概率分布差异，适用于分类。

### 17. 随机森林中，每棵决策树在训练时使用的数据是通过______采样得到的，这增加了模型的多样性。

**答案**：自助（Bootstrap）  
**知识点**：集成学习 - Bagging  
**解析**：Bootstrap采样从原始数据集中有放回地抽取n个样本，每个树使用不同的训练子集。

### 18. 在特征工程中，______技术可以将高维特征映射到低维空间，同时保留主要信息。

**答案**：降维（Dimensionality Reduction）或主成分分析（PCA）  
**知识点**：特征工程 - 降维  
**解析**：降维技术如PCA、t-SNE、UMAP等可以减少特征数量，缓解维度灾难，提高计算效率。

---

## 三、编程题（每题10分，共20分）

### 19. 实现梯度下降求解线性回归

```python
import numpy as np

def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    """
    使用梯度下降求解线性回归参数
    
    参数:
    X: 特征矩阵 (n_samples, n_features)
    y: 目标向量 (n_samples,)
    learning_rate: 学习率
    epochs: 迭代次数
    
    返回:
    w: 权重向量
    b: 偏置项
    losses: 每轮损失值列表
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # 初始化权重
    b = 0  # 初始化偏置
    losses = []
    
    for epoch in range(epochs):
        # 预测值
        y_pred = X @ w + b
        
        # 计算损失（均方误差）
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # 计算梯度
        dw = -(2/n_samples) * X.T @ (y - y_pred)
        db = -(2/n_samples) * np.sum(y - y_pred)
        
        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 每100轮打印一次损失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b, losses

# 测试示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_w = np.array([1.5, -2.0])
    true_b = 0.5
    y = X @ true_w + true_b + np.random.randn(100) * 0.1
    
    # 运行梯度下降
    w, b, losses = gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=500)
    print(f"\n真实参数: w={true_w}, b={true_b}")
    print(f"估计参数: w={w}, b={b:.4f}")
```

**评分要点**：
1. 正确实现梯度计算（dw, db）得3分
2. 正确更新参数得2分  
3. 记录并返回损失值得2分
4. 提供测试示例得2分
5. 代码清晰、注释完整得1分

### 20. 实现特征标准化和模型评估Pipeline

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def build_ml_pipeline():
    """
    构建完整的机器学习Pipeline：数据加载 → 标准化 → 模型训练 → 评估
    """
    # 1. 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 5. 预测与评估
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # 6. 输出结果
    print("机器学习Pipeline评估结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # 特征重要性（系数绝对值）
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 5重要特征:")
    print(importance.head(5).to_string(index=False))
    
    return metrics, importance

if __name__ == "__main__":
    metrics, importance = build_ml_pipeline()
```

**评分要点**：
1. 正确加载数据并划分数据集得2分
2. 正确实现特征标准化得2分
3. 训练模型并进行预测得2分
4. 计算至少4种评估指标得2分
5. 输出特征重要性得1分
6. 代码完整可运行得1分

---

## 四、答案与解析汇总

| 题号 | 正确答案 | 知识点 | 难度 |
|------|----------|--------|------|
| 1 | A | 线性回归假设 | ★★ |
| 2 | B | 逻辑回归输出 | ★ |
| 3 | A | 决策树基尼指数 | ★★ |
| 4 | B | 随机森林集成策略 | ★★ |
| 5 | A | SVM软间隔 | ★★ |
| 6 | C | 梯度下降学习率 | ★★ |
| 7 | A | Adam优化器优势 | ★★★ |
| 8 | A | 精确率计算 | ★★ |
| 9 | B | 召回率定义 | ★★ |
| 10 | A | ROC曲线坐标 | ★★★ |
| 11 | A | 特征标准化公式 | ★★ |
| 12 | A | 独热编码缺点 | ★★ |
| 13 | A | PCA目标 | ★★ |
| 14 | B | 时间复杂度对应算法 | ★★★ |
| 15 | A | 动态规划条件 | ★★★ |
| 16 | 均方误差、交叉熵 | 损失函数应用 | ★★ |
| 17 | 自助（Bootstrap） | Bagging采样 | ★★ |
| 18 | 降维/PCA | 特征降维技术 | ★★ |

**编程题评分标准**：
- 19题：梯度计算正确、参数更新正确、损失记录完整、测试示例合理
- 20题：Pipeline完整、标准化正确、评估指标全面、特征重要性分析

---

## 五、学习建议

1. **选择题错题分析**：如果错题超过5道，建议重读速查手册对应章节
2. **填空题巩固**：记忆关键术语和公式，理解其应用场景
3. **编程题实践**：独立完成代码编写，调试运行确保理解每个步骤
4. **综合应用**：结合房价预测项目，将测验知识点应用到实际问题中

**进阶挑战**：
- 修改编程题19，尝试添加动量（Momentum）或自适应学习率
- 扩展编程题20，添加特征选择步骤（如递归特征消除RFE）
- 使用Week2所学，在房价预测项目中尝试不同特征工程方法

---

**祝学习进步！完成后可对照评估工具检测掌握程度。**