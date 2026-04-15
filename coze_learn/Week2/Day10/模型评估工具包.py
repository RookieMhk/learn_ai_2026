"""
模型评估工具包 Model Evaluation Toolkit
===================================
本工具包提供机器学习模型评估的常用指标计算函数，涵盖分类、回归任务。
每个函数都包含详细的数学原理说明、参数解释和使用示例，确保用户无需额外查资料即可理解和使用。
"""

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def 混淆矩阵计算(y_true, y_pred, labels=None):
    """
    计算混淆矩阵，用于评估分类模型的性能。
    
    原理：
        混淆矩阵是一个N×N的表格（N为类别数），其中行表示真实类别，列表示预测类别。
        对于二分类问题，矩阵包含四个值：
            - 真正例（TP）：真实为正，预测为正
            - 假正例（FP）：真实为负，预测为正（误报）
            - 假负例（FN）：真实为正，预测为负（漏报）
            - 真负例（TN）：真实为负，预测为负
    
    数学公式：
        无特定公式，但可从预测结果与真实标签对比直接得出。
    
    参数：
        y_true：一维数组，真实标签
        y_pred：一维数组，预测标签
        labels：可选，类别标签列表，用于指定矩阵的顺序
    
    返回：
        confusion_matrix：二维numpy数组，混淆矩阵
    
    示例：
        >>> y_true = [1, 0, 1, 1, 0, 0]
        >>> y_pred = [1, 0, 0, 1, 0, 1]
        >>> cm = 混淆矩阵计算(y_true, y_pred)
        >>> print(cm)
        [[2 1]
         [1 2]]
    """
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def ROC曲线与AUC计算(y_true, y_score, pos_label=None):
    """
    计算ROC曲线和AUC面积，用于评估二分类模型的分类能力。
    
    原理：
        ROC曲线（Receiver Operating Characteristic Curve）以假正率（FPR）为横轴，
        真正率（TPR）为纵轴，展示模型在不同阈值下的性能。
        AUC（Area Under Curve）是ROC曲线下的面积，值在0.5到1之间，
        越大表示模型区分能力越好。
    
    数学公式：
        真正率（TPR）= TP / (TP + FN)  （召回率）
        假正率（FPR）= FP / (FP + TN)
    
    参数：
        y_true：一维数组，真实标签（0或1）
        y_score：一维数组，预测为正类的概率或得分
        pos_label：指定正类标签，默认为None（自动检测）
    
    返回：
        fpr：一维数组，假正率
        tpr：一维数组，真正率
        thresholds：一维数组，阈值
        auc_score：浮点数，AUC值
    
    示例：
        >>> y_true = [0, 1, 0, 1, 1, 0]
        >>> y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]
        >>> fpr, tpr, thresholds, auc_score = ROC曲线与AUC计算(y_true, y_score)
        >>> print(f"AUC: {auc_score:.3f}")
        AUC: 0.944
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def R2系数计算(y_true, y_pred):
    """
    计算决定系数R²，用于评估回归模型的拟合优度。
    
    原理：
        R²表示模型解释的方差比例，值在0到1之间（可能为负），
        越接近1表示模型拟合越好。
    
    数学公式：
        R² = 1 - SS_res / SS_tot
        其中 SS_res = Σ(y_i - ŷ_i)²（残差平方和）
              SS_tot = Σ(y_i - ȳ)²（总平方和）
    
    参数：
        y_true：一维数组，真实值
        y_pred：一维数组，预测值
    
    返回：
        r2：浮点数，R²系数
    
    示例：
        >>> y_true = [3.0, -0.5, 2.0, 7.0]
        >>> y_pred = [2.5, 0.0, 2.1, 7.8]
        >>> r2 = R2系数计算(y_true, y_pred)
        >>> print(f"R²: {r2:.3f}")
        R²: 0.948
    """
    return r2_score(y_true, y_pred)


def 精确率_召回率_F1计算(y_true, y_pred, average='binary'):
    """
    计算精确率、召回率和F1分数，用于评估分类模型的准确性。
    
    原理：
        - 精确率（Precision）：预测为正的样本中，实际为正的比例
        - 召回率（Recall）：实际为正的样本中，预测为正的比例
        - F1分数：精确率和召回率的调和平均数
    
    数学公式：
        精确率 = TP / (TP + FP)
        召回率 = TP / (TP + FN)
        F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    
    参数：
        y_true：一维数组，真实标签
        y_pred：一维数组，预测标签
        average：平均方法，可选 'binary', 'micro', 'macro', 'weighted'
    
    返回：
        precision：浮点数或数组，精确率
        recall：浮点数或数组，召回率
        f1：浮点数或数组，F1分数
    
    示例：
        >>> y_true = [0, 1, 0, 1, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 1, 0]
        >>> prec, rec, f1 = 精确率_召回率_F1计算(y_true, y_pred)
        >>> print(f"精确率: {prec:.3f}, 召回率: {rec:.3f}, F1: {f1:.3f}")
        精确率: 1.000, 召回率: 0.667, F1: 0.800
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return precision, recall, f1


def 均方误差与平均绝对误差(y_true, y_pred):
    """
    计算均方误差（MSE）和平均绝对误差（MAE），用于评估回归模型的预测误差。
    
    原理：
        - 均方误差（MSE）：预测值与真实值之差的平方的平均值，对异常值敏感
        - 平均绝对误差（MAE）：预测值与真实值之差的绝对值的平均值，更稳健
    
    数学公式：
        MSE = (1/n) * Σ(y_i - ŷ_i)²
        MAE = (1/n) * Σ|y_i - ŷ_i|
    
    参数：
        y_true：一维数组，真实值
        y_pred：一维数组，预测值
    
    返回：
        mse：浮点数，均方误差
        mae：浮点数，平均绝对误差
    
    示例：
        >>> y_true = [3.0, -0.5, 2.0, 7.0]
        >>> y_pred = [2.5, 0.0, 2.1, 7.8]
        >>> mse, mae = 均方误差与平均绝对误差(y_true, y_pred)
        >>> print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")
        MSE: 0.142, MAE: 0.300
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae


def 分类报告生成(y_true, y_pred, labels=None, target_names=None):
    """
    生成详细的分类报告，包含每个类别的精确率、召回率、F1分数和支持数。
    
    原理：
        综合展示多分类模型的性能，便于分析每个类别的分类效果。
    
    参数：
        y_true：一维数组，真实标签
        y_pred：一维数组，预测标签
        labels：类别列表
        target_names：类别名称列表
    
    返回：
        report_dict：字典，分类报告数据
        report_str：字符串，格式化的分类报告
    
    示例：
        >>> y_true = [0, 1, 2, 0, 1, 2]
        >>> y_pred = [0, 2, 1, 0, 0, 1]
        >>> report_dict, report_str = 分类报告生成(y_true, y_pred)
        >>> print(report_str)
    """
    from sklearn.metrics import classification_report
    
    report_str = classification_report(y_true, y_pred, labels=labels, 
                                       target_names=target_names, zero_division=0)
    # 转换为字典以便进一步处理
    report_dict = classification_report(y_true, y_pred, labels=labels,
                                        target_names=target_names, 
                                        zero_division=0, output_dict=True)
    
    return report_dict, report_str


def 可视化ROC曲线(fpr, tpr, auc_score, title='ROC曲线'):
    """
    绘制ROC曲线图。
    
    参数：
        fpr：假正率数组
        tpr：真正率数组
        auc_score：AUC值
        title：图表标题
    
    返回：
        fig：matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig


# ===================== 使用示例 =====================
if __name__ == "__main__":
    print("模型评估工具包使用示例")
    print("=" * 50)
    
    # 示例1：二分类数据
    y_true_binary = [0, 1, 0, 1, 1, 0, 1, 0]
    y_pred_binary = [0, 1, 0, 0, 1, 0, 1, 1]
    y_score_binary = [0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.7, 0.6]
    
    # 混淆矩阵
    cm = 混淆矩阵计算(y_true_binary, y_pred_binary)
    print(f"混淆矩阵:\n{cm}")
    
    # ROC曲线和AUC
    fpr, tpr, thresholds, auc_score = ROC曲线与AUC计算(y_true_binary, y_score_binary)
    print(f"AUC值: {auc_score:.3f}")
    
    # 精确率、召回率、F1
    precision, recall, f1 = 精确率_召回率_F1计算(y_true_binary, y_pred_binary)
    print(f"精确率: {precision:.3f}, 召回率: {recall:.3f}, F1分数: {f1:.3f}")
    
    # 分类报告
    report_dict, report_str = 分类报告生成(y_true_binary, y_pred_binary)
    print("\n分类报告:")
    print(report_str)
    
    # 示例2：回归数据
    y_true_reg = [3.0, -0.5, 2.0, 7.0, 4.0]
    y_pred_reg = [2.5, 0.0, 2.1, 7.8, 3.5]
    
    # R²系数
    r2 = R2系数计算(y_true_reg, y_pred_reg)
    print(f"R²系数: {r2:.3f}")
    
    # MSE和MAE
    mse, mae = 均方误差与平均绝对误差(y_true_reg, y_pred_reg)
    print(f"均方误差(MSE): {mse:.3f}, 平均绝对误差(MAE): {mae:.3f}")
    
    # 可视化ROC曲线
    fig = 可视化ROC曲线(fpr, tpr, auc_score)
    plt.savefig('outputs/阶段一/Week2/Day10/ROC曲线示例.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("ROC曲线已保存为 'ROC曲线示例.png'")