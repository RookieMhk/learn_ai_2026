#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1 Day 6: 线性回归模型评估工具
AI学习计划 - 第一阶段：基础夯实 - 机器学习项目实战

本文件提供专业的线性回归模型评估工具，包含：
1. 多种性能指标计算（MSE、RMSE、MAE、R²、调整R²）
2. 残差分析可视化
3. 预测误差分布分析
4. 模型假设检验（线性、独立性、同方差性、正态性）
5. 学习曲线与验证曲线

学习目标：
1. 掌握机器学习模型评估的关键指标
2. 学习如何诊断模型问题与改进方向
3. 理解模型假设检验的重要性
4. 掌握专业的数据可视化分析方法

时间投入：约1-2小时
前置要求：已安装NumPy、Matplotlib、Scipy、Scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 60)
print("Week 1 Day 6: 线性回归模型评估工具")
print("=" * 60)

class LinearRegressionEvaluator:
    """
    线性回归模型评估器
    
    提供完整的模型评估功能，包括：
    - 性能指标计算
    - 残差分析
    - 模型诊断
    - 可视化报告
    """
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        """
        初始化评估器
        
        参数：
        model: 训练好的线性回归模型（需有predict方法）
        X_train: 训练特征（已包含偏置项）
        y_train: 训练标签
        X_test: 测试特征（已包含偏置项）
        y_test: 测试标签
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # 计算预测值
        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred = self.model.predict(X_test)
        
        # 计算残差
        self.train_residuals = y_train - self.y_train_pred
        self.test_residuals = y_test - self.y_test_pred
        
        # 存储评估结果
        self.metrics = {}
    
    def compute_all_metrics(self):
        """
        计算所有性能指标
        
        返回：
        metrics_dict: 包含所有指标的字典
        """
        metrics = {}
        
        # 训练集指标
        metrics['train_mse'] = mean_squared_error(self.y_train, self.y_train_pred)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['train_mae'] = mean_absolute_error(self.y_train, self.y_train_pred)
        metrics['train_r2'] = r2_score(self.y_train, self.y_train_pred)
        
        # 测试集指标
        metrics['test_mse'] = mean_squared_error(self.y_test, self.y_test_pred)
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        metrics['test_mae'] = mean_absolute_error(self.y_test, self.y_test_pred)
        metrics['test_r2'] = r2_score(self.y_test, self.y_test_pred)
        
        # 计算调整R²（考虑特征数量）
        n_train = self.X_train.shape[0]
        n_test = self.X_test.shape[0]
        p = self.X_train.shape[1] - 1  # 特征数量（排除偏置项）
        
        metrics['train_adj_r2'] = 1 - (1 - metrics['train_r2']) * (n_train - 1) / (n_train - p - 1)
        metrics['test_adj_r2'] = 1 - (1 - metrics['test_r2']) * (n_test - 1) / (n_test - p - 1)
        
        # 计算平均绝对百分比误差（MAPE）
        # 避免除零错误
        epsilon = 1e-10
        metrics['train_mape'] = np.mean(np.abs(self.train_residuals) / (np.abs(self.y_train) + epsilon)) * 100
        metrics['test_mape'] = np.mean(np.abs(self.test_residuals) / (np.abs(self.y_test) + epsilon)) * 100
        
        self.metrics = metrics
        return metrics
    
    def print_metrics_report(self):
        """
        打印详细的指标报告
        """
        if not self.metrics:
            self.compute_all_metrics()
        
        print("\n" + "=" * 60)
        print("线性回归模型评估报告")
        print("=" * 60)
        
        print("\n📊 性能指标对比:")
        print("-" * 40)
        print(f"{'指标':<15} {'训练集':<15} {'测试集':<15}")
        print("-" * 40)
        print(f"{'MSE':<15} {self.metrics['train_mse']:.6f} {self.metrics['test_mse']:.6f}")
        print(f"{'RMSE':<15} {self.metrics['train_rmse']:.6f} {self.metrics['test_rmse']:.6f}")
        print(f"{'MAE':<15} {self.metrics['train_mae']:.6f} {self.metrics['test_mae']:.6f}")
        print(f"{'R²':<15} {self.metrics['train_r2']:.6f} {self.metrics['test_r2']:.6f}")
        print(f"{'调整R²':<15} {self.metrics['train_adj_r2']:.6f} {self.metrics['test_adj_r2']:.6f}")
        print(f"{'MAPE(%)':<15} {self.metrics['train_mape']:.2f} {self.metrics['test_mape']:.2f}")
        
        # 过拟合/欠拟合分析
        print("\n🔍 模型诊断:")
        print("-" * 40)
        mse_gap = self.metrics['test_mse'] - self.metrics['train_mse']
        r2_gap = self.metrics['train_r2'] - self.metrics['test_r2']
        
        if mse_gap > 0.1:
            print("⚠️  可能存在过拟合：测试集MSE显著高于训练集")
        elif mse_gap < -0.05:
            print("⚠️  可能存在欠拟合：训练集MSE高于测试集")
        else:
            print("✅ 模型泛化能力良好：训练集与测试集性能接近")
        
        if r2_gap > 0.1:
            print(f"⚠️  R²下降明显：训练集R²比测试集高{r2_gap:.3f}")
        
        # 误差分析
        print(f"\n📈 误差统计:")
        print(f"   训练集残差均值: {np.mean(self.train_residuals):.6f} (应接近0)")
        print(f"   测试集残差均值: {np.mean(self.test_residuals):.6f} (应接近0)")
        print(f"   训练集残差标准差: {np.std(self.train_residuals):.6f}")
        print(f"   测试集残差标准差: {np.std(self.test_residuals):.6f}")
    
    def plot_comprehensive_diagnostics(self):
        """
        绘制综合诊断图（4x4子图）
        """
        if not self.metrics:
            self.compute_all_metrics()
        
        fig = plt.figure(figsize=(16, 16))
        
        # 1. 真实值 vs 预测值散点图
        ax1 = plt.subplot(4, 4, 1)
        ax1.scatter(self.y_train, self.y_train_pred, alpha=0.6, label='训练集', color='blue')
        ax1.scatter(self.y_test, self.y_test_pred, alpha=0.6, label='测试集', color='green')
        
        # 绘制完美预测线
        min_val = min(self.y_train.min(), self.y_test.min())
        max_val = max(self.y_train.max(), self.y_test.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('真实值 vs 预测值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差 vs 预测值图（检查同方差性）
        ax2 = plt.subplot(4, 4, 2)
        ax2.scatter(self.y_train_pred, self.train_residuals, alpha=0.6, label='训练集', color='blue')
        ax2.scatter(self.y_test_pred, self.test_residuals, alpha=0.6, label='测试集', color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        ax2.set_title('残差 vs 预测值（检查同方差性）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差直方图（检查正态性）
        ax3 = plt.subplot(4, 4, 3)
        ax3.hist(self.train_residuals, bins=30, alpha=0.6, label='训练集', color='blue', density=True)
        ax3.hist(self.test_residuals, bins=30, alpha=0.6, label='测试集', color='green', density=True)
        
        # 添加正态分布曲线
        x_norm = np.linspace(min(self.train_residuals.min(), self.test_residuals.min()),
                            max(self.train_residuals.max(), self.test_residuals.max()), 100)
        train_norm = stats.norm.pdf(x_norm, np.mean(self.train_residuals), np.std(self.train_residuals))
        test_norm = stats.norm.pdf(x_norm, np.mean(self.test_residuals), np.std(self.test_residuals))
        
        ax3.plot(x_norm, train_norm, 'b-', linewidth=2, alpha=0.8)
        ax3.plot(x_norm, test_norm, 'g-', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('残差')
        ax3.set_ylabel('密度')
        ax3.set_title('残差分布（检查正态性）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q图（正态性检验）
        ax4 = plt.subplot(4, 4, 4)
        stats.probplot(self.train_residuals, dist="norm", plot=ax4)
        ax4.get_lines()[0].set_markerfacecolor('blue')
        ax4.get_lines()[0].set_markeredgecolor('blue')
        ax4.get_lines()[0].set_markersize(4)
        ax4.get_lines()[1].set_color('red')
        ax4.get_lines()[1].set_linewidth(2)
        ax4.set_title('训练集残差Q-Q图')
        ax4.grid(True, alpha=0.3)
        
        # 5. 误差分布箱线图
        ax5 = plt.subplot(4, 4, 5)
        error_data = [self.train_residuals, self.test_residuals]
        bp = ax5.boxplot(error_data, labels=['训练集', '测试集'], patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax5.set_ylabel('残差')
        ax5.set_title('误差分布箱线图')
        ax5.grid(True, alpha=0.3)
        
        # 6. 累积残差图
        ax6 = plt.subplot(4, 4, 6)
        sorted_train_indices = np.argsort(self.y_train_pred)
        sorted_test_indices = np.argsort(self.y_test_pred)
        
        cum_train_residuals = np.cumsum(self.train_residuals[sorted_train_indices])
        cum_test_residuals = np.cumsum(self.test_residuals[sorted_test_indices])
        
        ax6.plot(range(len(cum_train_residuals)), cum_train_residuals, 
                label='训练集', color='blue', linewidth=2)
        ax6.plot(range(len(cum_test_residuals)), cum_test_residuals, 
                label='测试集', color='green', linewidth=2)
        
        ax6.set_xlabel('样本索引（按预测值排序）')
        ax6.set_ylabel('累积残差')
        ax6.set_title('累积残差图')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 学习曲线（需要额外计算）
        ax7 = plt.subplot(4, 4, 7)
        try:
            # 简化版学习曲线
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []
            
            for size in train_sizes:
                n_samples = int(size * len(self.y_train))
                indices = np.random.choice(len(self.y_train), n_samples, replace=False)
                
                X_subset = self.X_train[indices]
                y_subset = self.y_train[indices]
                
                # 重新训练模型（简化版）
                # 注意：这里使用正规方程快速计算
                weights = np.linalg.pinv(X_subset.T @ X_subset) @ X_subset.T @ y_subset
                y_subset_pred = X_subset @ weights
                
                train_scores.append(r2_score(y_subset, y_subset_pred))
                
                # 在完整测试集上评估
                y_test_pred_sub = self.X_test @ weights
                test_scores.append(r2_score(self.y_test, y_test_pred_sub))
            
            ax7.plot(train_sizes * 100, train_scores, 'o-', label='训练集R²', color='blue')
            ax7.plot(train_sizes * 100, test_scores, 's-', label='测试集R²', color='green')
            ax7.set_xlabel('训练集比例 (%)')
            ax7.set_ylabel('R²分数')
            ax7.set_title('学习曲线')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        except Exception as e:
            ax7.text(0.5, 0.5, f'学习曲线计算失败:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('学习曲线')
        
        # 8. 特征重要性（对于单特征，展示权重）
        ax8 = plt.subplot(4, 4, 8)
        if hasattr(self.model, 'weights'):
            weights = self.model.weights
            if len(weights) <= 10:  # 只显示少量特征
                features = [f'w{i}' for i in range(len(weights))]
                colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
                
                bars = ax8.bar(features, np.abs(weights), color=colors)
                ax8.set_xlabel('参数')
                ax8.set_ylabel('绝对值')
                ax8.set_title('参数重要性（绝对值）')
                
                # 添加数值标签
                for bar, weight in zip(bars, weights):
                    height = bar.get_height()
                    ax8.text(bar.get_x() + bar.get_width()/2., height,
                            f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax8.text(0.5, 0.5, '模型没有weights属性', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('参数重要性')
        
        ax8.grid(True, alpha=0.3)
        
        # 9. 预测误差直方图
        ax9 = plt.subplot(4, 4, 9)
        ax9.hist(self.train_residuals, bins=30, alpha=0.6, label='训练集', color='blue', density=True)
        ax9.hist(self.test_residuals, bins=30, alpha=0.6, label='测试集', color='green', density=True)
        ax9.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax9.set_xlabel('预测误差')
        ax9.set_ylabel('密度')
        ax9.set_title('预测误差分布')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. 残差自相关图（检查独立性）
        ax10 = plt.subplot(4, 4, 10)
        from statsmodels.graphics.tsaplots import plot_acf
        try:
            plot_acf(self.train_residuals, lags=20, ax=ax10, alpha=0.05)
            ax10.set_title('训练集残差自相关图')
        except ImportError:
            ax10.text(0.5, 0.5, '需要statsmodels库', 
                     ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('残差自相关图')
        
        ax10.grid(True, alpha=0.3)
        
        # 11. 残差vs特征图（单特征）
        ax11 = plt.subplot(4, 4, 11)
        if self.X_train.shape[1] == 2:  # 单特征+偏置
            X_feature = self.X_train[:, 1]  # 去掉偏置项
            ax11.scatter(X_feature, self.train_residuals, alpha=0.6, label='训练集', color='blue')
            
            # 添加局部回归线
            from scipy.interpolate import UnivariateSpline
            if len(X_feature) > 10:
                sorted_idx = np.argsort(X_feature)
                spline = UnivariateSpline(X_feature[sorted_idx], 
                                         self.train_residuals[sorted_idx], s=len(X_feature)*2)
                ax11.plot(X_feature[sorted_idx], spline(X_feature[sorted_idx]), 
                         'r-', linewidth=2, label='趋势线')
            
            ax11.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax11.set_xlabel('特征X')
            ax11.set_ylabel('残差')
            ax11.set_title('残差 vs 特征')
            ax11.legend()
        else:
            ax11.text(0.5, 0.5, '仅适用于单特征模型', 
                     ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('残差 vs 特征')
        
        ax11.grid(True, alpha=0.3)
        
        # 12. Cook距离（异常值检测）
        ax12 = plt.subplot(4, 4, 12)
        try:
            # 计算简化版Cook距离
            n = len(self.y_train)
            p = self.X_train.shape[1]
            h = np.diag(self.X_train @ np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T)
            cook_d = (self.train_residuals ** 2) / (p * self.metrics['train_mse']) * (h / (1 - h) ** 2)
            
            ax12.stem(range(n), cook_d, linefmt='b-', markerfmt='bo', basefmt=' ')
            ax12.axhline(y=4/n, color='r', linestyle='--', label=f'阈值={4/n:.3f}')
            ax12.set_xlabel('样本索引')
            ax12.set_ylabel('Cook距离')
            ax12.set_title('Cook距离（异常值检测）')
            ax12.legend()
        except Exception as e:
            ax12.text(0.5, 0.5, f'Cook距离计算失败:\n{str(e)[:50]}...', 
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Cook距离')
        
        ax12.grid(True, alpha=0.3)
        
        # 13. 杠杆值 vs 标准化残差
        ax13 = plt.subplot(4, 4, 13)
        try:
            h = np.diag(self.X_train @ np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T)
            standardized_residuals = self.train_residuals / np.sqrt(self.metrics['train_mse'] * (1 - h))
            
            ax13.scatter(h, standardized_residuals, alpha=0.6, color='blue')
            ax13.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax13.axhline(y=2, color='r', linestyle=':', linewidth=1, label='±2σ')
            ax13.axhline(y=-2, color='r', linestyle=':', linewidth=1)
            ax13.axvline(x=2*p/n, color='g', linestyle='--', label=f'杠杆阈值={2*p/n:.3f}')
            
            ax13.set_xlabel('杠杆值')
            ax13.set_ylabel('标准化残差')
            ax13.set_title('杠杆值 vs 标准化残差')
            ax13.legend()
        except Exception as e:
            ax13.text(0.5, 0.5, '计算失败', 
                     ha='center', va='center', transform=ax13.transAxes)
            ax13.set_title('杠杆值 vs 标准化残差')
        
        ax13.grid(True, alpha=0.3)
        
        # 14. 残差概率图
        ax14 = plt.subplot(4, 4, 14)
        try:
            from scipy.stats import probplot
            _, (slope, intercept, r) = probplot(self.train_residuals, dist="norm", plot=ax14)
            ax14.set_title(f'概率图 (R={r:.3f})')
        except:
            ax14.text(0.5, 0.5, '概率图计算失败', 
                     ha='center', va='center', transform=ax14.transAxes)
            ax14.set_title('概率图')
        
        ax14.grid(True, alpha=0.3)
        
        # 15. 误差累积分布函数
        ax15 = plt.subplot(4, 4, 15)
        sorted_train_errors = np.sort(self.train_residuals)
        sorted_test_errors = np.sort(self.test_residuals)
        
        train_cdf = np.arange(1, len(sorted_train_errors)+1) / len(sorted_train_errors)
        test_cdf = np.arange(1, len(sorted_test_errors)+1) / len(sorted_test_errors)
        
        ax15.plot(sorted_train_errors, train_cdf, label='训练集', color='blue', linewidth=2)
        ax15.plot(sorted_test_errors, test_cdf, label='测试集', color='green', linewidth=2)
        
        ax15.set_xlabel('误差')
        ax15.set_ylabel('累积概率')
        ax15.set_title('误差累积分布函数')
        ax15.legend()
        ax15.grid(True, alpha=0.3)
        
        # 16. 模型性能总结表
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('tight')
        ax16.axis('off')
        
        # 创建总结表格
        table_data = [
            ['指标', '训练集', '测试集'],
            ['MSE', f"{self.metrics['train_mse']:.6f}", f"{self.metrics['test_mse']:.6f}"],
            ['RMSE', f"{self.metrics['train_rmse']:.6f}", f"{self.metrics['test_rmse']:.6f}"],
            ['MAE', f"{self.metrics['train_mae']:.6f}", f"{self.metrics['test_mae']:.6f}"],
            ['R²', f"{self.metrics['train_r2']:.6f}", f"{self.metrics['test_r2']:.6f}"],
            ['调整R²', f"{self.metrics['train_adj_r2']:.6f}", f"{self.metrics['test_adj_r2']:.6f}"],
            ['MAPE%', f"{self.metrics['train_mape']:.2f}", f"{self.metrics['test_mape']:.2f}"]
        ]
        
        table = ax16.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4C72B0')
                    cell.set_text_props(color='white', weight='bold')
                elif i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
        
        plt.suptitle('线性回归模型综合诊断报告', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
    
    def generate_text_report(self):
        """
        生成文本格式的评估报告
        
        返回：
        report: 文本报告字符串
        """
        if not self.metrics:
            self.compute_all_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("线性回归模型评估报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. 性能指标")
        report.append("-" * 40)
        report.append(f"训练集 MSE: {self.metrics['train_mse']:.6f}")
        report.append(f"训练集 RMSE: {self.metrics['train_rmse']:.6f}")
        report.append(f"训练集 MAE: {self.metrics['train_mae']:.6f}")
        report.append(f"训练集 R²: {self.metrics['train_r2']:.6f}")
        report.append(f"训练集 调整R²: {self.metrics['train_adj_r2']:.6f}")
        report.append("")
        report.append(f"测试集 MSE: {self.metrics['test_mse']:.6f}")
        report.append(f"测试集 RMSE: {self.metrics['test_rmse']:.6f}")
        report.append(f"测试集 MAE: {self.metrics['test_mae']:.6f}")
        report.append(f"测试集 R²: {self.metrics['test_r2']:.6f}")
        report.append(f"测试集 调整R²: {self.metrics['test_adj_r2']:.6f}")
        report.append("")
        
        report.append("2. 模型诊断")
        report.append("-" * 40)
        
        # 过拟合/欠拟合分析
        mse_gap = self.metrics['test_mse'] - self.metrics['train_mse']
        r2_gap = self.metrics['train_r2'] - self.metrics['test_r2']
        
        if mse_gap > 0.1:
            report.append("⚠️  警告：可能存在过拟合")
            report.append(f"   测试集MSE比训练集高{mse_gap:.6f}")
        elif mse_gap < -0.05:
            report.append("⚠️  警告：可能存在欠拟合")
            report.append(f"   训练集MSE比测试集高{-mse_gap:.6f}")
        else:
            report.append("✅ 模型泛化能力良好")
        
        if r2_gap > 0.1:
            report.append(f"⚠️  R²下降明显：训练集R²比测试集高{r2_gap:.3f}")
        
        report.append("")
        
        # 残差统计
        report.append("3. 残差分析")
        report.append("-" * 40)
        report.append(f"训练集残差均值: {np.mean(self.train_residuals):.6f} (理论值应为0)")
        report.append(f"训练集残差标准差: {np.std(self.train_residuals):.6f}")
        report.append(f"测试集残差均值: {np.mean(self.test_residuals):.6f} (理论值应为0)")
        report.append(f"测试集残差标准差: {np.std(self.test_residuals):.6f}")
        report.append("")
        
        # 正态性检验
        report.append("4. 正态性检验")
        report.append("-" * 40)
        try:
            _, train_pvalue = stats.shapiro(self.train_residuals)
            _, test_pvalue = stats.shapiro(self.test_residuals)
            
            report.append(f"训练集Shapiro-Wilk检验p值: {train_pvalue:.6f}")
            report.append(f"测试集Shapiro-Wilk检验p值: {test_pvalue:.6f}")
            
            alpha = 0.05
            if train_pvalue > alpha:
                report.append("✅ 训练集残差服从正态分布")
            else:
                report.append("⚠️  训练集残差不服从正态分布")
            
            if test_pvalue > alpha:
                report.append("✅ 测试集残差服从正态分布")
            else:
                report.append("⚠️  测试集残差不服从正态分布")
        except:
            report.append("正态性检验需要足够样本量")
        
        report.append("")
        
        # 改进建议
        report.append("5. 改进建议")
        report.append("-" * 40)
        
        if self.metrics['test_r2'] < 0.7:
            report.append("🔧 建议：模型解释力不足(R²<0.7)")
            report.append("   - 检查特征与目标的关系是否线性")
            report.append("   - 考虑添加多项式特征")
            report.append("   - 检查数据中是否有异常值")
        
        if mse_gap > 0.2:
            report.append("🔧 建议：过拟合风险较高")
            report.append("   - 增加训练数据量")
            report.append("   - 添加正则化（L1/L2）")
            report.append("   - 简化模型复杂度")
        
        if np.abs(np.mean(self.train_residuals)) > 0.1:
            report.append("🔧 建议：残差均值偏离0较多")
            report.append("   - 检查模型是否包含所有相关特征")
            report.append("   - 检查数据预处理是否正确")
        
        return "\n".join(report)

# ============================================================================
# 示例使用代码
# ============================================================================
if __name__ == "__main__":
    print("示例：使用评估工具分析线性回归模型")
    print("-" * 50)
    
    # 生成示例数据
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_regression(n_samples=200, n_features=3, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 添加偏置项（用于评估器）
    X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # 创建评估器
    evaluator = LinearRegressionEvaluator(
        model, X_train_with_bias, y_train, X_test_with_bias, y_test
    )
    
    # 计算指标
    metrics = evaluator.compute_all_metrics()
    
    # 打印报告
    evaluator.print_metrics_report()
    
    # 生成文本报告
    text_report = evaluator.generate_text_report()
    print("\n详细文本报告:")
    print(text_report)
    
    # 保存报告到文件
    with open('outputs/阶段一/Week1/Day6/模型评估报告.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print("\n🎯 评估完成！")
    print("报告已保存至：outputs/阶段一/Week1/Day6/模型评估报告.txt")
    print("运行 evaluator.plot_comprehensive_diagnostics() 可查看可视化报告")

print("\n" + "=" * 60)
print("评估工具加载完成！")
print("=" * 60)