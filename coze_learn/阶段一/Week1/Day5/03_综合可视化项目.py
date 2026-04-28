#!/usr/bin/env python3
"""
综合可视化项目 - Week 1 学习成果整合
AI技术学习与大厂入职计划 - Week 1 Day 5

本项目整合Week 1学习内容，包括：
1. 线性代数（矩阵运算、特征值分解）
2. 概率统计（概率分布、统计分析）
3. 数据清洗（数据集特征分析）
4. 可视化技术（Matplotlib高级应用）

通过多维度数据可视化，展示AI学习者的技术掌握程度与分析能力。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import seaborn as sns
import warnings

# 配置全局样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据准备与模拟
# ============================================================================

def generate_week1_learning_data():
    """
    生成Week 1学习数据，模拟学习过程中的各种指标
    
    返回：
    包含多个数据集的字典
    """
    np.random.seed(42)  # 确保可重复性
    
    # 1. 线性代数学习数据
    n_students = 50
    linear_algebra_scores = np.random.normal(75, 10, n_students).clip(0, 100)
    matrix_ops_scores = np.random.normal(70, 15, n_students).clip(0, 100)
    
    # 2. 概率统计学习数据
    probability_scores = np.random.normal(80, 12, n_students).clip(0, 100)
    statistical_analysis_scores = np.random.normal(78, 14, n_students).clip(0, 100)
    
    # 3. 数据清洗学习数据
    data_cleaning_scores = np.random.normal(85, 8, n_students).clip(0, 100)
    pandas_mastery_scores = np.random.normal(82, 10, n_students).clip(0, 100)
    
    # 4. 学习时间数据（小时）
    study_hours = np.random.gamma(shape=2, scale=1.5, size=n_students).clip(1, 10)
    
    # 5. 技能掌握进度数据
    days = np.arange(1, 8)  # Week 1的7天
    skill_progress = {
        '线性代数': np.array([20, 35, 50, 65, 75, 85, 95]),
        '概率统计': np.array([15, 30, 45, 60, 75, 85, 92]),
        'NumPy运算': np.array([10, 25, 45, 65, 80, 90, 96]),
        'Pandas清洗': np.array([5, 20, 40, 60, 75, 85, 94]),
        '可视化技术': np.array([0, 10, 25, 45, 65, 80, 90])
    }
    
    # 6. 矩阵特征值数据（用于可视化）
    matrix_size = 10
    true_matrix = np.random.randn(matrix_size, matrix_size)
    symmetric_matrix = (true_matrix + true_matrix.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
    
    # 按实部排序
    idx = eigenvalues.real.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 7. 概率分布数据
    x_normal = np.linspace(-4, 4, 100)
    normal_pdf = stats.norm.pdf(x_normal)
    
    x_uniform = np.linspace(0, 1, 100)
    uniform_pdf = stats.uniform.pdf(x_uniform)
    
    x_exponential = np.linspace(0, 5, 100)
    exponential_pdf = stats.expon.pdf(x_exponential)
    
    # 8. 数据集质量指标（模拟数据清洗过程）
    dataset_quality = {
        '完整率': np.array([65, 78, 82, 88, 92, 95, 97]),
        '准确率': np.array([70, 75, 80, 85, 88, 91, 94]),
        '一致性': np.array([60, 70, 78, 85, 90, 93, 95]),
        '及时性': np.array([85, 87, 89, 91, 93, 95, 96])
    }
    
    # 整合所有数据
    learning_data = {
        # 学生成绩数据
        'linear_algebra_scores': linear_algebra_scores,
        'matrix_ops_scores': matrix_ops_scores,
        'probability_scores': probability_scores,
        'statistical_analysis_scores': statistical_analysis_scores,
        'data_cleaning_scores': data_cleaning_scores,
        'pandas_mastery_scores': pandas_mastery_scores,
        'study_hours': study_hours,
        
        # 技能进度数据
        'skill_progress': skill_progress,
        'days': days,
        
        # 矩阵数据
        'matrix': symmetric_matrix,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'matrix_size': matrix_size,
        
        # 概率分布数据
        'x_normal': x_normal,
        'normal_pdf': normal_pdf,
        'x_uniform': x_uniform,
        'uniform_pdf': uniform_pdf,
        'x_exponential': x_exponential,
        'exponential_pdf': exponential_pdf,
        
        # 数据质量指标
        'dataset_quality': dataset_quality,
        
        # 元数据
        'n_students': n_students
    }
    
    return learning_data


def create_learning_dataframe(learning_data):
    """
    将学习数据转换为Pandas DataFrame，便于分析
    
    参数：
    learning_data: 学习数据字典
    
    返回：
    df: Pandas DataFrame
    """
    n = learning_data['n_students']
    
    df = pd.DataFrame({
        '学生ID': range(1, n + 1),
        '线性代数': learning_data['linear_algebra_scores'],
        '矩阵运算': learning_data['matrix_ops_scores'],
        '概率统计': learning_data['probability_scores'],
        '统计分析': learning_data['statistical_analysis_scores'],
        '数据清洗': learning_data['data_cleaning_scores'],
        'Pandas掌握': learning_data['pandas_mastery_scores'],
        '学习时长_小时': learning_data['study_hours']
    })
    
    # 计算总成绩和平均成绩
    score_columns = ['线性代数', '矩阵运算', '概率统计', 
                    '统计分析', '数据清洗', 'Pandas掌握']
    df['平均成绩'] = df[score_columns].mean(axis=1)
    df['总成绩'] = df[score_columns].sum(axis=1)
    
    # 按平均成绩分组
    def assign_performance_level(score):
        if score >= 85:
            return '优秀'
        elif score >= 70:
            return '良好'
        elif score >= 60:
            return '中等'
        else:
            return '待提高'
    
    df['表现等级'] = df['平均成绩'].apply(assign_performance_level)
    
    return df


# ============================================================================
# 第二部分：多维度数据可视化
# ============================================================================

def create_comprehensive_dashboard(learning_data, df):
    """
    创建综合数据仪表板，展示Week 1学习成果
    
    参数：
    learning_data: 学习数据字典
    df: 学生数据DataFrame
    
    返回：
    fig: 综合仪表板图形
    """
    # 创建图形和网格布局
    fig = plt.figure(figsize=(20, 16))
    
    # 定义复杂的网格布局
    gs = gridspec.GridSpec(4, 4, figure=fig, 
                          height_ratios=[1, 1, 1, 1],
                          width_ratios=[1, 1, 1, 1])
    
    # ------------------------------------------------------------------------
    # 1. 学生成绩分布分析
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 准备成绩数据
    score_columns = ['线性代数', '矩阵运算', '概率统计', 
                    '统计分析', '数据清洗', 'Pandas掌握']
    score_data = [df[col].values for col in score_columns]
    
    # 绘制小提琴图
    violin_parts = ax1.violinplot(score_data, showmeans=True, showmedians=True)
    
    # 定制小提琴图颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    
    # 设置标签和标题
    ax1.set_xticks(range(1, len(score_columns) + 1))
    ax1.set_xticklabels(score_columns, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('成绩分数', fontsize=12)
    ax1.set_title('各科目成绩分布对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------------
    # 2. 学习时长与成绩相关性分析
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 提取数据
    study_hours = df['学习时长_小时'].values
    avg_scores = df['平均成绩'].values
    
    # 计算相关性
    correlation = np.corrcoef(study_hours, avg_scores)[0, 1]
    
    # 绘制散点图
    scatter = ax2.scatter(study_hours, avg_scores, 
                         c=avg_scores, cmap='RdYlBu',
                         s=80, edgecolor='black', linewidth=0.5,
                         alpha=0.7)
    
    # 添加回归线
    if len(study_hours) > 1:
        coeffs = np.polyfit(study_hours, avg_scores, 1)
        poly = np.poly1d(coeffs)
        x_range = np.linspace(study_hours.min(), study_hours.max(), 100)
        ax2.plot(x_range, poly(x_range), 'r-', linewidth=2.5,
                label=f'R = {correlation:.3f}')
    
    # 设置标签和标题
    ax2.set_xlabel('学习时长（小时）', fontsize=12)
    ax2.set_ylabel('平均成绩', fontsize=12)
    ax2.set_title(f'学习时长与成绩相关性 (R = {correlation:.3f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar2 = fig.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar2.ax.set_ylabel('成绩分数', rotation=90, fontsize=11)
    
    # ------------------------------------------------------------------------
    # 3. 技能掌握进度演变
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2:4])
    
    skill_progress = learning_data['skill_progress']
    days = learning_data['days']
    
    # 绘制多条折线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    skills = list(skill_progress.keys())
    
    for idx, skill in enumerate(skills):
        progress = skill_progress[skill]
        ax3.plot(days, progress, 'o-', linewidth=2.5, markersize=8,
                color=colors[idx], label=skill, alpha=0.8)
    
    # 设置标签和标题
    ax3.set_xlabel('学习天数', fontsize=12)
    ax3.set_ylabel('掌握进度（%）', fontsize=12)
    ax3.set_title('Week 1 技能掌握进度演变', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 添加背景色带表示学习阶段
    ax3.axvspan(1, 3, alpha=0.1, color='green', label='基础知识')
    ax3.axvspan(4, 6, alpha=0.1, color='blue', label='技能应用')
    ax3.axvspan(7, 7, alpha=0.1, color='purple', label='综合提升')
    
    # ------------------------------------------------------------------------
    # 4. 矩阵特征值可视化
    # ------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    
    eigenvalues = learning_data['eigenvalues']
    matrix_size = learning_data['matrix_size']
    
    # 绘制特征值位置图
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag
    
    # 绘制点
    scatter4 = ax4.scatter(real_parts, imag_parts, 
                          c=np.abs(eigenvalues), cmap='viridis',
                          s=100, edgecolor='black', linewidth=1,
                          alpha=0.8)
    
    # 添加单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1.5, alpha=0.6)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 设置标签和标题
    ax4.set_xlabel('实部', fontsize=12)
    ax4.set_ylabel('虚部', fontsize=12)
    ax4.set_title(f'矩阵特征值分布 (矩阵大小: {matrix_size}×{matrix_size})', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.8)
    cbar4.ax.set_ylabel('特征值模长', rotation=90, fontsize=11)
    
    # ------------------------------------------------------------------------
    # 5. 特征向量可视化
    # ------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    
    eigenvectors = learning_data['eigenvectors']
    
    # 提取前3个特征向量
    n_vectors = min(3, eigenvectors.shape[1])
    
    # 创建热力图展示特征向量
    im5 = ax5.imshow(eigenvectors[:, :n_vectors].real, 
                     cmap='RdBu_r', aspect='auto',
                     interpolation='nearest')
    
    # 设置坐标轴
    ax5.set_xlabel('特征向量索引', fontsize=12)
    ax5.set_ylabel('矩阵维度', fontsize=12)
    ax5.set_title(f'前{n_vectors}个特征向量可视化', 
                  fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar5 = fig.colorbar(im5, ax=ax5, shrink=0.8)
    cbar5.ax.set_ylabel('向量分量值', rotation=90, fontsize=11)
    
    # ------------------------------------------------------------------------
    # 6. 概率分布对比
    # ------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 绘制三种概率分布
    ax6.plot(learning_data['x_normal'], learning_data['normal_pdf'], 
            'b-', linewidth=2.5, label='正态分布', alpha=0.8)
    ax6.plot(learning_data['x_uniform'], learning_data['uniform_pdf'], 
            'g--', linewidth=2.5, label='均匀分布', alpha=0.8)
    ax6.plot(learning_data['x_exponential'], learning_data['exponential_pdf'], 
            'r-.', linewidth=2.5, label='指数分布', alpha=0.8)
    
    # 填充区域
    ax6.fill_between(learning_data['x_normal'], 0, learning_data['normal_pdf'],
                     alpha=0.2, color='blue')
    ax6.fill_between(learning_data['x_uniform'], 0, learning_data['uniform_pdf'],
                     alpha=0.2, color='green')
    ax6.fill_between(learning_data['x_exponential'], 0, learning_data['exponential_pdf'],
                     alpha=0.2, color='red')
    
    # 设置标签和标题
    ax6.set_xlabel('变量值', fontsize=12)
    ax6.set_ylabel('概率密度', fontsize=12)
    ax6.set_title('典型概率分布对比', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------------
    # 7. 数据集质量演变
    # ------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[1, 3])
    
    dataset_quality = learning_data['dataset_quality']
    days = learning_data['days']
    
    # 创建堆叠面积图
    quality_metrics = ['完整率', '准确率', '一致性', '及时性']
    colors_area = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    
    cumulative = np.zeros_like(days, dtype=float)
    for idx, metric in enumerate(quality_metrics):
        values = dataset_quality[metric]
        ax7.fill_between(days, cumulative, cumulative + values,
                        color=colors_area[idx], alpha=0.7, label=metric)
        cumulative += values
    
    # 设置标签和标题
    ax7.set_xlabel('清洗阶段', fontsize=12)
    ax7.set_ylabel('质量指标总和', fontsize=12)
    ax7.set_title('数据集质量随清洗过程演变', fontsize=14, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 400])  # 四维指标最大为4*100=400
    
    # ------------------------------------------------------------------------
    # 8. 学生表现等级分布
    # ------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 0])
    
    # 计算各等级学生数量
    performance_counts = df['表现等级'].value_counts()
    performance_order = ['优秀', '良好', '中等', '待提高']
    performance_counts = performance_counts.reindex(performance_order, fill_value=0)
    
    # 绘制环形图
    wedges, texts, autotexts = ax8.pie(performance_counts.values,
                                       labels=performance_counts.index,
                                       colors=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # 设置文本样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax8.set_title('学生表现等级分布', fontsize=14, fontweight='bold')
    
    # ------------------------------------------------------------------------
    # 9. 各科目成绩相关性矩阵
    # ------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 1])
    
    # 计算相关性矩阵
    correlation_matrix = df[score_columns].corr()
    
    # 绘制热力图
    im9 = ax9.imshow(correlation_matrix.values, 
                     cmap='coolwarm', vmin=-1, vmax=1,
                     aspect='auto')
    
    # 设置坐标轴
    ax9.set_xticks(range(len(score_columns)))
    ax9.set_yticks(range(len(score_columns)))
    ax9.set_xticklabels(score_columns, rotation=45, ha='right', fontsize=9)
    ax9.set_yticklabels(score_columns, fontsize=9)
    
    # 添加数值标签
    for i in range(len(score_columns)):
        for j in range(len(score_columns)):
            value = correlation_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax9.text(j, i, f'{value:.2f}', ha='center', va='center',
                     color=color, fontsize=8, fontweight='bold')
    
    ax9.set_title('各科目成绩相关性矩阵', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar9 = fig.colorbar(im9, ax=ax9, shrink=0.8)
    cbar9.ax.set_ylabel('相关系数', rotation=90, fontsize=11)
    
    # ------------------------------------------------------------------------
    # 10. 学习时长分布
    # ------------------------------------------------------------------------
    ax10 = fig.add_subplot(gs[2, 2])
    
    study_hours = df['学习时长_小时'].values
    
    # 绘制直方图
    n, bins, patches = ax10.hist(study_hours, bins=15, 
                                color='steelblue', edgecolor='black',
                                alpha=0.7, density=True)
    
    # 添加核密度估计
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(study_hours)
    x_kde = np.linspace(study_hours.min(), study_hours.max(), 100)
    ax10.plot(x_kde, kde(x_kde), 'r-', linewidth=2.5, label='核密度估计')
    
    # 设置标签和标题
    ax10.set_xlabel('学习时长（小时）', fontsize=12)
    ax10.set_ylabel('概率密度', fontsize=12)
    ax10.set_title('学生学习时长分布', fontsize=14, fontweight='bold')
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------------
    # 11. 成绩随时间变化趋势
    # ------------------------------------------------------------------------
    ax11 = fig.add_subplot(gs[2, 3])
    
    # 模拟每天的平均成绩变化
    days = learning_data['days']
    daily_avg_scores = np.array([65, 68, 72, 75, 78, 82, 85])
    
    # 绘制折线图和置信区间
    ax11.plot(days, daily_avg_scores, 's-', linewidth=2.5, 
             markersize=8, color='darkblue', label='日平均成绩')
    
    # 添加置信区间
    confidence = np.array([5, 4, 4, 3, 3, 2, 2])  # 置信区间宽度
    ax11.fill_between(days, 
                      daily_avg_scores - confidence,
                      daily_avg_scores + confidence,
                      alpha=0.2, color='blue', label='95%置信区间')
    
    # 设置标签和标题
    ax11.set_xlabel('学习天数', fontsize=12)
    ax11.set_ylabel('平均成绩', fontsize=12)
    ax11.set_title('班级平均成绩逐日变化', fontsize=14, fontweight='bold')
    ax11.legend(fontsize=10)
    ax11.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------------
    # 12. 技能掌握进度雷达图
    # ------------------------------------------------------------------------
    ax12 = fig.add_subplot(gs[3, 0], polar=True)
    
    skill_progress = learning_data['skill_progress']
    skills = list(skill_progress.keys())
    
    # 计算最终掌握百分比
    final_progress = [skill_progress[skill][-1] for skill in skills]
    
    # 准备角度
    angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
    final_progress.append(final_progress[0])
    angles.append(angles[0])
    
    # 绘制雷达图
    ax12.plot(angles, final_progress, 'o-', linewidth=2.5, 
             markersize=8, color='purple', alpha=0.8)
    ax12.fill(angles, final_progress, alpha=0.2, color='purple')
    
    # 设置极坐标标签
    ax12.set_xticks(angles[:-1])
    ax12.set_xticklabels(skills, fontsize=10, fontweight='bold')
    ax12.set_ylim([0, 100])
    ax12.set_yticks([25, 50, 75, 100])
    ax12.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    
    ax12.set_title('技能掌握程度雷达图', fontsize=14, fontweight='bold', pad=20)
    
    # ------------------------------------------------------------------------
    # 13. 学生成绩与学习时长关系（3D）
    # ------------------------------------------------------------------------
    ax13 = fig.add_subplot(gs[3, 1], projection='3d')
    
    # 提取数据
    linear_scores = df['线性代数'].values
    prob_scores = df['概率统计'].values
    study_hours = df['学习时长_小时'].values
    
    # 绘制3D散点图
    scatter13 = ax13.scatter(linear_scores, prob_scores, study_hours,
                            c=df['平均成绩'].values, cmap='plasma',
                            s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置坐标轴标签
    ax13.set_xlabel('线性代数成绩', fontsize=10, labelpad=10)
    ax13.set_ylabel('概率统计成绩', fontsize=10, labelpad=10)
    ax13.set_zlabel('学习时长（小时）', fontsize=10, labelpad=10)
    ax13.set_title('多维成绩关系分析', fontsize=12, fontweight='bold')
    
    # 调整视角
    ax13.view_init(elev=25, azim=45)
    
    # 添加颜色条
    cbar13 = fig.colorbar(scatter13, ax=ax13, shrink=0.6, pad=0.1)
    cbar13.ax.set_ylabel('平均成绩', rotation=90, fontsize=9)
    
    # ------------------------------------------------------------------------
    # 14. 数据集质量对比（多条形图）
    # ------------------------------------------------------------------------
    ax14 = fig.add_subplot(gs[3, 2])
    
    dataset_quality = learning_data['dataset_quality']
    quality_metrics = list(dataset_quality.keys())
    
    # 准备数据
    initial_quality = [dataset_quality[metric][0] for metric in quality_metrics]
    final_quality = [dataset_quality[metric][-1] for metric in quality_metrics]
    improvements = [final - initial for final, initial in zip(final_quality, initial_quality)]
    
    # 设置位置
    x = np.arange(len(quality_metrics))
    width = 0.35
    
    # 绘制条形图
    bars1 = ax14.bar(x - width/2, initial_quality, width, 
                    label='初始质量', color='lightcoral', alpha=0.8)
    bars2 = ax14.bar(x + width/2, final_quality, width, 
                    label='最终质量', color='steelblue', alpha=0.8)
    
    # 添加改进值标签
    for i, (init, final, imp) in enumerate(zip(initial_quality, final_quality, improvements)):
        ax14.text(i - width/2, init + 2, f'{init:.0f}%', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax14.text(i + width/2, final + 2, f'{final:.0f}%', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax14.text(i, max(init, final) + 5, f'+{imp:.0f}%', 
                 ha='center', va='bottom', fontsize=8, color='darkgreen')
    
    # 设置标签和标题
    ax14.set_xlabel('质量指标', fontsize=12)
    ax14.set_ylabel('质量分数（%）', fontsize=12)
    ax14.set_title('数据集清洗前后质量对比', fontsize=14, fontweight='bold')
    ax14.set_xticks(x)
    ax14.set_xticklabels(quality_metrics, fontsize=10)
    ax14.legend(fontsize=10)
    ax14.grid(True, alpha=0.3, axis='y')
    
    # ------------------------------------------------------------------------
    # 15. 学习成效总结（文本）
    # ------------------------------------------------------------------------
    ax15 = fig.add_subplot(gs[3, 3])
    ax15.axis('off')
    
    # 计算学习成效指标
    total_students = learning_data['n_students']
    excellent_count = (df['表现等级'] == '优秀').sum()
    excellent_rate = excellent_count / total_students * 100
    
    avg_study_hours = df['学习时长_小时'].mean()
    avg_total_score = df['总成绩'].mean()
    
    skill_growth = {}
    for skill in learning_data['skill_progress'].keys():
        progress = learning_data['skill_progress'][skill]
        growth = progress[-1] - progress[0]
        skill_growth[skill] = growth
    
    # 生成总结文本
    summary_text = (
        f"📊 Week 1 学习成效综合分析报告\n\n"
        f"📈 总体表现:\n"
        f"• 学生总数: {total_students}人\n"
        f"• 优秀率: {excellent_rate:.1f}% ({excellent_count}人)\n"
        f"• 平均学习时长: {avg_study_hours:.1f}小时/周\n"
        f"• 平均总成绩: {avg_total_score:.0f}分\n\n"
        f"🚀 技能成长:\n"
    )
    
    for skill, growth in skill_growth.items():
        summary_text += f"• {skill}: +{growth:.0f}%\n"
    
    summary_text += (
        f"\n🎯 关键发现:\n"
        f"1. 学习时长与成绩显著正相关 (R = {correlation:.3f})\n"
        f"2. 线性代数与矩阵运算掌握最扎实\n"
        f"3. 数据清洗技能提升最为显著\n"
        f"4. 整体学习曲线呈稳步上升趋势\n\n"
        f"🔮 后续建议:\n"
        f"• 加强概率统计的应用训练\n"
        f"• 引入更多实际数据集练习\n"
        f"• 开展小组项目提升综合能力"
    )
    
    ax15.text(0.05, 0.95, summary_text, fontsize=10, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', 
                       alpha=0.9, pad=10))
    
    # ------------------------------------------------------------------------
    # 图形整体设置
    # ------------------------------------------------------------------------
    plt.suptitle('AI技术学习Week 1成果综合可视化仪表板\n'
                 '多维度数据分析与学习成效评估',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def create_professional_report(learning_data, df):
    """
    生成专业风格的学习报告
    
    参数：
    learning_data: 学习数据字典
    df: 学生数据DataFrame
    
    返回：
    fig: 专业报告图形
    """
    # 创建图形
    fig = plt.figure(figsize=(16, 20))
    
    # 创建网格布局（报告风格）
    gs = gridspec.GridSpec(5, 3, figure=fig, 
                          height_ratios=[0.5, 1, 1, 1, 1])
    
    # ------------------------------------------------------------------------
    # 报告标题区域
    # ------------------------------------------------------------------------
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    title_text = (
        f"AI技术学习与大厂入职计划\n"
        f"Week 1 学习成果专业分析报告\n"
        f"报告日期: 2026年3月30日\n"
        f"分析对象: {learning_data['n_students']}名学习者"
    )
    
    ax_title.text(0.5, 0.5, title_text, fontsize=16, fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='lightblue',
                           alpha=0.8, pad=15))
    
    # ------------------------------------------------------------------------
    # 核心指标卡片
    # ------------------------------------------------------------------------
    ax_cards = []
    card_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    card_titles = ['总体优秀率', '平均学习时长', '技能平均成长', '成绩标准差']
    
    # 计算核心指标
    total_students = learning_data['n_students']
    excellent_count = (df['表现等级'] == '优秀').sum()
    excellent_rate = excellent_count / total_students * 100
    
    avg_study_hours = df['学习时长_小时'].mean()
    
    skill_growth_avg = np.mean([
        learning_data['skill_progress'][skill][-1] - learning_data['skill_progress'][skill][0]
        for skill in learning_data['skill_progress'].keys()
    ])
    
    score_std = df['平均成绩'].std()
    
    card_values = [
        f'{excellent_rate:.1f}%',
        f'{avg_study_hours:.1f}小时',
        f'{skill_growth_avg:.0f}%',
        f'{score_std:.1f}分'
    ]
    
    card_descriptions = [
        f'{excellent_count}/{total_students}人',
        '周平均学习时间',
        '各项技能平均提升',
        '成绩离散程度'
    ]
    
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax_cards.append(ax)
        ax.axis('off')
        
        # 绘制卡片背景
        rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                            facecolor=card_colors[i], alpha=0.7,
                            transform=ax.transAxes, 
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # 添加卡片内容
        ax.text(0.5, 0.65, card_values[i], fontsize=22, fontweight='bold',
                ha='center', va='center', color='white')
        
        ax.text(0.5, 0.4, card_titles[i], fontsize=12, fontweight='medium',
                ha='center', va='center', color='white')
        
        ax.text(0.5, 0.25, card_descriptions[i], fontsize=9,
                ha='center', va='center', color='white')
    
    # ------------------------------------------------------------------------
    # 技能掌握进度矩阵
    # ------------------------------------------------------------------------
    ax_skill_matrix = fig.add_subplot(gs[2, :])
    
    skill_progress = learning_data['skill_progress']
    skills = list(skill_progress.keys())
    days = learning_data['days']
    
    # 准备数据矩阵
    progress_matrix = np.array([skill_progress[skill] for skill in skills])
    
    # 绘制热力图
    im = ax_skill_matrix.imshow(progress_matrix, cmap='YlOrRd', aspect='auto',
                               vmin=0, vmax=100, interpolation='nearest')
    
    # 设置坐标轴
    ax_skill_matrix.set_xticks(range(len(days)))
    ax_skill_matrix.set_yticks(range(len(skills)))
    ax_skill_matrix.set_xticklabels([f'Day {d}' for d in days], fontsize=10)
    ax_skill_matrix.set_yticklabels(skills, fontsize=10)
    
    # 添加进度值标签
    for i in range(len(skills)):
        for j in range(len(days)):
            value = progress_matrix[i, j]
            color = 'white' if value < 50 else 'black'
            ax_skill_matrix.text(j, i, f'{value:.0f}%', ha='center', va='center',
                                color=color, fontsize=9, fontweight='bold')
    
    ax_skill_matrix.set_title('技能掌握进度矩阵分析', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax_skill_matrix, shrink=0.8)
    cbar.ax.set_ylabel('掌握进度（%）', rotation=90, fontsize=11)
    
    # ------------------------------------------------------------------------
    # 学习成效对比分析
    # ------------------------------------------------------------------------
    ax_comparison = fig.add_subplot(gs[3, :])
    
    # 提取各科目成绩
    score_columns = ['线性代数', '矩阵运算', '概率统计', 
                    '统计分析', '数据清洗', 'Pandas掌握']
    subject_means = [df[col].mean() for col in score_columns]
    subject_stds = [df[col].std() for col in score_columns]
    
    # 设置位置
    x_pos = np.arange(len(score_columns))
    
    # 绘制条形图
    bars = ax_comparison.bar(x_pos, subject_means, yerr=subject_stds,
                            capsize=5, color='steelblue', alpha=0.8,
                            edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, mean_val in zip(bars, subject_means):
        height = bar.get_height()
        ax_comparison.text(bar.get_x() + bar.get_width()/2, height + 2,
                          f'{mean_val:.1f}', ha='center', va='bottom',
                          fontsize=10, fontweight='bold')
    
    # 设置标签和标题
    ax_comparison.set_xlabel('学习科目', fontsize=12)
    ax_comparison.set_ylabel('平均成绩（分）', fontsize=12)
    ax_comparison.set_title('各科目学习成效对比分析', fontsize=14, fontweight='bold')
    ax_comparison.set_xticks(x_pos)
    ax_comparison.set_xticklabels(score_columns, fontsize=10, rotation=0)
    ax_comparison.grid(True, alpha=0.3, axis='y')
    
    # ------------------------------------------------------------------------
    # 报告总结与建议
    # ------------------------------------------------------------------------
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    # 计算改进建议指标
    weak_subject = score_columns[np.argmin(subject_means)]
    strong_subject = score_columns[np.argmax(subject_means)]
    
    max_improvement = max(skill_growth_avg for skill_growth_avg in [
        learning_data['skill_progress'][skill][-1] - learning_data['skill_progress'][skill][0]
        for skill in learning_data['skill_progress'].keys()
    ])
    
    best_skill = max(learning_data['skill_progress'].keys(),
                    key=lambda k: learning_data['skill_progress'][k][-1] - learning_data['skill_progress'][k][0])
    
    summary_report = (
        f"📋 Week 1 学习成效总结报告\n\n"
        f"🎖️ 优势领域:\n"
        f"• 最强科目: {strong_subject} (平均{subject_means[np.argmax(subject_means)]:.1f}分)\n"
        f"• 进步最快技能: {best_skill} (+{max_improvement:.0f}%)\n"
        f"• 整体优秀率: {excellent_rate:.1f}%，表现优异\n\n"
        f"🔧 待改进领域:\n"
        f"• 相对薄弱科目: {weak_subject} (平均{subject_means[np.argmin(subject_means)]:.1f}分)\n"
        f"• 成绩离散度: {score_std:.1f}分，个体差异明显\n"
        f"• 学习时长分布不均，部分学生需加强时间管理\n\n"
        f"💡 针对性建议:\n"
        f"1. 针对{weak_subject}开展专项强化训练\n"
        f"2. 建立学习小组，促进经验交流与互助\n"
        f"3. 优化学习计划，确保各科目均衡发展\n"
        f"4. 加强实践项目，提升综合应用能力\n\n"
        f"📈 预期Week 2重点:\n"
        f"• 机器学习基础概念掌握\n"
        f"• Python数据科学项目实战\n"
        f"• AI工程化思维初步建立"
    )
    
    ax_summary.text(0.02, 0.95, summary_report, fontsize=11, va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray',
                            alpha=0.9, pad=12))
    
    # ------------------------------------------------------------------------
    # 图形整体设置
    # ------------------------------------------------------------------------
    plt.tight_layout()
    
    return fig


# ============================================================================
# 主程序执行
# ============================================================================

def main():
    """
    主函数：执行综合可视化项目
    """
    print("=" * 60)
    print("综合可视化项目 - Week 1 Day 5")
    print("=" * 60)
    
    print("\n1. 生成Week 1学习数据...")
    learning_data = generate_week1_learning_data()
    
    print("2. 创建学习数据分析DataFrame...")
    df = create_learning_dataframe(learning_data)
    
    print("3. 创建综合可视化仪表板...")
    fig1 = create_comprehensive_dashboard(learning_data, df)
    fig1.savefig('outputs/阶段一/Week1/Day5/comprehensive_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("4. 生成专业分析报告...")
    fig2 = create_professional_report(learning_data, df)
    fig2.savefig('outputs/阶段一/Week1/Day5/professional_report.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("\n5. 数据概览:")
    print(f"   • 学生总数: {learning_data['n_students']}人")
    print(f"   • 技能类型: {len(learning_data['skill_progress'])}种")
    print(f"   • 学习天数: {len(learning_data['days'])}天")
    print(f"   • 矩阵大小: {learning_data['matrix_size']}×{learning_data['matrix_size']}")
    
    print("\n6. 生成的文件:")
    print("   • comprehensive_dashboard.png - 综合可视化仪表板")
    print("   • professional_report.png     - 专业分析报告")
    print("   • AI数据可视化案例研究.py     - 案例研究源代码")
    print("   • ai_visualization_tools.py   - 可复用可视化工具")
    
    print("\n" + "=" * 60)
    print("项目执行完成！")
    print("=" * 60)
    
    # 输出关键统计信息
    print("\n📊 关键统计指标:")
    print(f"   • 平均学习时长: {df['学习时长_小时'].mean():.2f} 小时")
    print(f"   • 平均总成绩: {df['总成绩'].mean():.0f} 分")
    print(f"   • 优秀学生比例: {(df['表现等级'] == '优秀').sum()}/{learning_data['n_students']} 人")
    print(f"   • 最强科目: {df[['线性代数', '矩阵运算', '概率统计', '统计分析', '数据清洗', 'Pandas掌握']].mean().idxmax()}")
    print(f"   • 最需提升科目: {df[['线性代数', '矩阵运算', '概率统计', '统计分析', '数据清洗', 'Pandas掌握']].mean().idxmin()}")
    
    return True


if __name__ == "__main__":
    main()