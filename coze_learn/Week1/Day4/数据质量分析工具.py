"""
数据质量分析工具
===================================

本工具提供泰坦尼克号数据集的质量分析和可视化功能，包括：
1. 数据分布分析（直方图、箱线图、小提琴图）
2. 缺失值分析（热力图、条形图）
3. 特征相关性分析（热力图、散点图矩阵）
4. 数据质量报告生成

使用方法：
1. 加载数据后，调用相应分析函数
2. 可视化结果将保存为图片文件
3. 结合数据清洗实战手册进行对比分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# 1. 数据概览分析
# ============================================================================

def data_overview(df, save_fig=True):
    """
    数据概览分析：基本信息、数据类型、缺失情况
    
    参数：
    df: 待分析的DataFrame
    save_fig: 是否保存可视化图表
    
    返回：
    包含概览信息的字典
    """
    print("=" * 60)
    print("数据概览分析")
    print("=" * 60)
    
    overview = {}
    
    # 基本信息
    overview['shape'] = df.shape
    overview['dtypes'] = df.dtypes.to_dict()
    overview['memory_usage'] = df.memory_usage().sum() / 1024**2  # MB
    
    print(f"数据形状: {df.shape[0]}行 × {df.shape[1]}列")
    print(f"内存占用: {overview['memory_usage']:.2f} MB")
    
    # 数据类型统计
    dtype_counts = df.dtypes.value_counts()
    print("\n数据类型分布:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}列")
    
    # 缺失值统计
    missing_total = df.isnull().sum().sum()
    missing_percentage = missing_total / (df.shape[0] * df.shape[1]) * 100
    
    overview['missing_total'] = missing_total
    overview['missing_percentage'] = missing_percentage
    
    print(f"\n缺失值总数: {missing_total}")
    print(f"缺失值比例: {missing_percentage:.2f}%")
    
    # 数值型特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        overview['numeric_stats'] = numeric_stats
        
        print(f"\n数值型特征: {len(numeric_cols)}个")
        print("基本统计量:")
        print(numeric_stats.round(2))
    
    # 分类型特征统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'missing': df[col].isnull().sum()
            }
        
        overview['categorical_stats'] = categorical_stats
        
        print(f"\n分类型特征: {len(categorical_cols)}个")
        for col, stats in categorical_stats.items():
            print(f"  {col}: {stats['unique_values']}个唯一值, "
                  f"最常见: {stats['most_common']}, 缺失: {stats['missing']}个")
    
    return overview

# ============================================================================
# 2. 缺失值可视化
# ============================================================================

def visualize_missing_data(df, save_fig=True):
    """
    可视化缺失值分布
    
    参数：
    df: 待分析的DataFrame
    save_fig: 是否保存图片
    
    返回：
    缺失值统计DataFrame
    """
    print("\n" + "=" * 60)
    print("缺失值可视化分析")
    print("=" * 60)
    
    # 计算缺失值统计
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例': df.isnull().sum() / len(df) * 100
    })
    missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失比例', ascending=False)
    
    if len(missing_stats) == 0:
        print("数据集中没有缺失值！")
        return missing_stats
    
    print("各列缺失情况:")
    for idx, row in missing_stats.iterrows():
        print(f"  {idx}: {row['缺失数量']}个缺失 ({row['缺失比例']:.1f}%)")
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('泰坦尼克号数据集缺失值分析', fontsize=16, fontweight='bold')
    
    # 1. 缺失值数量条形图
    ax1 = axes[0, 0]
    bars = ax1.barh(missing_stats.index, missing_stats['缺失数量'], color='#ff6b6b')
    ax1.set_xlabel('缺失值数量', fontsize=12)
    ax1.set_title('各列缺失值数量', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # 最多的在顶部
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + max(missing_stats['缺失数量']) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{int(width)}', va='center', fontsize=10)
    
    # 2. 缺失值比例条形图
    ax2 = axes[0, 1]
    bars2 = ax2.barh(missing_stats.index, missing_stats['缺失比例'], color='#4ecdc4')
    ax2.set_xlabel('缺失比例 (%)', fontsize=12)
    ax2.set_title('各列缺失值比例', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + max(missing_stats['缺失比例']) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', va='center', fontsize=10)
    
    # 3. 缺失值热力图
    ax3 = axes[1, 0]
    # 创建缺失值矩阵（True表示缺失）
    missing_matrix = df.isnull()
    # 计算每行的缺失值数量
    row_missing = missing_matrix.sum(axis=1)
    # 取缺失值最多的前100行（避免图表过密）
    sample_size = min(100, len(df))
    sample_idx = row_missing.nlargest(sample_size).index
    sample_missing = missing_matrix.loc[sample_idx]
    
    sns.heatmap(sample_missing, cmap=['#f7f7f7', '#ff6b6b'], 
                cbar_kws={'label': '缺失值 (红色)'}, ax=ax3)
    ax3.set_title(f'缺失值分布热力图 (缺失最多的{sample_size}行)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('特征列', fontsize=12)
    ax3.set_ylabel('数据行索引', fontsize=12)
    
    # 4. 缺失值模式分析
    ax4 = axes[1, 1]
    # 计算缺失值组合模式
    missing_pattern = missing_matrix.astype(int).sum(axis=1).value_counts().sort_index()
    ax4.bar(missing_pattern.index.astype(str), missing_pattern.values, color='#45b7d1')
    ax4.set_xlabel('每行缺失值数量', fontsize=12)
    ax4.set_ylabel('行数', fontsize=12)
    ax4.set_title('缺失值模式分布', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, (idx, val) in enumerate(missing_pattern.items()):
        ax4.text(i, val + max(missing_pattern.values) * 0.01, 
                str(val), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
        print("\n缺失值分析图表已保存为: missing_data_analysis.png")
    
    plt.show()
    
    return missing_stats

# ============================================================================
# 3. 数据分布可视化
# ============================================================================

def visualize_data_distribution(df, numeric_cols=None, save_fig=True):
    """
    可视化数值型特征的数据分布
    
    参数：
    df: 待分析的DataFrame
    numeric_cols: 指定要分析的数值列，None表示自动选择
    save_fig: 是否保存图片
    
    返回：
    分布统计信息
    """
    print("\n" + "=" * 60)
    print("数据分布可视化分析")
    print("=" * 60)
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("没有数值型特征可供分析")
        return None
    
    print(f"分析以下数值型特征: {list(numeric_cols)}")
    
    # 计算分布统计
    distribution_stats = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        
        distribution_stats[col] = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'missing': df[col].isnull().sum()
        }
    
    # 创建可视化图表
    n_cols = min(4, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle('数值型特征分布分析', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(numeric_cols):
        if i >= len(axes):
            break
        
        ax = axes[i]
        data = df[col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title(f'{col}\n(全部缺失)', fontsize=12)
            continue
        
        # 直方图与核密度估计
        sns.histplot(data, kde=True, ax=ax, color='#45b7d1', bins=30, 
                    stat='density', alpha=0.6)
        
        # 添加均值和垂直线
        mean_val = data.mean()
        median_val = data.median()
        
        ax.axvline(mean_val, color='#ff6b6b', linestyle='--', linewidth=2, 
                  label=f'均值: {mean_val:.2f}')
        ax.axvline(median_val, color='#4ecdc4', linestyle='-.', linewidth=2,
                  label=f'中位数: {median_val:.2f}')
        
        # 添加箱线图（在上方）
        box_ax = ax.inset_axes([0.6, 0.7, 0.35, 0.25])
        box_ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='#98ddca'))
        box_ax.set_xticks([])
        box_ax.set_yticks([])
        box_ax.set_title('箱线图', fontsize=9)
        
        # 设置标题和标签
        skewness = data.skew()
        skew_label = '右偏' if skewness > 0.5 else '左偏' if skewness < -0.5 else '基本对称'
        
        ax.set_title(f'{col}\n偏度: {skewness:.2f} ({skew_label})', fontsize=12)
        ax.set_xlabel('值', fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
    
    # 隐藏多余的子图
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("\n数据分布分析图表已保存为: data_distribution_analysis.png")
    
    plt.show()
    
    # 创建分类特征分布图（如果有）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\n分类特征分布分析:")
        
        # 选择前4个分类特征
        top_cats = categorical_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(top_cats):
            if i >= 4:
                break
            
            ax = axes[i]
            value_counts = df[col].value_counts().head(10)  # 最多显示前10个
            
            if len(value_counts) > 0:
                # 条形图
                bars = ax.barh(range(len(value_counts)), value_counts.values, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
                
                ax.set_yticks(range(len(value_counts)))
                ax.set_yticklabels(value_counts.index)
                ax.invert_yaxis()
                
                ax.set_title(f'{col}\n(前{len(value_counts)}个类别)', fontsize=12)
                ax.set_xlabel('频数', fontsize=10)
                
                # 添加数值标签
                for j, (idx, val) in enumerate(value_counts.items()):
                    ax.text(val + max(value_counts.values) * 0.01, j, 
                           str(val), va='center', fontsize=9)
        
        plt.suptitle('分类特征分布分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('categorical_distribution_analysis.png', dpi=300, bbox_inches='tight')
            print("分类特征分布图表已保存为: categorical_distribution_analysis.png")
        
        plt.show()
    
    return distribution_stats

# ============================================================================
# 4. 相关性分析
# ============================================================================

def visualize_correlations(df, target_col=None, save_fig=True):
    """
    可视化特征之间的相关性
    
    参数：
    df: 待分析的DataFrame
    target_col: 目标变量列名，用于分析与其他特征的相关性
    save_fig: 是否保存图片
    
    返回：
    相关性矩阵
    """
    print("\n" + "=" * 60)
    print("特征相关性分析")
    print("=" * 60)
    
    # 选择数值型特征
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        print("数值型特征不足，无法进行相关性分析")
        return None
    
    # 计算相关性矩阵
    correlation_matrix = numeric_df.corr()
    
    print("相关性矩阵摘要:")
    print(correlation_matrix.round(3))
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('特征相关性分析', fontsize=16, fontweight='bold')
    
    # 1. 相关性热力图
    ax1 = axes[0, 0]
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot=True, fmt='.2f', ax=ax1)
    ax1.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
    
    # 2. 与目标变量的相关性（如果指定）
    if target_col and target_col in correlation_matrix.columns:
        ax2 = axes[0, 1]
        target_corr = correlation_matrix[target_col].drop(target_col).sort_values()
        
        bars = ax2.barh(range(len(target_corr)), target_corr.values,
                       color=['#ff6b6b' if x < 0 else '#4ecdc4' for x in target_corr.values])
        
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index)
        ax2.set_xlabel('相关系数', fontsize=12)
        ax2.set_title(f'各特征与"{target_col}"的相关性', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, (idx, val) in enumerate(target_corr.items()):
            color = '#333333'
            ha = 'left' if val > 0 else 'right'
            x_pos = val + 0.01 if val > 0 else val - 0.01
            ax2.text(x_pos, i, f'{val:.3f}', 
                    va='center', ha=ha, color=color, fontsize=9)
        
        ax2.axvline(x=0, color='gray', linewidth=0.8, alpha=0.7)
    
    # 3. 强相关性散点图（找出相关性最强的特征对）
    ax3 = axes[1, 0]
    
    # 找出相关性最强的特征对（排除对角线）
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            corr_pairs.append((col1, col2, corr))
    
    # 按绝对值排序
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # 选择前2个强相关特征对
    top_pairs = corr_pairs[:2]
    
    if top_pairs:
        for idx, (col1, col2, corr) in enumerate(top_pairs):
            if idx == 0:
                ax = ax3
            else:
                ax = axes[1, 1]
            
            # 绘制散点图
            sns.scatterplot(data=df, x=col1, y=col2, ax=ax, 
                           hue=target_col if target_col and target_col in df.columns else None,
                           palette='viridis', alpha=0.7, s=60)
            
            # 添加回归线
            from scipy import stats
            x_data = df[col1].dropna()
            y_data = df[col2].dropna()
            common_idx = x_data.index.intersection(y_data.index)
            
            if len(common_idx) > 1:
                x_vals = df.loc[common_idx, col1].values
                y_vals = df.loc[common_idx, col2].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                
                # 生成回归线
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept
                
                ax.plot(x_line, y_line, '--', color='red', linewidth=2,
                       label=f'回归线 (r={r_value:.3f})')
            
            ax.set_title(f'{col1} vs {col2}\n相关系数: {corr:.3f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(col1, fontsize=10)
            ax.set_ylabel(col2, fontsize=10)
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n相关性分析图表已保存为: correlation_analysis.png")
    
    plt.show()
    
    return correlation_matrix

# ============================================================================
# 5. 数据质量综合报告
# ============================================================================

def generate_data_quality_report(df, report_name='数据质量报告'):
    """
    生成数据质量综合报告
    
    参数：
    df: 待分析的DataFrame
    report_name: 报告名称
    
    返回：
    包含所有分析结果的字典
    """
    print("=" * 60)
    print(f"生成数据质量报告: {report_name}")
    print("=" * 60)
    
    report = {
        'report_name': report_name,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_shape': df.shape,
        'data_memory_mb': df.memory_usage().sum() / 1024**2
    }
    
    # 1. 数据概览
    print("\n1. 数据概览分析...")
    report['overview'] = data_overview(df, save_fig=False)
    
    # 2. 缺失值分析
    print("\n2. 缺失值分析...")
    report['missing_analysis'] = visualize_missing_data(df, save_fig=False)
    
    # 3. 数据分布分析
    print("\n3. 数据分布分析...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['distribution_analysis'] = visualize_data_distribution(
            df, numeric_cols=numeric_cols, save_fig=False
        )
    
    # 4. 相关性分析
    print("\n4. 相关性分析...")
    target_candidate = None
    for col in ['Survived', 'survived', 'target']:
        if col in df.columns:
            target_candidate = col
            break
    
    report['correlation_analysis'] = visualize_correlations(
        df, target_col=target_candidate, save_fig=False
    )
    
    # 5. 数据质量评分
    print("\n5. 数据质量评分...")
    quality_scores = calculate_quality_scores(df)
    report['quality_scores'] = quality_scores
    
    print("\n" + "=" * 60)
    print("数据质量报告生成完成!")
    print("=" * 60)
    
    return report

def calculate_quality_scores(df):
    """
    计算数据质量评分
    
    参数：
    df: 待分析的DataFrame
    
    返回：
    包含各项质量评分的字典
    """
    scores = {}
    
    # 1. 完整性评分（基于缺失值比例）
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness_score = 100 * (1 - missing_cells / total_cells) if total_cells > 0 else 100
    scores['完整性'] = round(completeness_score, 1)
    
    # 2. 一致性评分（检查数据类型一致性）
    # 简化为检查是否有列全部为同一值（可能表示数据问题）
    consistency_issues = 0
    for col in df.columns:
        if df[col].nunique() == 1:
            consistency_issues += 1
    
    consistency_score = 100 * (1 - consistency_issues / len(df.columns))
    scores['一致性'] = round(consistency_score, 1)
    
    # 3. 准确性评分（基于异常值比例）
    # 使用IQR方法检测异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_percentage = 0
    
    if len(numeric_cols) > 0:
        total_numeric_values = 0
        total_outliers = 0
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            total_numeric_values += len(data)
            total_outliers += len(outliers)
        
        if total_numeric_values > 0:
            outlier_percentage = total_outliers / total_numeric_values * 100
    
    accuracy_score = max(0, 100 - outlier_percentage)
    scores['准确性'] = round(accuracy_score, 1)
    
    # 4. 总体质量评分
    overall_score = np.mean(list(scores.values()))
    scores['总体质量'] = round(overall_score, 1)
    
    print("数据质量评分:")
    for dimension, score in scores.items():
        rating = "优秀" if score >= 90 else "良好" if score >= 75 else "中等" if score >= 60 else "较差"
        print(f"  {dimension}: {score}分 ({rating})")
    
    return scores

# ============================================================================
# 6. 主程序：示例使用
# ============================================================================

if __name__ == "__main__":
    # 示例：完整数据质量分析流程
    print("泰坦尼克号数据质量分析工具")
    print("-" * 40)
    
    # 1. 加载数据（使用Seaborn内置数据集）
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        print(f"数据加载成功，形状: {df.shape}")
    except ImportError:
        print("Seaborn未安装，尝试从URL加载...")
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        df = pd.read_csv(url)
        print(f"从URL加载成功，形状: {df.shape}")
    
    # 2. 生成数据质量报告
    report = generate_data_quality_report(df, '泰坦尼克号数据质量分析报告')
    
    # 3. 保存报告摘要
    report_summary = {
        '数据形状': report['data_shape'],
        '内存占用(MB)': round(report['data_memory_mb'], 2),
        '质量评分': report['quality_scores']
    }
    
    summary_df = pd.DataFrame([report_summary])
    summary_file = '数据质量报告摘要.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"\n报告摘要已保存到: {summary_file}")
    print("\n数据质量分析完成！")
    
    # 4. 可视化图表生成（如果需要）
    print("\n生成可视化图表...")
    visualize_missing_data(df, save_fig=True)
    visualize_data_distribution(df, save_fig=True)
    visualize_correlations(df, target_col='survived' if 'survived' in df.columns else None, save_fig=True)
    
    print("\n所有分析图表已生成并保存！")