#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程完整Pipeline代码示例
====================================
本文件提供特征工程的完整实现，包括：
1. 数据预处理（缺失值处理、异常值检测）
2. 特征缩放（标准化、归一化）
3. 特征编码（独热编码、标签编码）
4. 特征选择（基于统计、基于模型）
5. 特征提取（PCA、LDA）
6. 特征构造（多项式特征、交互特征）

所有功能模块化设计，注释清晰，可直接运行。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# ============================================================================
# 1. 生成模拟数据集
# ============================================================================
def generate_sample_data(n_samples=1000, n_features=20, n_informative=10):
    """
    生成模拟分类数据集
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_informative: 信息特征数量（真正有用的特征）
    
    返回:
        X, y: 特征矩阵和标签
    """
    print("生成模拟数据集...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,  # 冗余特征（由信息特征线性组合生成）
        n_repeated=2,   # 重复特征
        n_clusters_per_class=2,
        flip_y=0.05,    # 噪声标签比例
        random_state=42
    )
    
    # 转换为DataFrame以便演示
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 添加一些缺失值和异常值
    # 随机添加5%的缺失值
    mask = np.random.rand(*X_df.shape) < 0.05
    X_df[mask] = np.nan
    
    # 添加一些异常值
    outlier_mask = np.random.rand(*X_df.shape) < 0.02
    X_df[outlier_mask] = X_df[outlier_mask] * 10
    
    print(f"数据集形状: {X_df.shape}")
    print(f"标签分布: {pd.Series(y).value_counts().to_dict()}")
    
    return X_df, y, feature_names

# ============================================================================
# 2. 数据预处理模块
# ============================================================================
class DataPreprocessor:
    """数据预处理类，处理缺失值和异常值"""
    
    @staticmethod
    def handle_missing_values(df, strategy='mean'):
        """
        处理缺失值
        
        参数:
            df: 输入DataFrame
            strategy: 缺失值填充策略，可选'mean', 'median', 'mode', 'constant', 'drop'
        
        返回:
            处理后的DataFrame
        """
        df_clean = df.copy()
        missing_count = df_clean.isnull().sum().sum()
        
        if missing_count == 0:
            print("数据中没有缺失值")
            return df_clean
        
        print(f"发现缺失值数量: {missing_count}")
        
        if strategy == 'mean':
            # 数值列用均值填充，分类列用众数填充
            for col in df_clean.columns:
                if df_clean[col].dtype in [np.int64, np.float64]:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            print("使用均值/众数填充缺失值")
            
        elif strategy == 'median':
            for col in df_clean.columns:
                if df_clean[col].dtype in [np.int64, np.float64]:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            print("使用中位数填充缺失值")
            
        elif strategy == 'drop':
            df_clean.dropna(inplace=True)
            print(f"删除包含缺失值的行，剩余样本数: {df_clean.shape[0]}")
            
        elif strategy == 'constant':
            df_clean.fillna(0, inplace=True)
            print("使用常数0填充缺失值")
        
        return df_clean
    
    @staticmethod
    def detect_outliers(df, method='iqr', threshold=1.5):
        """
        检测异常值
        
        参数:
            df: 输入DataFrame
            method: 检测方法，可选'iqr'（四分位距）或'zscore'（Z-score）
            threshold: 异常值阈值
        
        返回:
            outlier_mask: 异常值布尔掩码
        """
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                if method == 'iqr':
                    # IQR方法
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    # Z-score方法
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    col_mask = z_scores > threshold
                
                outlier_mask[col] = col_mask
        
        outlier_count = outlier_mask.sum().sum()
        print(f"检测到异常值数量: {outlier_count}")
        
        return outlier_mask
    
    @staticmethod
    def handle_outliers(df, outlier_mask, strategy='cap'):
        """
        处理异常值
        
        参数:
            df: 输入DataFrame
            outlier_mask: 异常值掩码
            strategy: 处理策略，可选'cap'（缩尾）或'remove'（删除）
        
        返回:
            处理后的DataFrame
        """
        df_clean = df.copy()
        
        if strategy == 'cap':
            # 缩尾处理：将异常值替换为边界值
            for col in df_clean.columns:
                if df_clean[col].dtype in [np.int64, np.float64]:
                    # 计算正常值的边界
                    normal_values = df_clean[col][~outlier_mask[col]]
                    if len(normal_values) > 0:
                        lower_bound = normal_values.quantile(0.01)  # 1%分位数
                        upper_bound = normal_values.quantile(0.99)  # 99%分位数
                        
                        # 替换异常值
                        df_clean.loc[outlier_mask[col], col] = df_clean.loc[outlier_mask[col], col].clip(
                            lower=lower_bound, upper=upper_bound
                        )
            print("使用缩尾法处理异常值")
            
        elif strategy == 'remove':
            # 删除包含异常值的行
            rows_with_outliers = outlier_mask.any(axis=1)
            df_clean = df_clean[~rows_with_outliers].copy()
            print(f"删除包含异常值的行，剩余样本数: {df_clean.shape[0]}")
        
        return df_clean

# ============================================================================
# 3. 特征缩放模块
# ============================================================================
class FeatureScaler:
    """特征缩放类"""
    
    @staticmethod
    def standardize(df):
        """
        标准化（Z-score标准化）：均值为0，标准差为1
        
        公式: z = (x - μ) / σ
        
        参数:
            df: 输入DataFrame
        
        返回:
            标准化后的DataFrame
        """
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        print(f"标准化处理特征: {list(numeric_cols)}")
        return df_scaled
    
    @staticmethod
    def normalize(df, method='minmax'):
        """
        归一化：将特征缩放到指定范围
        
        参数:
            df: 输入DataFrame
            method: 归一化方法，可选'minmax'（0-1）或'maxabs'（-1到1）
        
        返回:
            归一化后的DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            print("使用MinMax归一化（范围0-1）")
            
        elif method == 'maxabs':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            print("使用MaxAbs归一化（范围-1到1）")
        
        df_normalized = df.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df_normalized

# ============================================================================
# 4. 特征编码模块
# ============================================================================
class FeatureEncoder:
    """特征编码类，处理分类特征"""
    
    @staticmethod
    def create_categorical_features(df, n_categorical=3):
        """
        创建模拟分类特征用于演示
        
        参数:
            df: 输入DataFrame
            n_categorical: 要创建的分类特征数量
        
        返回:
            添加了分类特征的DataFrame
        """
        df_with_cat = df.copy()
        
        for i in range(n_categorical):
            col_name = f'category_{i}'
            # 创建3-5个类别的分类特征
            n_categories = np.random.randint(3, 6)
            categories = [f'cat_{j}' for j in range(n_categories)]
            df_with_cat[col_name] = np.random.choice(categories, size=len(df))
        
        print(f"创建了 {n_categorical} 个分类特征")
        return df_with_cat
    
    @staticmethod
    def one_hot_encode(df, categorical_cols=None):
        """
        独热编码（One-Hot Encoding）
        
        参数:
            df: 输入DataFrame
            categorical_cols: 需要编码的分类列列表，None则自动检测
        
        返回:
            独热编码后的DataFrame
        """
        if categorical_cols is None:
            # 自动检测分类列（非数值型且唯一值数量较少）
            categorical_cols = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    categorical_cols.append(col)
        
        if not categorical_cols:
            print("未找到需要独热编码的分类特征")
            return df
        
        from sklearn.preprocessing import OneHotEncoder
        
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_array = encoder.fit_transform(df[categorical_cols])
        
        # 创建编码后的列名
        encoded_col_names = []
        for i, col in enumerate(categorical_cols):
            categories = encoder.categories_[i][1:]  # 去掉第一个类别（避免多重共线性）
            for cat in categories:
                encoded_col_names.append(f"{col}_{cat}")
        
        # 创建新的DataFrame
        df_encoded = df.drop(columns=categorical_cols)
        df_encoded = pd.concat([
            df_encoded,
            pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
        ], axis=1)
        
        print(f"独热编码特征: {categorical_cols}")
        print(f"编码后新增特征数: {encoded_array.shape[1]}")
        
        return df_encoded
    
    @staticmethod
    def label_encode(df, categorical_cols=None):
        """
        标签编码（Label Encoding）
        
        参数:
            df: 输入DataFrame
            categorical_cols: 需要编码的分类列列表
        
        返回:
            标签编码后的DataFrame
        """
        if categorical_cols is None:
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        if not categorical_cols:
            print("未找到需要标签编码的分类特征")
            return df
        
        from sklearn.preprocessing import LabelEncoder
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            print(f"标签编码特征 '{col}'，类别数: {len(le.classes_)}")
        
        return df_encoded

# ============================================================================
# 5. 特征选择模块
# ============================================================================
class FeatureSelector:
    """特征选择类"""
    
    @staticmethod
    def variance_threshold(df, threshold=0.01):
        """
        方差选择法：移除方差低于阈值的特征
        
        参数:
            df: 输入DataFrame
            threshold: 方差阈值
        
        返回:
            选择后的DataFrame和保留的特征索引
        """
        from sklearn.feature_selection import VarianceThreshold
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        X_numeric = df[numeric_cols].values
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X_numeric)
        
        # 获取保留的特征索引
        retained_indices = selector.get_support(indices=True)
        retained_cols = [numeric_cols[i] for i in retained_indices]
        
        df_selected = df.copy()
        # 只保留数值特征中被选中的列
        cols_to_drop = [col for col in numeric_cols if col not in retained_cols]
        df_selected = df_selected.drop(columns=cols_to_drop)
        
        print(f"方差选择: 原始特征数 {len(numeric_cols)} → 保留特征数 {len(retained_cols)}")
        
        return df_selected, retained_indices
    
    @staticmethod
    def correlation_based(df, target, threshold=0.3):
        """
        基于相关系数的特征选择
        
        参数:
            df: 输入DataFrame（仅数值特征）
            target: 目标变量
            threshold: 相关系数绝对值阈值
        
        返回:
            选择后的DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        
        # 计算每个特征与目标的相关系数
        correlations = {}
        for col in numeric_cols:
            corr = np.corrcoef(df[col], target)[0, 1]
            correlations[col] = corr
        
        # 选择相关系数绝对值大于阈值的特征
        selected_cols = [col for col, corr in correlations.items() if abs(corr) > threshold]
        
        df_selected = df[selected_cols].copy()
        
        print(f"相关系数选择: 原始特征数 {len(numeric_cols)} → 保留特征数 {len(selected_cols)}")
        print(f"最大相关系数: {max(correlations.values(), key=abs):.3f}")
        
        return df_selected
    
    @staticmethod
    def model_based_selection(df, target, n_features=10):
        """
        基于模型的特征选择（使用随机森林）
        
        参数:
            df: 输入DataFrame
            target: 目标变量
            n_features: 要选择的特征数量
        
        返回:
            选择后的DataFrame和特征重要性
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        X = df[numeric_cols].values
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, target)
        
        # 基于特征重要性选择特征
        selector = SelectFromModel(rf, max_features=n_features, threshold=-np.inf)
        selector.fit(X, target)
        
        # 获取保留的特征
        retained_indices = selector.get_support(indices=True)
        retained_cols = [numeric_cols[i] for i in retained_indices]
        
        df_selected = df[retained_cols].copy()
        
        # 获取特征重要性
        importances = rf.feature_importances_
        
        print(f"模型选择: 原始特征数 {len(numeric_cols)} → 保留特征数 {len(retained_cols)}")
        print(f"Top {min(5, len(retained_cols))} 重要特征:")
        for i, idx in enumerate(retained_indices[:5]):
            print(f"  {numeric_cols[idx]}: {importances[idx]:.4f}")
        
        return df_selected, importances[retained_indices]

# ============================================================================
# 6. 特征提取模块
# ============================================================================
class FeatureExtractor:
    """特征提取类"""
    
    @staticmethod
    def pca_extraction(df, n_components=None, variance_ratio=0.95):
        """
        主成分分析（PCA）特征提取
        
        参数:
            df: 输入DataFrame
            n_components: 主成分数量，None则根据variance_ratio自动确定
            variance_ratio: 保留的方差比例
        
        返回:
            PCA转换后的DataFrame
        """
        from sklearn.decomposition import PCA
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        X = df[numeric_cols].values
        
        if n_components is None:
            pca = PCA(n_components=variance_ratio)
        else:
            pca = PCA(n_components=n_components)
        
        X_pca = pca.fit_transform(X)
        
        # 创建新的列名
        pca_cols = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        print(f"PCA提取: 原始特征数 {X.shape[1]} → PCA特征数 {X_pca.shape[1]}")
        print(f"解释方差比例: {sum(pca.explained_variance_ratio_):.3f}")
        
        return df_pca, pca
    
    @staticmethod
    def lda_extraction(df, target, n_components=None):
        """
        线性判别分析（LDA）特征提取
        
        参数:
            df: 输入DataFrame
            target: 目标变量
            n_components: 提取的特征数量
        
        返回:
            LDA转换后的DataFrame
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        X = df[numeric_cols].values
        
        if n_components is None:
            n_components = min(len(np.unique(target)) - 1, X.shape[1])
        
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, target)
        
        # 创建新的列名
        lda_cols = [f'LD_{i+1}' for i in range(X_lda.shape[1])]
        df_lda = pd.DataFrame(X_lda, columns=lda_cols, index=df.index)
        
        print(f"LDA提取: 原始特征数 {X.shape[1]} → LDA特征数 {X_lda.shape[1]}")
        
        return df_lda, lda

# ============================================================================
# 7. 特征构造模块
# ============================================================================
class FeatureConstructor:
    """特征构造类"""
    
    @staticmethod
    def polynomial_features(df, degree=2, interaction_only=False):
        """
        构造多项式特征
        
        参数:
            df: 输入DataFrame
            degree: 多项式阶数
            interaction_only: 是否只构造交互特征
        
        返回:
            添加了多项式特征的DataFrame
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        X_numeric = df[numeric_cols].values
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        X_poly = poly.fit_transform(X_numeric)
        
        # 获取特征名称
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # 创建新的DataFrame
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        print(f"多项式特征构造: 原始特征数 {len(numeric_cols)} → 多项式特征数 {len(feature_names)}")
        print(f"多项式阶数: {degree}, 交互特征: {interaction_only}")
        
        return df_poly
    
    @staticmethod
    def interaction_features(df, feature_pairs=None):
        """
        构造交互特征
        
        参数:
            df: 输入DataFrame
            feature_pairs: 特征对列表，如[('feat1', 'feat2'), ('feat3', 'feat4')]
        
        返回:
            添加了交互特征的DataFrame
        """
        df_interaction = df.copy()
        
        if feature_pairs is None:
            # 自动选择前几个特征构造交互
            numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
            if len(numeric_cols) >= 2:
                feature_pairs = [(numeric_cols[0], numeric_cols[1])]
                if len(numeric_cols) >= 4:
                    feature_pairs.append((numeric_cols[2], numeric_cols[3]))
        
        if feature_pairs:
            for feat1, feat2 in feature_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    interaction_name = f"{feat1}_x_{feat2}"
                    df_interaction[interaction_name] = df[feat1] * df[feat2]
                    print(f"构造交互特征: {interaction_name}")
        
        return df_interaction
    
    @staticmethod
    def statistical_features(df, window=3):
        """
        构造统计特征（滑动窗口统计）
        
        参数:
            df: 输入DataFrame（假设为时间序列）
            window: 滑动窗口大小
        
        返回:
            添加了统计特征的DataFrame
        """
        df_stats = df.copy()
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
        
        for col in numeric_cols:
            # 滑动窗口均值
            df_stats[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            # 滑动窗口标准差
            df_stats[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
            # 滑动窗口最大值
            df_stats[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
            # 滑动窗口最小值
            df_stats[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
        
        print(f"构造统计特征: 每个数值特征添加4个滑动窗口统计特征（窗口={window}）")
        
        return df_stats

# ============================================================================
# 8. 完整Pipeline演示
# ============================================================================
def run_complete_pipeline():
    """运行完整的特征工程Pipeline演示"""
    print("=" * 60)
    print("特征工程完整Pipeline演示")
    print("=" * 60)
    
    # 1. 生成数据
    X_df, y, feature_names = generate_sample_data(n_samples=500, n_features=15)
    
    # 2. 数据预处理
    print("\n" + "=" * 60)
    print("步骤1: 数据预处理")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    X_clean = preprocessor.handle_missing_values(X_df, strategy='mean')
    
    # 检测异常值
    outlier_mask = preprocessor.detect_outliers(X_clean, method='iqr', threshold=1.5)
    
    # 处理异常值
    X_clean = preprocessor.handle_outliers(X_clean, outlier_mask, strategy='cap')
    
    # 3. 特征编码（添加并处理分类特征）
    print("\n" + "=" * 60)
    print("步骤2: 特征编码")
    print("=" * 60)
    
    encoder = FeatureEncoder()
    X_with_cat = encoder.create_categorical_features(X_clean, n_categorical=2)
    X_encoded = encoder.one_hot_encode(X_with_cat)
    
    # 4. 特征缩放
    print("\n" + "=" * 60)
    print("步骤3: 特征缩放")
    print("=" * 60)
    
    scaler = FeatureScaler()
    X_scaled = scaler.standardize(X_encoded)
    
    # 5. 特征选择
    print("\n" + "=" * 60)
    print("步骤4: 特征选择")
    print("=" * 60)
    
    selector = FeatureSelector()
    X_selected, retained_indices = selector.variance_threshold(X_scaled, threshold=0.01)
    
    # 6. 特征提取
    print("\n" + "=" * 60)
    print("步骤5: 特征提取")
    print("=" * 60)
    
    extractor = FeatureExtractor()
    X_pca, pca_model = extractor.pca_extraction(X_selected, variance_ratio=0.9)
    
    # 7. 特征构造
    print("\n" + "=" * 60)
    print("步骤6: 特征构造")
    print("=" * 60)
    
    constructor = FeatureConstructor()
    X_final = constructor.polynomial_features(X_pca, degree=2, interaction_only=False)
    
    # 8. 结果展示
    print("\n" + "=" * 60)
    print("Pipeline完成总结")
    print("=" * 60)
    
    print(f"原始数据形状: {X_df.shape}")
    print(f"最终数据形状: {X_final.shape}")
    print(f"特征数量变化: {X_df.shape[1]} → {X_final.shape[1]}")
    
    # 可视化特征重要性（如果进行了模型选择）
    try:
        # 使用最终特征训练简单模型查看效果
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_final.values, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # 评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型性能评估:")
        print(f"逻辑回归准确率: {accuracy:.3f}")
        
    except Exception as e:
        print(f"模型训练评估跳过: {e}")
    
    return X_final, y

# ============================================================================
# 9. 主程序
# ============================================================================
if __name__ == "__main__":
    # 运行完整Pipeline
    X_final, y = run_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("特征工程Pipeline执行完成!")
    print("=" * 60)
    print("\n生成的文件包含以下模块:")
    print("1. 数据预处理: 缺失值处理、异常值检测")
    print("2. 特征编码: 独热编码、标签编码")
    print("3. 特征缩放: 标准化、归一化")
    print("4. 特征选择: 方差选择、相关系数选择、模型选择")
    print("5. 特征提取: PCA、LDA")
    print("6. 特征构造: 多项式特征、交互特征、统计特征")
    
    print("\n所有功能已模块化，可直接导入使用:")
    print("from 特征工程代码示例 import DataPreprocessor, FeatureScaler, ...")
    
    # 保存最终数据示例
    X_final.to_csv("outputs/阶段一/Week2/Day11/processed_features.csv", index=False)
    print(f"\n最终处理后的特征已保存至: outputs/阶段一/Week2/Day11/processed_features.csv")