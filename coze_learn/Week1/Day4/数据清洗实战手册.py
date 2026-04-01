"""
泰坦尼克号数据清洗实战手册
===================================

本手册提供了完整的数据清洗流程，涵盖：
1. 缺失值处理（删除、填充、插值）
2. 异常值识别与处理
3. 数据格式转换
4. 特征工程基础（特征创建、特征缩放、特征选择）

使用方法：
1. 首先运行数据集获取与加载指南中的代码加载数据
2. 运行本脚本中的清洗函数
3. 结合数据质量分析工具评估清洗效果
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 数据加载与初步探索
# ============================================================================

def load_titanic_data(method='seaborn'):
    """
    加载泰坦尼克号数据集
    
    参数：
    method: 加载方式，可选 'seaborn'（推荐）、'url'、'kaggle'
    
    返回：
    DataFrame格式的泰坦尼克号数据
    """
    if method == 'seaborn':
        try:
            import seaborn as sns
            df = sns.load_dataset('titanic')
            print(f"使用Seaborn加载成功，数据形状: {df.shape}")
            return df
        except ImportError:
            print("Seaborn未安装，尝试使用URL方式加载...")
            method = 'url'
    
    if method == 'url':
        # GitHub托管的泰坦尼克号数据集
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        try:
            df = pd.read_csv(url)
            print(f"从URL加载成功，数据形状: {df.shape}")
            return df
        except:
            print("URL加载失败，尝试本地加载...")
            # 如果网络不可用，可以手动下载后加载本地文件
            try:
                df = pd.read_csv('titanic.csv')
                print(f"从本地文件加载成功，数据形状: {df.shape}")
                return df
            except:
                print("所有加载方式均失败，请检查网络或文件路径")
                return None
    
    if method == 'kaggle':
        try:
            df = pd.read_csv('train.csv')
            print(f"从Kaggle文件加载成功，数据形状: {df.shape}")
            return df
        except:
            print("Kaggle文件加载失败，请确保train.csv在正确路径")
            return None

# ============================================================================
# 2. 缺失值处理
# ============================================================================

def analyze_missing_data(df):
    """
    分析数据集中缺失值情况
    
    参数：
    df: 待分析的DataFrame
    
    返回：
    包含缺失值统计信息的DataFrame
    """
    missing_stats = pd.DataFrame({
        '缺失值数量': df.isnull().sum(),
        '缺失值比例': df.isnull().sum() / len(df) * 100
    })
    missing_stats = missing_stats[missing_stats['缺失值数量'] > 0].sort_values('缺失值比例', ascending=False)
    
    print("缺失值分析报告:")
    print("=" * 50)
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"有缺失值的列数: {len(missing_stats)}")
    print("\n各列缺失情况:")
    for idx, row in missing_stats.iterrows():
        print(f"  {idx}: {row['缺失值数量']}个缺失 ({row['缺失值比例']:.1f}%)")
    
    return missing_stats

def handle_missing_values(df, strategy='smart'):
    """
    处理缺失值
    
    参数：
    df: 待处理的DataFrame
    strategy: 处理策略，可选 'smart'（智能处理）、'drop'、'fill'、'impute'
    
    返回：
    处理后的DataFrame
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        # 简单删除包含缺失值的行
        original_len = len(df_clean)
        df_clean = df_clean.dropna()
        print(f"删除包含缺失值的行: {original_len} → {len(df_clean)} 行")
        return df_clean
    
    elif strategy == 'fill':
        # 简单填充
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
        df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
        df_clean['Cabin'] = df_clean['Cabin'].fillna('Unknown')
        print("使用中位数/众数填充缺失值")
        return df_clean
    
    elif strategy == 'impute':
        # 使用插值或更高级的方法
        from sklearn.impute import SimpleImputer
        
        # 数值型特征使用中位数插值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        imputer_numeric = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer_numeric.fit_transform(df_clean[numeric_cols])
        
        # 分类型特征使用众数插值
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        print("使用sklearn插值器处理缺失值")
        return df_clean
    
    elif strategy == 'smart':
        # 智能处理：根据列特征选择最合适的处理方式
        print("执行智能缺失值处理...")
        
        # 1. Cabin列：缺失太多，创建新类别
        df_clean['Cabin'] = df_clean['Cabin'].fillna('Unknown')
        print("  - Cabin列: 缺失过多，填充为'Unknown'")
        
        # 2. Age列：使用中位数填充（或基于其他特征预测，这里简化）
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
        print(f"  - Age列: 使用中位数({df_clean['Age'].median():.1f})填充")
        
        # 3. Embarked列：使用众数填充
        embarked_mode = df_clean['Embarked'].mode()
        if not embarked_mode.empty:
            df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode[0])
            print(f"  - Embarked列: 使用众数({embarked_mode[0]})填充")
        else:
            df_clean['Embarked'] = df_clean['Embarked'].fillna('S')
            print("  - Embarked列: 默认填充为'S'")
        
        # 4. Fare列：如果有缺失，使用中位数填充
        if df_clean['Fare'].isnull().any():
            df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
            print(f"  - Fare列: 使用中位数({df_clean['Fare'].median():.2f})填充")
        
        print("智能缺失值处理完成")
        return df_clean
    
    else:
        print(f"未知策略: {strategy}，返回原始数据")
        return df_clean

# ============================================================================
# 3. 异常值识别与处理
# ============================================================================

def detect_outliers(df, method='iqr', threshold=1.5):
    """
    识别数值型特征中的异常值
    
    参数：
    df: 待分析的DataFrame
    method: 检测方法，可选 'iqr'（四分位距）、'zscore'（Z分数）
    threshold: 阈值，IQR方法默认为1.5，Z分数默认为3
    
    返回：
    包含异常值检测结果的字典
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if method == 'iqr':
            # IQR方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            # Z分数方法
            from scipy import stats
            z_scores = np.abs(stats.zscore(data))
            outliers = df.iloc[np.where(z_scores > threshold)[0]]
        
        else:
            print(f"未知方法: {method}")
            continue
        
        outlier_count = len(outliers)
        total_count = len(df[col].dropna())
        
        if total_count > 0:
            percentage = outlier_count / total_count * 100
        else:
            percentage = 0
        
        outliers_info[col] = {
            '异常值数量': outlier_count,
            '异常值比例': percentage,
            '异常值索引': outliers.index.tolist(),
            '异常值数据': outliers[col].tolist()
        }
        
        print(f"{col}: 异常值{outlier_count}个 ({percentage:.1f}%)")
    
    return outliers_info

def handle_outliers(df, strategy='cap', outlier_info=None):
    """
    处理异常值
    
    参数：
    df: 待处理的DataFrame
    strategy: 处理策略，可选 'cap'（缩尾处理）、'remove'（删除）、'transform'（变换）
    outlier_info: 异常值检测结果，如果为None则自动检测
    
    返回：
    处理后的DataFrame
    """
    df_clean = df.copy()
    
    if outlier_info is None:
        print("未提供异常值信息，使用IQR方法自动检测...")
        outlier_info = detect_outliers(df_clean, method='iqr')
    
    for col, info in outlier_info.items():
        if info['异常值数量'] == 0:
            continue
        
        if strategy == 'cap':
            # 缩尾处理：将异常值替换为边界值
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 确保边界值在合理范围内
            lower_bound = max(lower_bound, df_clean[col].min())
            upper_bound = min(upper_bound, df_clean[col].max())
            
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            print(f"  - {col}: 缩尾处理，边界[{lower_bound:.2f}, {upper_bound:.2f}]")
        
        elif strategy == 'remove':
            # 删除包含异常值的行
            outliers_idx = info['异常值索引']
            df_clean = df_clean.drop(outliers_idx)
            print(f"  - {col}: 删除{len(outliers_idx)}个异常值所在行")
        
        elif strategy == 'transform':
            # 使用对数变换处理右偏分布
            if df_clean[col].min() > 0:  # 对数变换要求正值
                df_clean[f'{col}_log'] = np.log1p(df_clean[col])
                print(f"  - {col}: 应用对数变换，创建新列{col}_log")
            else:
                print(f"  - {col}: 无法应用对数变换（存在非正值），跳过")
    
    return df_clean

# ============================================================================
# 4. 数据格式转换
# ============================================================================

def convert_data_types(df):
    """
    优化数据类型，减少内存占用并提高处理效率
    
    参数：
    df: 待处理的DataFrame
    
    返回：
    转换后的DataFrame
    """
    df_clean = df.copy()
    original_memory = df_clean.memory_usage().sum() / 1024**2
    
    # 分类变量转换
    categorical_cols = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    for col in categorical_cols:
        if col in df_clean.columns:
            if df_clean[col].nunique() < 50:  # 唯一值较少时转换为category
                df_clean[col] = df_clean[col].astype('category')
    
    # 整数类型优化
    int_cols = df_clean.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        
        if col_min >= -128 and col_max <= 127:
            df_clean[col] = df_clean[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df_clean[col] = df_clean[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df_clean[col] = df_clean[col].astype('int32')
    
    # 浮点数优化
    float_cols = df_clean.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_clean[col] = df_clean[col].astype('float32')
    
    new_memory = df_clean.memory_usage().sum() / 1024**2
    print(f"内存占用优化: {original_memory:.2f}MB → {new_memory:.2f}MB (减少{(original_memory - new_memory)/original_memory*100:.1f}%)")
    
    return df_clean

# ============================================================================
# 5. 特征工程基础
# ============================================================================

def create_features(df):
    """
    创建新特征，丰富数据信息
    
    参数：
    df: 待处理的DataFrame
    
    返回：
    添加了新特征的DataFrame
    """
    df_enhanced = df.copy()
    
    # 1. 从姓名中提取称谓（Mr, Mrs, Miss等）
    if 'Name' in df_enhanced.columns:
        df_enhanced['Title'] = df_enhanced['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # 简化称谓类别
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Lady': 'Rare',
            'Countess': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Jonkheer': 'Rare',
            'Sir': 'Rare', 'Capt': 'Rare'
        }
        df_enhanced['Title'] = df_enhanced['Title'].map(title_mapping)
        df_enhanced['Title'] = df_enhanced['Title'].fillna('Rare')
    
    # 2. 家庭大小（兄弟姐妹+父母子女+自己）
    if all(col in df_enhanced.columns for col in ['SibSp', 'Parch']):
        df_enhanced['FamilySize'] = df_enhanced['SibSp'] + df_enhanced['Parch'] + 1
        # 家庭大小分类
        df_enhanced['FamilyType'] = pd.cut(df_enhanced['FamilySize'], 
                                           bins=[0, 1, 4, 20], 
                                           labels=['单身', '小家庭', '大家庭'])
    
    # 3. 是否独自旅行
    df_enhanced['IsAlone'] = (df_enhanced['FamilySize'] == 1).astype(int)
    
    # 4. 票价区间
    if 'Fare' in df_enhanced.columns:
        df_enhanced['FareCategory'] = pd.qcut(df_enhanced['Fare'], 4, labels=['低价', '中低价', '中高价', '高价'])
    
    # 5. 年龄分段
    if 'Age' in df_enhanced.columns:
        bins = [0, 12, 18, 35, 60, 100]
        labels = ['儿童', '青少年', '青年', '中年', '老年']
        df_enhanced['AgeGroup'] = pd.cut(df_enhanced['Age'], bins=bins, labels=labels, right=False)
    
    # 6. 船舱号提取（首字母表示甲板）
    if 'Cabin' in df_enhanced.columns:
        df_enhanced['Deck'] = df_enhanced['Cabin'].str[0]
        df_enhanced['Deck'] = df_enhanced['Deck'].fillna('Unknown')
    
    print(f"创建了{len(df_enhanced.columns) - len(df.columns)}个新特征")
    return df_enhanced

def scale_features(df, method='standard'):
    """
    特征缩放
    
    参数：
    df: 待处理的DataFrame
    method: 缩放方法，可选 'standard'（标准化）、'minmax'（归一化）、'robust'（鲁棒缩放）
    
    返回：
    缩放后的DataFrame（仅数值型特征）
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        print(f"未知缩放方法: {method}，使用标准化")
        scaler = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print(f"使用{method}方法缩放{numeric_cols.size}个数值特征")
    return df_scaled

def select_features(df, target_col='Survived', method='correlation', top_k=10):
    """
    特征选择
    
    参数：
    df: 待处理的DataFrame
    target_col: 目标变量列名
    method: 选择方法，可选 'correlation'（相关性）、'importance'（树模型重要性）
    top_k: 选择前k个特征
    
    返回：
    特征重要性/相关性排序
    """
    df_temp = df.copy()
    
    # 确保目标变量存在且是数值型
    if target_col not in df_temp.columns:
        print(f"目标变量{target_col}不存在")
        return None
    
    # 处理分类变量：独热编码
    categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df_temp, columns=categorical_cols, drop_first=True)
    
    if method == 'correlation':
        # 基于相关性
        if target_col in df_encoded.columns:
            correlations = df_encoded.corr()[target_col].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != target_col]
            
            print(f"与{target_col}相关性最高的{top_k}个特征:")
            for i, (feature, corr) in enumerate(correlations.head(top_k).items(), 1):
                print(f"  {i:2d}. {feature}: {corr:.3f}")
            
            return correlations
    
    elif method == 'importance':
        # 基于树模型特征重要性
        from sklearn.ensemble import RandomForestClassifier
        
        # 分离特征和目标
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 特征重要性
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"随机森林特征重要性最高的{top_k}个特征:")
        for i, row in importances.head(top_k).iterrows():
            print(f"  {row.name+1:2d}. {row['feature']}: {row['importance']:.3f}")
        
        return importances
    
    else:
        print(f"未知特征选择方法: {method}")
        return None

# ============================================================================
# 6. 完整清洗流程
# ============================================================================

def complete_cleaning_pipeline(df, missing_strategy='smart', outlier_strategy='cap', 
                               create_new_features=True, scale_method=None):
    """
    完整的数据清洗管道
    
    参数：
    df: 原始数据
    missing_strategy: 缺失值处理策略
    outlier_strategy: 异常值处理策略
    create_new_features: 是否创建新特征
    scale_method: 特征缩放方法，None表示不缩放
    
    返回：
    清洗后的DataFrame
    """
    print("=" * 60)
    print("开始数据清洗管道")
    print("=" * 60)
    
    # 1. 缺失值处理
    print("\n1. 缺失值处理...")
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    
    # 2. 异常值处理
    print("\n2. 异常值处理...")
    outlier_info = detect_outliers(df_clean, method='iqr')
    df_clean = handle_outliers(df_clean, strategy=outlier_strategy, outlier_info=outlier_info)
    
    # 3. 数据格式转换
    print("\n3. 数据格式转换...")
    df_clean = convert_data_types(df_clean)
    
    # 4. 特征工程
    if create_new_features:
        print("\n4. 特征工程...")
        df_clean = create_features(df_clean)
    
    # 5. 特征缩放
    if scale_method:
        print(f"\n5. 特征缩放 ({scale_method})...")
        df_clean = scale_features(df_clean, method=scale_method)
    
    print("\n" + "=" * 60)
    print("数据清洗管道完成!")
    print(f"原始数据形状: {df.shape}")
    print(f"清洗后数据形状: {df_clean.shape}")
    print(f"原始特征数: {len(df.columns)}")
    print(f"清洗后特征数: {len(df_clean.columns)}")
    print("=" * 60)
    
    return df_clean

# ============================================================================
# 7. 主程序：示例使用
# ============================================================================

if __name__ == "__main__":
    # 示例：完整清洗流程
    print("泰坦尼克号数据清洗实战示例")
    print("-" * 40)
    
    # 1. 加载数据
    print("加载数据...")
    df = load_titanic_data(method='seaborn')
    
    if df is not None:
        # 2. 初始数据探索
        print("\n初始数据探索:")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n数据类型:")
        print(df.dtypes)
        
        # 3. 缺失值分析
        print("\n缺失值分析:")
        missing_stats = analyze_missing_data(df)
        
        # 4. 运行完整清洗管道
        print("\n运行完整清洗管道...")
        df_clean = complete_cleaning_pipeline(
            df, 
            missing_strategy='smart',
            outlier_strategy='cap',
            create_new_features=True,
            scale_method=None  # 可选 'standard', 'minmax', 'robust'
        )
        
        # 5. 特征选择示例
        print("\n特征选择示例（基于相关性）:")
        feature_importance = select_features(
            df_clean, 
            target_col='survived',  # Seaborn数据集列名是小写
            method='correlation',
            top_k=10
        )
        
        # 6. 保存清洗后的数据
        save_path = 'titanic_cleaned.csv'
        df_clean.to_csv(save_path, index=False)
        print(f"\n清洗后的数据已保存到: {save_path}")
        
        print("\n数据清洗实战完成！")