"""
概率分布计算练习
目标：通过实践掌握核心概率分布的特性与统计特征计算
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# 设置中文字体和美观的样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 60)
print("概率分布计算练习")
print("=" * 60)

# ============================================================================
# 练习1：正态分布（高斯分布）
# ============================================================================
print("\n📊 练习1：正态分布（高斯分布）")
print("-" * 40)

# 参数设置
mu1, sigma1 = 0, 1      # 标准正态分布
mu2, sigma2 = 3, 1.5    # 偏移的正态分布
mu3, sigma3 = -2, 0.8   # 较窄的正态分布

# 生成样本
np.random.seed(42)  # 设置随机种子确保结果可复现
samples1 = np.random.normal(mu1, sigma1, 1000)
samples2 = np.random.normal(mu2, sigma2, 1000)
samples3 = np.random.normal(mu3, sigma3, 1000)

# 计算统计特征
def calculate_statistics(samples, name):
    """计算基本统计特征"""
    mean_val = np.mean(samples)
    median_val = np.median(samples)
    std_val = np.std(samples)
    var_val = np.var(samples)
    skew_val = stats.skew(samples)
    kurt_val = stats.kurtosis(samples)
    
    print(f"\n{name}统计特征:")
    print(f"  均值: {mean_val:.4f}")
    print(f"  中位数: {median_val:.4f}")
    print(f"  标准差: {std_val:.4f}")
    print(f"  方差: {var_val:.4f}")
    print(f"  偏度: {skew_val:.4f} (对称性)")
    print(f"  峰度: {kurt_val:.4f} (尾部厚度)")
    
    return mean_val, std_val

# 计算三个分布的统计特征
mean1, std1 = calculate_statistics(samples1, "标准正态分布")
mean2, std2 = calculate_statistics(samples2, "偏移正态分布(μ=3)")
mean3, std3 = calculate_statistics(samples3, "窄峰正态分布(μ=-2)")

# 可视化：三个正态分布对比
plt.figure(figsize=(12, 5))

# 子图1：直方图与概率密度函数
plt.subplot(1, 2, 1)

# 绘制直方图
plt.hist(samples1, bins=30, density=True, alpha=0.6, color='skyblue', label='N(0,1)')
plt.hist(samples2, bins=30, density=True, alpha=0.6, color='salmon', label='N(3,1.5²)')
plt.hist(samples3, bins=30, density=True, alpha=0.6, color='lightgreen', label='N(-2,0.8²)')

# 绘制理论概率密度函数
x = np.linspace(-6, 8, 1000)
pdf1 = stats.norm.pdf(x, mu1, sigma1)
pdf2 = stats.norm.pdf(x, mu2, sigma2)
pdf3 = stats.norm.pdf(x, mu3, sigma3)

plt.plot(x, pdf1, 'blue', linewidth=2, label='PDF N(0,1)')
plt.plot(x, pdf2, 'red', linewidth=2, label='PDF N(3,1.5²)')
plt.plot(x, pdf3, 'green', linewidth=2, label='PDF N(-2,0.8²)')

plt.xlabel('值')
plt.ylabel('概率密度')
plt.title('正态分布对比：直方图 vs PDF')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：累积分布函数
plt.subplot(1, 2, 2)

cdf1 = stats.norm.cdf(x, mu1, sigma1)
cdf2 = stats.norm.cdf(x, mu2, sigma2)
cdf3 = stats.norm.cdf(x, mu3, sigma3)

plt.plot(x, cdf1, 'blue', linewidth=2, label='CDF N(0,1)')
plt.plot(x, cdf2, 'red', linewidth=2, label='CDF N(3,1.5²)')
plt.plot(x, cdf3, 'green', linewidth=2, label='CDF N(-2,0.8²)')

plt.xlabel('值')
plt.ylabel('累积概率')
plt.title('正态分布累积分布函数(CDF)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day3/正态分布对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 练习2：伯努利分布与二项分布
# ============================================================================
print("\n🎯 练习2：伯努利分布与二项分布")
print("-" * 40)

# 伯努利分布：单次试验
p_bernoulli = 0.7  # 成功概率
print(f"\n伯努利分布参数: 成功概率 p = {p_bernoulli}")

# 生成伯努利样本
np.random.seed(42)
bernoulli_samples = np.random.binomial(1, p_bernoulli, 1000)

# 统计成功失败次数
success_count = np.sum(bernoulli_samples)
failure_count = len(bernoulli_samples) - success_count

print(f"成功次数: {success_count} (观测概率: {success_count/len(bernoulli_samples):.3f})")
print(f"失败次数: {failure_count} (观测概率: {failure_count/len(bernoulli_samples):.3f})")
print(f"理论期望: 成功概率 = {p_bernoulli:.3f}, 失败概率 = {1-p_bernoulli:.3f}")

# 二项分布：n次独立伯努利试验
n_trials = 10  # 试验次数
p_binomial = 0.6  # 每次试验成功概率

print(f"\n二项分布参数: n = {n_trials}, p = {p_binomial}")
print(f"理论期望: 均值 = n*p = {n_trials * p_binomial:.1f}")
print(f"理论方差: n*p*(1-p) = {n_trials * p_binomial * (1-p_binomial):.1f}")

# 生成二项分布样本
np.random.seed(42)
binomial_samples = np.random.binomial(n_trials, p_binomial, 1000)

# 计算统计特征
binomial_mean = np.mean(binomial_samples)
binomial_var = np.var(binomial_samples)
binomial_std = np.std(binomial_samples)

print(f"样本统计: 均值 = {binomial_mean:.3f}, 方差 = {binomial_var:.3f}, 标准差 = {binomial_std:.3f}")

# 可视化伯努利与二项分布
plt.figure(figsize=(12, 5))

# 子图1：伯努利分布
plt.subplot(1, 2, 1)
bernoulli_counts = [failure_count, success_count]
bernoulli_labels = ['失败 (0)', '成功 (1)']
bernoulli_colors = ['lightcoral', 'lightgreen']

plt.bar(bernoulli_labels, bernoulli_counts, color=bernoulli_colors, alpha=0.7)
plt.xlabel('结果')
plt.ylabel('次数')
plt.title(f'伯努利分布: p = {p_bernoulli}')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, (label, count) in enumerate(zip(bernoulli_labels, bernoulli_counts)):
    plt.text(i, count + 10, f'{count}\n({count/len(bernoulli_samples):.1%})', 
             ha='center', va='bottom')

# 子图2：二项分布
plt.subplot(1, 2, 2)

# 计算理论概率
x_binomial = np.arange(0, n_trials + 1)
theoretical_probs = stats.binom.pmf(x_binomial, n_trials, p_binomial)

# 计算观测频率
unique, counts = np.unique(binomial_samples, return_counts=True)
observed_freq = np.zeros(len(x_binomial))
for i, val in enumerate(unique):
    idx = np.where(x_binomial == val)[0][0]
    observed_freq[idx] = counts[i] / len(binomial_samples)

# 绘制理论概率与观测频率
bar_width = 0.35
x_positions = np.arange(len(x_binomial))

plt.bar(x_positions - bar_width/2, theoretical_probs, width=bar_width, 
        alpha=0.7, color='skyblue', label='理论概率')
plt.bar(x_positions + bar_width/2, observed_freq, width=bar_width, 
        alpha=0.7, color='salmon', label='观测频率')

plt.xlabel('成功次数 (k)')
plt.ylabel('概率/频率')
plt.title(f'二项分布: n = {n_trials}, p = {p_binomial}')
plt.xticks(x_positions, x_binomial)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day3/伯努利与二项分布.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 练习3：多项式分布
# ============================================================================
print("\n🎲 练习3：多项式分布")
print("-" * 40)

# 模拟掷骰子试验
n_dice_rolls = 100  # 掷骰子次数
dice_probabilities = [1/6] * 6  # 均匀骰子，每个面概率1/6

print(f"\n多项式分布: 掷骰子 {n_dice_rolls} 次")
print(f"理论概率: 每个面 p = {1/6:.4f} (期望次数: {n_dice_rolls/6:.1f})")

# 生成多项式分布样本
np.random.seed(42)
multinomial_samples = np.random.multinomial(n_dice_rolls, dice_probabilities, size=1000)

# 计算平均次数
mean_counts = np.mean(multinomial_samples, axis=0)

print("\n前5次试验结果:")
for i in range(min(5, len(multinomial_samples))):
    print(f"  试验 {i+1}: {multinomial_samples[i]}")

print(f"\n1000次试验的平均次数:")
for i in range(6):
    print(f"  面 {i+1}: {mean_counts[i]:.2f} (理论: {n_dice_rolls/6:.2f})")

# 可视化多项式分布
plt.figure(figsize=(10, 6))

# 绘制第一个试验的柱状图
plt.subplot(2, 1, 1)
face_labels = [f'面 {i+1}' for i in range(6)]
colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid', 'lightcoral']

plt.bar(face_labels, multinomial_samples[0], color=colors, alpha=0.7)
plt.xlabel('骰子面')
plt.ylabel('出现次数')
plt.title(f'单个试验: 掷骰子 {n_dice_rolls} 次的结果')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, count in enumerate(multinomial_samples[0]):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

# 子图2：所有试验的平均分布
plt.subplot(2, 1, 2)
x_pos = np.arange(6)

plt.bar(x_pos - 0.2, mean_counts, width=0.4, alpha=0.7, color='skyblue', label='观测平均')
plt.bar(x_pos + 0.2, [n_dice_rolls/6] * 6, width=0.4, alpha=0.7, color='salmon', label='理论期望')

plt.xlabel('骰子面')
plt.ylabel('平均次数')
plt.title(f'1000次试验的平均分布 (标准差 = {np.std(mean_counts):.3f})')
plt.xticks(x_pos, face_labels)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day3/多项式分布.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 练习4：统计特征计算与相关性分析
# ============================================================================
print("\n📈 练习4：统计特征计算与相关性分析")
print("-" * 40)

# 生成两个相关变量
np.random.seed(42)
n_samples = 500

# 变量X：正态分布
X = np.random.normal(0, 1, n_samples)

# 变量Y：与X相关，加随机噪声
Y = 0.7 * X + np.random.normal(0, 0.5, n_samples)

# 变量Z：与X、Y无关
Z = np.random.normal(3, 1.2, n_samples)

print(f"\n生成数据统计:")
print(f"  样本数: {n_samples}")
print(f"  变量X: 均值 = {np.mean(X):.3f}, 标准差 = {np.std(X):.3f}")
print(f"  变量Y: 均值 = {np.mean(Y):.3f}, 标准差 = {np.std(Y):.3f}")
print(f"  变量Z: 均值 = {np.mean(Z):.3f}, 标准差 = {np.std(Z):.3f}")

# 计算相关系数
corr_xy = np.corrcoef(X, Y)[0, 1]
corr_xz = np.corrcoef(X, Z)[0, 1]
corr_yz = np.corrcoef(Y, Z)[0, 1]

print(f"\n相关系数矩阵:")
print(f"  corr(X,Y) = {corr_xy:.4f} (强正相关)")
print(f"  corr(X,Z) = {corr_xz:.4f} (弱相关)")
print(f"  corr(Y,Z) = {corr_yz:.4f} (弱相关)")

# 计算协方差矩阵
cov_matrix = np.cov(np.vstack([X, Y, Z]))

print(f"\n协方差矩阵:")
print("  " + str(cov_matrix).replace("\n", "\n  "))

# 可视化相关性分析
plt.figure(figsize=(14, 10))

# 子图1：散点图矩阵
plt.subplot(2, 2, 1)
plt.scatter(X, Y, alpha=0.5, color='skyblue', s=30)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'X vs Y (相关系数 = {corr_xy:.3f})')
plt.grid(True, alpha=0.3)

# 添加回归线
m, b = np.polyfit(X, Y, 1)
plt.plot(X, m*X + b, color='red', linewidth=2, label=f'Y = {m:.3f}X + {b:.3f}')
plt.legend()

# 子图2：X vs Z
plt.subplot(2, 2, 2)
plt.scatter(X, Z, alpha=0.5, color='lightgreen', s=30)
plt.xlabel('X')
plt.ylabel('Z')
plt.title(f'X vs Z (相关系数 = {corr_xz:.3f})')
plt.grid(True, alpha=0.3)

m, b = np.polyfit(X, Z, 1)
plt.plot(X, m*X + b, color='red', linewidth=2, label=f'Z = {m:.3f}X + {b:.3f}')
plt.legend()

# 子图3：Y vs Z
plt.subplot(2, 2, 3)
plt.scatter(Y, Z, alpha=0.5, color='salmon', s=30)
plt.xlabel('Y')
plt.ylabel('Z')
plt.title(f'Y vs Z (相关系数 = {corr_yz:.3f})')
plt.grid(True, alpha=0.3)

m, b = np.polyfit(Y, Z, 1)
plt.plot(Y, m*Y + b, color='red', linewidth=2, label=f'Z = {m:.3f}Y + {b:.3f}')
plt.legend()

# 子图4：协方差矩阵热图
plt.subplot(2, 2, 4)

# 创建DataFrame用于热图
corr_df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Z': Z
})

corr_matrix = corr_df.corr()

# 绘制热图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title('相关系数矩阵热图')

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day3/相关性分析.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 练习5：贝叶斯定理简单实现
# ============================================================================
print("\n🔮 练习5：贝叶斯定理简单实现")
print("-" * 40)

def bayes_update(prior, likelihood, evidence):
    """
    贝叶斯更新函数
    prior: 先验概率 P(A)
    likelihood: 似然概率 P(B|A)
    evidence: 证据概率 P(B)
    返回: 后验概率 P(A|B)
    """
    posterior = (likelihood * prior) / evidence
    return posterior

# 示例1：垃圾邮件过滤
print("\n示例1：垃圾邮件过滤")
print("-" * 30)

# 先验概率：历史数据中垃圾邮件的比例
P_spam = 0.3  # 30%的邮件是垃圾邮件

# 似然概率：垃圾邮件中出现"优惠"这个词的概率
P_offer_given_spam = 0.8  # 80%的垃圾邮件包含"优惠"

# 证据概率：任意邮件中出现"优惠"的总概率
# P(优惠) = P(优惠|垃圾邮件)*P(垃圾邮件) + P(优惠|正常邮件)*P(正常邮件)
P_offer_given_ham = 0.1  # 10%的正常邮件包含"优惠"
P_offer = P_offer_given_spam * P_spam + P_offer_given_ham * (1 - P_spam)

# 计算后验概率：给定邮件包含"优惠"，它是垃圾邮件的概率
P_spam_given_offer = bayes_update(P_spam, P_offer_given_spam, P_offer)

print(f"先验概率 P(垃圾邮件) = {P_spam:.3f}")
print(f"似然概率 P(优惠|垃圾邮件) = {P_offer_given_spam:.3f}")
print(f"证据概率 P(优惠) = {P_offer:.3f}")
print(f"后验概率 P(垃圾邮件|优惠) = {P_spam_given_offer:.3f}")

# 可视化贝叶斯更新
plt.figure(figsize=(10, 6))

prior = P_spam
posterior = P_spam_given_offer

plt.bar(['先验概率', '后验概率'], [prior, posterior], color=['lightblue', 'lightcoral'], alpha=0.7)
plt.ylabel('概率')
plt.title('贝叶斯更新: 先验 vs 后验')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, (label, prob) in enumerate(zip(['先验概率', '后验概率'], [prior, posterior])):
    plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/阶段一/Week1/Day3/贝叶斯更新示例.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 总结与思考题
# ============================================================================
print("\n" + "=" * 60)
print("🎓 总结与思考题")
print("=" * 60)

print("\n✅ 今日学习总结:")
print("  1. 掌握了正态、伯努利、多项式三大核心分布的特性")
print("  2. 学会了计算均值、方差、标准差、相关系数等统计特征")
print("  3. 理解了贝叶斯定理在AI中的实际应用意义")
print("  4. 实践了概率分布的可视化与数据分析")

print("\n🤔 思考题:")
print("  1. 在机器学习中，为什么经常假设误差项服从正态分布？")
print("  2. 贝叶斯定理如何帮助我们结合先验知识与新证据？")
print("  3. 相关性不等于因果性，如何设计实验验证因果关系？")
print("  4. 如何将今日学到的概率统计知识应用到线性回归项目中？")

print("\n📚 扩展实践建议:")
print("  1. 尝试用真实数据集重复今日的统计分析")
print("  2. 实现更复杂的贝叶斯更新，如多类别分类")
print("  3. 探索不同解码策略对LLM生成质量的影响")

print("\n✨ 恭喜完成今日概率统计学习！")
print("明日将进入Pandas数据清洗实战，应用今日所学。")