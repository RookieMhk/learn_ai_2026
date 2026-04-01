#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复coze_learn目录下的乱码文件名
根据文件内容确定正确的文件名
"""

import os
import re

# 文件名映射表（乱码 -> 正确中文名）
# 基于文件内容分析得出的映射
filename_mapping = {
    # Day 1
    "浠婃棩鎵ц鍗＄墖妯℃澘.md": "今日执行卡片模板.md",
    "瀛︿範閲嶇偣鎸囧崡.md": "学习重点指南.md",
    "鐭ヨ瘑娴嬮獙.md": "知识测验.md",
    
    # Day 2
    "Day2浠婃棩鎵ц鍗＄墖妯℃澘.md": "Day2今日执行卡片模板.md",
    "Day2浠诲姟浜у嚭娓呭崟.md": "Day2任务产出清单.md",
    "鎬ц兘瀵规瘮瀹為獙鎸囧崡.md": "性能对比实验指南.md",
    "鐭╅樀杩愮畻瀹炴垬缁冧範.py": "矩阵运算实战练习.py",
    
    # Day 3
    "Day3浠婃棩鎵ц鍗＄墖妯℃澘.md": "Day3今日执行卡片模板.md",
    "姒傜巼缁熻鍩虹瀛︿範鎸囧崡.md": "概率统计基础学习指南.md",
    "姒傜巼鍒嗗竷璁＄畻缁冧範.py": "概率分布计算练习.py",
    
    # Day 4
    "Day4浠婃棩鎵ц鍗＄墖妯℃澘.md": "Day4今日执行卡片模板.md",
    "鏁版嵁闆嗚幏鍙栦笌鍔犺浇鎸囧崡.md": "数据集获取与加载指南.md",
    "鏁版嵁璐ㄩ噺鎶ュ憡妯℃澘.md": "数据质量报告模板.md",
    "鏁版嵁璐ㄩ噺鍒嗘瀽宸ュ叿.py": "数据质量分析工具.py",
    "鏁版嵁娓呮礂瀹炴垬鎵嬪唽.py": "数据清洗实战手册.py",
    
    # Day 5
    "Day5浠婃棩鎵ц鍗＄墖妯℃澘.md": "Day5今日执行卡片模板.md",
    
    # Day 6
    "Day6浠婃棩鎵ц鍗＄墖妯℃澘.md": "Day6今日执行卡片模板.md",
    "绾挎€у洖褰掔悊璁轰笌姊害涓嬮檷鍘熺悊澶嶄範.md": "线性回归理论与梯度下降原理复习.md",
    "姊害涓嬮檷浼樺寲鍙樹綋浠嬬粛.md": "梯度下降优化变体介绍.md",
    "鎵嬪姩瀹炵幇 vs Scikit-learn瀵规瘮鍒嗘瀽鎸囧崡.md": "手动实现 vs Scikit-learn对比分析指南.md",
}

def fix_filenames(base_dir):
    """修复指定目录下的乱码文件名"""
    fixed_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename in filename_mapping:
                old_path = os.path.join(root, filename)
                new_filename = filename_mapping[filename]
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"✅ 重命名: {filename} -> {new_filename}")
                    fixed_count += 1
                except Exception as e:
                    print(f"❌ 错误: 无法重命名 {filename}: {e}")
                    error_count += 1
    
    print(f"\n总结: 成功修复 {fixed_count} 个文件, 失败 {error_count} 个文件")
    return fixed_count, error_count

if __name__ == "__main__":
    base_dir = "coze_learn"
    if os.path.exists(base_dir):
        fix_filenames(base_dir)
    else:
        print(f"错误: 目录 {base_dir} 不存在")
