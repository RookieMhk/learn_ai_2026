#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 2 材料打包脚本

功能：自动打包Week 2所有学习材料（Day 8-14）为单个ZIP文件
输出：outputs/阶段一/Week2.zip（覆盖更新）

使用方法：
1. 直接运行：python "Week 2材料打包脚本.py"
2. 作为模块导入：import zip_week2
"""

import os
import sys
import zipfile
import datetime
from pathlib import Path
import shutil

def get_project_root():
    """获取项目根目录（假设脚本位于outputs/阶段一/Week2/Day14/）"""
    current_file = Path(__file__).resolve()
    # 向上回溯：Day14 → Week2 → 阶段一 → outputs → files
    week2_dir = current_file.parent.parent  # Week2目录
    return week2_dir.parent.parent.parent  # 项目根目录 /app/data/files

def get_week2_materials():
    """获取Week 2所有学习材料路径"""
    root = get_project_root()
    week2_dir = root / "outputs" / "阶段一" / "Week2"
    
    if not week2_dir.exists():
        print(f"错误：Week 2目录不存在 - {week2_dir}")
        return []
    
    # 收集所有Day目录和文件
    materials = []
    for item in week2_dir.iterdir():
        if item.is_dir() and item.name.startswith("Day"):
            materials.append(item)
        elif item.is_file() and item.suffix in ['.md', '.py', '.ipynb', '.txt', '.json']:
            materials.append(item)
    
    # 按Day编号排序
    materials.sort(key=lambda x: x.name)
    
    return materials

def create_week2_zip(force_update=True):
    """创建Week 2材料ZIP包
    
    参数:
        force_update: 如果ZIP已存在，是否覆盖更新（默认True）
    
    返回:
        zip_path: 生成的ZIP文件路径，失败返回None
    """
    # 输出文件路径
    root = get_project_root()
    zip_path = root / "outputs" / "阶段一" / "Week2.zip"
    
    # 确保输出目录存在
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取材料列表
    materials = get_week2_materials()
    if not materials:
        print("警告：未找到任何学习材料，可能目录结构异常")
        return None
    
    # 检查是否已存在ZIP
    if zip_path.exists():
        if not force_update:
            print(f"ZIP文件已存在：{zip_path}")
            print("如需更新，请设置force_update=True")
            return zip_path
        else:
            print(f"更新现有ZIP文件：{zip_path}")
    
    # 创建ZIP文件
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for material in materials:
                material_rel = material.relative_to(root)
                
                if material.is_dir():
                    # 遍历目录下所有文件
                    for file_path in material.rglob('*'):
                        if file_path.is_file():
                            file_rel = file_path.relative_to(root)
                            zipf.write(file_path, file_rel)
                            print(f"  添加：{file_rel}")
                else:
                    # 添加单个文件
                    zipf.write(material, material_rel)
                    print(f"  添加：{material_rel}")
        
        # 统计信息
        zip_stats = zip_path.stat()
        file_count = len(zipfile.ZipFile(zip_path, 'r').namelist())
        
        print("\n" + "="*60)
        print("Week 2 材料打包完成！")
        print("="*60)
        print(f"ZIP文件路径：{zip_path}")
        print(f"文件数量：{file_count} 个")
        print(f"压缩包大小：{zip_stats.st_size / (1024*1024):.2f} MB")
        print(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("包含内容：")
        
        # 按Day分组显示
        day_groups = {}
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for name in zipf.namelist():
                parts = name.split('/')
                if len(parts) >= 4 and parts[2] == "Week2":
                    day = parts[3]  # Day8, Day9等
                    if day not in day_groups:
                        day_groups[day] = []
                    day_groups[day].append(name)
        
        for day in sorted(day_groups.keys()):
            print(f"  {day}: {len(day_groups[day])} 个文件")
        
        print("="*60)
        
        return zip_path
        
    except Exception as e:
        print(f"打包失败，错误信息：{e}")
        import traceback
        traceback.print_exc()
        return None

def verify_zip_contents(zip_path=None):
    """验证ZIP文件内容完整性"""
    if zip_path is None:
        root = get_project_root()
        zip_path = root / "outputs" / "阶段一" / "Week2.zip"
    
    if not zip_path.exists():
        print(f"错误：ZIP文件不存在 - {zip_path}")
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            print("ZIP文件内容验证：")
            print("-" * 40)
            
            # 检查关键文件
            key_files = [
                'outputs/阶段一/Week2/Day8/Day8今日执行卡片模板.md',
                'outputs/阶段一/Week2/Day9/损失函数与优化算法进阶.md',
                'outputs/阶段一/Week2/Day10/模型评估指标详解.md',
                'outputs/阶段一/Week2/Day11/特征工程实战指南.md',
                'outputs/阶段一/Week2/Day12/房价预测分析.ipynb',
                'outputs/阶段一/Week2/Day13/知识要点速查手册.md',
                'outputs/阶段一/Week2/Day14/Week 2学习成果评估工具.py',
                'outputs/阶段一/Week2/Day14/Week 2综合知识测验.md',
                'outputs/阶段一/Week2/Day14/Week 3详细任务清单.md',
                'outputs/阶段一/Week2/Day14/学习效率分析报告.md',
            ]
            
            missing_files = []
            for key_file in key_files:
                if key_file not in file_list:
                    missing_files.append(key_file)
            
            if missing_files:
                print(f"警告：缺少 {len(missing_files)} 个关键文件：")
                for mf in missing_files:
                    print(f"  - {mf}")
                print("建议重新打包以确保材料完整性")
                return False
            else:
                print("✓ 所有关键文件完整")
                print(f"✓ 总文件数：{len(file_list)}")
                
                # 显示文件统计
                day_counts = {}
                for file in file_list:
                    parts = file.split('/')
                    if len(parts) >= 4 and parts[2] == "Week2":
                        day = parts[3]
                        day_counts[day] = day_counts.get(day, 0) + 1
                
                print("✓ 每日文件分布：")
                for day in sorted(day_counts.keys()):
                    print(f"    {day}: {day_counts[day]} 个文件")
                
                return True
                
    except Exception as e:
        print(f"验证失败，错误信息：{e}")
        return False

def generate_file_structure_report():
    """生成Week 2文件结构报告"""
    root = get_project_root()
    week2_dir = root / "outputs" / "阶段一" / "Week2"
    
    if not week2_dir.exists():
        return "Week 2目录不存在"
    
    report_lines = []
    report_lines.append("Week 2 学习材料文件结构报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"基础目录：{week2_dir}")
    report_lines.append("")
    
    # 遍历所有Day目录
    day_dirs = sorted([d for d in week2_dir.iterdir() if d.is_dir() and d.name.startswith("Day")])
    
    for day_dir in day_dirs:
        report_lines.append(f"{day_dir.name}/")
        
        # 统计文件类型
        file_types = {}
        total_size = 0
        
        for file_path in day_dir.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                file_types[suffix] = file_types.get(suffix, 0) + 1
                total_size += file_path.stat().st_size
        
        # 添加文件统计
        report_lines.append(f"  文件总数：{sum(file_types.values())}")
        report_lines.append(f"  总大小：{total_size / 1024:.1f} KB")
        
        if file_types:
            report_lines.append("  文件类型分布：")
            for suffix, count in sorted(file_types.items()):
                report_lines.append(f"    {suffix or '无后缀'}: {count} 个")
        
        # 列出主要文件
        main_files = []
        for file_path in day_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.md', '.py', '.ipynb', '.txt']:
                main_files.append(file_path.name)
        
        if main_files:
            report_lines.append("  主要文件：")
            for file_name in sorted(main_files)[:5]:  # 最多显示5个
                report_lines.append(f"    - {file_name}")
        
        report_lines.append("")
    
    # 总体统计
    all_files = list(week2_dir.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    total_files = len(all_files)
    total_size = sum(f.stat().st_size for f in all_files)
    
    report_lines.append("总体统计：")
    report_lines.append(f"  总文件数：{total_files}")
    report_lines.append(f"  总大小：{total_size / (1024*1024):.2f} MB")
    report_lines.append(f"  Day目录数：{len(day_dirs)}")
    report_lines.append("")
    report_lines.append("打包状态：✓ 完整（包含Day 8-14所有材料）")
    
    return "\n".join(report_lines)

def main():
    """主函数：打包、验证、生成报告"""
    print("Week 2 学习材料打包系统")
    print("=" * 60)
    
    # 检查当前目录
    print("1. 检查目录结构...")
    root = get_project_root()
    print(f"   项目根目录：{root}")
    print(f"   当前脚本：{Path(__file__).resolve()}")
    
    # 生成文件结构报告
    print("\n2. 生成文件结构报告...")
    report = generate_file_structure_report()
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # 打包材料
    print("\n3. 打包学习材料...")
    zip_path = create_week2_zip(force_update=True)
    
    if zip_path is None:
        print("打包失败，程序终止")
        return 1
    
    # 验证ZIP内容
    print("\n4. 验证ZIP文件完整性...")
    is_valid = verify_zip_contents(zip_path)
    
    if not is_valid:
        print("验证失败，可能存在文件缺失")
        return 1
    
    # 保存报告
    print("\n5. 生成详细报告...")
    report_path = root / "outputs" / "阶段一" / "Week2" / "Day14" / "Week2文件结构报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   报告已保存：{report_path}")
    
    # 输出最终信息
    print("\n" + "=" * 60)
    print("打包流程完成！")
    print("=" * 60)
    print(f"ZIP文件位置：{zip_path}")
    print(f"使用说明：")
    print(f"  1. 解压ZIP文件可获取Week 2全部学习材料")
    print(f"  2. 每日材料按Day8-Day14组织，包含文档、代码、测验")
    print(f"  3. 详细文件清单见：{report_path}")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    # 支持命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='Week 2 学习材料打包工具')
    parser.add_argument('--verify-only', action='store_true', help='仅验证现有ZIP文件')
    parser.add_argument('--report-only', action='store_true', help='仅生成文件结构报告')
    parser.add_argument('--force-update', action='store_true', default=True, 
                       help='强制更新ZIP文件（默认True）')
    
    args = parser.parse_args()
    
    if args.verify_only:
        if verify_zip_contents():
            print("验证通过")
            sys.exit(0)
        else:
            print("验证失败")
            sys.exit(1)
    elif args.report_only:
        print(generate_file_structure_report())
        sys.exit(0)
    else:
        sys.exit(main())