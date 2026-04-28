#!/usr/bin/env python3
"""
Week 3 学习材料打包脚本
功能：将Week 3所有学习材料打包为Week3.zip
"""

import os
import sys
import zipfile
import datetime
from pathlib import Path

class Week3Packager:
    """Week 3 学习材料打包器"""
    
    def __init__(self, week3_path: str = "outputs/阶段一/Week3"):
        """
        初始化打包器
        
        Args:
            week3_path: Week 3 学习材料的根目录路径
        """
        self.week3_path = Path(week3_path)
        self.output_zip = self.week3_path.parent / "Week3.zip"
        
        # Week 3 所有学习日
        self.days = ["Day15", "Day16", "Day17", "Day18", "Day19", "Day20", "Day21"]
        
        # 需要排除的文件模式
        self.exclude_patterns = [
            "*.zip",  # 排除已有的zip文件
            "*.pyc",  # 排除Python字节码
            "__pycache__",  # 排除缓存目录
            ".DS_Store",  # 排除macOS系统文件
            "Thumbs.db",  # 排除Windows缩略图文件
        ]
    
    def collect_files(self) -> list:
        """
        收集需要打包的所有文件
        
        Returns:
            文件路径列表
        """
        all_files = []
        
        # 遍历Week 3目录下的所有文件
        for day in self.days:
            day_path = self.week3_path / day
            if not day_path.exists():
                print(f"警告: {day} 目录不存在")
                continue
            
            # 收集该目录下的所有文件
            for file_path in day_path.rglob("*"):
                if file_path.is_file():
                    # 检查是否应该排除
                    if self._should_exclude(file_path):
                        continue
                    all_files.append(file_path)
        
        # 添加Week 3根目录下的文件（除了zip文件）
        for file_path in self.week3_path.rglob("*"):
            if file_path.is_file() and file_path.suffix != ".zip":
                if self._should_exclude(file_path):
                    continue
                all_files.append(file_path)
        
        # 去重
        all_files = list(set(all_files))
        
        # 按路径排序
        all_files.sort()
        
        return all_files
    
    def _should_exclude(self, file_path: Path) -> bool:
        """
        判断文件是否应该排除
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否应该排除
        """
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if pattern.startswith("*."):
                # 文件扩展名匹配
                if file_path.suffix == pattern[1:]:
                    return True
            elif pattern in str(file_path):
                # 路径包含匹配
                return True
        
        return False
    
    def create_zip(self, files: list) -> bool:
        """
        创建zip文件
        
        Args:
            files: 需要打包的文件列表
            
        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            self.output_zip.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建zip文件
            with zipfile.ZipFile(self.output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    # 计算在zip中的相对路径
                    try:
                        relative_path = file_path.relative_to(self.week3_path.parent)
                    except ValueError:
                        # 如果文件不在Week3目录下，使用相对于项目根目录的路径
                        relative_path = file_path.relative_to(Path.cwd())
                    
                    # 添加文件到zip
                    zipf.write(file_path, relative_path)
                    print(f"已添加: {relative_path}")
            
            return True
            
        except Exception as e:
            print(f"创建zip文件时出错: {e}")
            return False
    
    def verify_zip(self) -> tuple:
        """
        验证zip文件
        
        Returns:
            (是否有效, 文件数量, 总大小)
        """
        if not self.output_zip.exists():
            return False, 0, 0
        
        try:
            with zipfile.ZipFile(self.output_zip, 'r') as zipf:
                file_list = zipf.namelist()
                total_size = sum(zipf.getinfo(f).file_size for f in file_list)
                
                return True, len(file_list), total_size
                
        except Exception as e:
            print(f"验证zip文件时出错: {e}")
            return False, 0, 0
    
    def generate_report(self, files: list, success: bool) -> str:
        """
        生成打包报告
        
        Args:
            files: 打包的文件列表
            success: 是否成功
            
        Returns:
            报告内容
        """
        report_lines = []
        
        # 报告头部
        report_lines.append("=" * 60)
        report_lines.append("Week 3 学习材料打包报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"打包状态: {'成功' if success else '失败'}")
        report_lines.append("")
        
        # 文件统计
        report_lines.append("文件统计:")
        report_lines.append(f"  总文件数: {len(files)}")
        
        # 按类型统计
        file_types = {}
        for file_path in files:
            ext = file_path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        report_lines.append("  文件类型分布:")
        for ext, count in sorted(file_types.items()):
            report_lines.append(f"    {ext if ext else '无扩展名'}: {count}个")
        
        # 按学习日统计
        day_stats = {}
        for file_path in files:
            # 提取学习日信息
            for day in self.days:
                if f"/{day}/" in str(file_path) or f"\\{day}\\" in str(file_path):
                    day_stats[day] = day_stats.get(day, 0) + 1
                    break
        
        report_lines.append("")
        report_lines.append("学习日文件分布:")
        for day in self.days:
            count = day_stats.get(day, 0)
            report_lines.append(f"  {day}: {count}个文件")
        
        # 文件列表
        report_lines.append("")
        report_lines.append("详细文件列表:")
        for i, file_path in enumerate(files, 1):
            try:
                relative_path = file_path.relative_to(self.week3_path.parent)
            except ValueError:
                relative_path = file_path.relative_to(Path.cwd())
            
            file_size = file_path.stat().st_size
            report_lines.append(f"  {i:3d}. {relative_path} ({file_size:,}字节)")
        
        # 验证结果
        if success:
            valid, file_count, total_size = self.verify_zip()
            if valid:
                report_lines.append("")
                report_lines.append("Zip文件验证结果:")
                report_lines.append(f"  Zip文件路径: {self.output_zip}")
                report_lines.append(f"  包含文件数: {file_count}")
                report_lines.append(f"  总大小: {total_size:,}字节 ({total_size/1024/1024:.2f} MB)")
            else:
                report_lines.append("")
                report_lines.append("警告: Zip文件验证失败")
        
        # 报告尾部
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("打包完成")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def run(self):
        """运行打包流程"""
        print("开始打包 Week 3 学习材料...")
        print("=" * 60)
        
        # 1. 收集文件
        print("步骤1: 收集文件...")
        files = self.collect_files()
        print(f"  找到 {len(files)} 个文件")
        
        if not files:
            print("错误: 没有找到需要打包的文件")
            return False
        
        # 2. 创建zip文件
        print("\n步骤2: 创建zip文件...")
        success = self.create_zip(files)
        
        if not success:
            print("错误: 创建zip文件失败")
            return False
        
        # 3. 验证zip文件
        print("\n步骤3: 验证zip文件...")
        valid, file_count, total_size = self.verify_zip()
        
        if valid:
            print(f"  验证成功: {file_count}个文件, {total_size:,}字节")
        else:
            print("  警告: Zip文件验证失败")
        
        # 4. 生成报告
        print("\n步骤4: 生成报告...")
        report = self.generate_report(files, success)
        
        # 保存报告
        report_path = self.week3_path / "Day21" / "打包报告.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  报告已保存到: {report_path}")
        
        # 5. 打印摘要
        print("\n" + "=" * 60)
        print("打包摘要")
        print("=" * 60)
        print(f"输出文件: {self.output_zip}")
        print(f"包含文件: {file_count}个")
        print(f"总大小: {total_size/1024/1024:.2f} MB")
        print(f"学习日: {len(self.days)}天")
        print("=" * 60)
        
        return True

def main():
    """主函数"""
    # 创建打包器
    packager = Week3Packager()
    
    # 运行打包
    success = packager.run()
    
    if success:
        print("\n打包成功！Week 3 学习材料已整理完毕。")
        print(f"请下载: {packager.output_zip}")
    else:
        print("\n打包失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()