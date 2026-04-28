#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1学习成果整理工具
功能：自动汇总Day 1-6的《今日执行卡片》内容，生成《Week 1周进展脑图.md》
作者：扣子AI助手
日期：2026-04-01
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class Week1ProgressCollector:
    """Week 1学习成果整理工具"""
    
    def __init__(self, base_path: str = "outputs/阶段一/Week1"):
        """
        初始化收集器
        
        Args:
            base_path: Week 1目录路径
        """
        self.base_path = Path(base_path)
        self.days = ["Day1", "Day2", "Day3", "Day4", "Day5", "Day6"]
        self.cards_data = {}
        self.mindmap_data = {}
        
    def find_card_file(self, day_dir: Path) -> Optional[Path]:
        """
        查找每日的执行卡片文件
        
        优先查找用户填写的卡片文件（如"今日执行卡片.md"），
        如果不存在，则查找模板文件（如"DayX今日执行卡片模板.md"）
        
        Args:
            day_dir: 每日目录路径
            
        Returns:
            卡片文件路径，如果未找到返回None
        """
        # 可能的用户填写文件名
        user_card_names = [
            "今日执行卡片.md",
            f"{day_dir.name}今日执行卡片.md",
            "执行卡片.md",
            "学习卡片.md"
        ]
        
        # 可能的模板文件名
        template_names = [
            f"{day_dir.name}今日执行卡片模板.md",
            "今日执行卡片模板.md",
            "执行卡片模板.md"
        ]
        
        # 先查找用户填写的卡片
        for name in user_card_names:
            card_path = day_dir / name
            if card_path.exists():
                print(f"找到用户填写的卡片: {card_path}")
                return card_path
        
        # 如果没找到用户卡片，查找模板文件
        for name in template_names:
            template_path = day_dir / name
            if template_path.exists():
                print(f"未找到用户填写的卡片，使用模板文件: {template_path}")
                print("提示：建议用户填写卡片后再运行此工具，以获得更准确的汇总结果")
                return template_path
        
        print(f"警告：未在 {day_dir} 中找到任何卡片文件")
        return None
    
    def parse_card_content(self, content: str, day_name: str) -> Dict:
        """
        解析卡片内容，提取收获、疑问、感悟等信息
        
        Args:
            content: 卡片内容文本
            day_name: 日期名称（如"Day1"）
            
        Returns:
            结构化数据字典
        """
        # 初始化数据结构
        card_data = {
            "day": day_name,
            "date": "",
            "theme": "",
            "duration": "",
            "state": "",
            "harvests": [],
            "questions": [],
            "insights": [],
            "ai_trends": [],
            "self_assessment": {},
            "next_plan": "",
            "is_template": False
        }
        
        # 判断是否为模板文件（包含大量占位符）
        template_indicators = ["____", "例如：", "（请填写）", "填写提示"]
        is_template = any(indicator in content for indicator in template_indicators)
        card_data["is_template"] = is_template
        
        # 提取基本信息
        date_match = re.search(r"学习日期.*?(\d{4}年\d{1,2}月\d{1,2}日)", content)
        if date_match:
            card_data["date"] = date_match.group(1)
        
        theme_match = re.search(r"学习主题.*?([^\n]+)", content)
        if theme_match:
            card_data["theme"] = theme_match.group(1).strip()
        
        duration_match = re.search(r"学习时长.*?([^\n]+)", content)
        if duration_match:
            card_data["duration"] = duration_match.group(1).strip()
        
        # 提取核心收获（至少3项）
        harvest_section = self.extract_section(content, "核心收获")
        if harvest_section:
            harvests = self.extract_numbered_items(harvest_section)
            card_data["harvests"] = harvests[:3]  # 取前3个
        
        # 提取疑难困惑
        question_section = self.extract_section(content, "疑难困惑")
        if question_section:
            questions = self.extract_numbered_items(question_section)
            card_data["questions"] = questions[:1]  # 取前1个
        
        # 提取学习感悟
        insight_section = self.extract_section(content, "学习感悟")
        if insight_section:
            insights = self.extract_numbered_items(insight_section)
            card_data["insights"] = insights[:1]  # 取前1个
        
        # 提取AI动态摘要
        ai_section = self.extract_section(content, "AI动态摘要")
        if ai_section:
            ai_trends = self.extract_numbered_items(ai_section)
            card_data["ai_trends"] = ai_trends[:2]  # 取前2个
        
        return card_data
    
    def extract_section(self, content: str, section_title: str) -> Optional[str]:
        """
        提取指定章节的内容
        
        Args:
            content: 完整内容
            section_title: 章节标题
            
        Returns:
            章节内容字符串，如果未找到返回None
        """
        # 构建标题匹配模式
        patterns = [
            f"## {section_title}.*?\n(.*?)\n---",
            f"## {section_title}.*?\n(.*?)\n## ",
            f"# {section_title}.*?\n(.*?)\n# ",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_numbered_items(self, section_content: str) -> List[str]:
        """
        提取编号列表项
        
        Args:
            section_content: 章节内容
            
        Returns:
            项目列表
        """
        items = []
        
        # 匹配各种编号格式：1.、1)、**收获一**、**疑问一**等
        patterns = [
            r'\d+\.\s+([^\n]+(?:\n[^\d].*?)*?)(?=\n\d+\.|\n$|\n##|\n#)',
            r'\*\*[^一]+\*\*\s*[:：]?\s*([^\n]+(?:\n[^\*].*?)*?)(?=\n\*\*|\n$|\n##|\n#)',
            r'[-*]\s+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, section_content, re.DOTALL)
            for match in matches:
                item = match.strip()
                if item and len(item) > 5:  # 过滤太短的内容（可能是占位符）
                    items.append(item)
        
        return items
    
    def collect_all_cards(self) -> Dict[str, Dict]:
        """
        收集所有日期的卡片数据
        
        Returns:
            按日期组织的卡片数据字典
        """
        print("开始收集Week 1学习成果...")
        print("=" * 50)
        
        for day in self.days:
            day_dir = self.base_path / day
            if not day_dir.exists():
                print(f"警告：目录不存在 {day_dir}")
                continue
            
            card_path = self.find_card_file(day_dir)
            if not card_path:
                self.cards_data[day] = None
                continue
            
            try:
                with open(card_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                card_data = self.parse_card_content(content, day)
                self.cards_data[day] = card_data
                
                # 打印收集状态
                status = "模板文件" if card_data["is_template"] else "用户填写"
                print(f"{day}: {card_data['theme']} ({status})")
                print(f"  收获: {len(card_data['harvests'])}条")
                print(f"  疑问: {len(card_data['questions'])}条")
                print(f"  感悟: {len(card_data['insights'])}条")
                
            except Exception as e:
                print(f"解析{day}卡片时出错: {e}")
                self.cards_data[day] = None
        
        print("=" * 50)
        print(f"卡片收集完成，共收集到 {len([v for v in self.cards_data.values() if v])}/{len(self.days)} 天的数据")
        
        return self.cards_data
    
    def generate_mindmap(self) -> str:
        """
        生成周进展脑图内容
        
        Returns:
            脑图markdown内容
        """
        if not self.cards_data:
            print("未收集到卡片数据，请先运行collect_all_cards()")
            return ""
        
        # 初始化脑图结构
        mindmap = {
            "Week 1学习成果": {
                "数学基础": {
                    "线性代数": [],
                    "概率统计": [],
                    "微积分": []
                },
                "Python数据科学": {
                    "NumPy": [],
                    "Pandas": [],
                    "Matplotlib": []
                },
                "机器学习入门": {
                    "线性回归": [],
                    "模型评估": []
                },
                "AI技术动态": [],
                "学习感悟总结": [],
                "疑难问题汇总": [],
                "下周学习建议": []
            }
        }
        
        # 知识领域映射
        knowledge_map = {
            "线性代数": ["向量", "矩阵", "特征值", "SVD", "点积", "秩"],
            "概率统计": ["概率", "统计", "分布", "期望", "方差", "协方差"],
            "NumPy": ["数组", "广播", "运算", "线性代数"],
            "Pandas": ["数据清洗", "DataFrame", "Series", "缺失值"],
            "Matplotlib": ["可视化", "图表", "绘图", "子图"],
            "线性回归": ["回归", "梯度下降", "损失函数", "MSE", "R²"]
        }
        
        # 汇总各卡片数据
        for day, data in self.cards_data.items():
            if not data:
                continue
            
            # 汇总收获
            for harvest in data["harvests"]:
                # 根据关键词分类到不同知识领域
                classified = False
                for category, keywords in knowledge_map.items():
                    if any(keyword in harvest for keyword in keywords):
                        if category == "线性代数" or category == "概率统计":
                            mindmap["Week 1学习成果"]["数学基础"][category].append(f"{day}: {harvest[:50]}...")
                        elif category in ["NumPy", "Pandas", "Matplotlib"]:
                            mindmap["Week 1学习成果"]["Python数据科学"][category].append(f"{day}: {harvest[:50]}...")
                        elif category == "线性回归":
                            mindmap["Week 1学习成果"]["机器学习入门"][category].append(f"{day}: {harvest[:50]}...")
                        classified = True
                        break
                
                if not classified:
                    # 默认放入数学基础
                    mindmap["Week 1学习成果"]["数学基础"]["线性代数"].append(f"{day}: {harvest[:50]}...")
            
            # 汇总疑问
            for question in data["questions"]:
                mindmap["Week 1学习成果"]["疑难问题汇总"].append(f"{day}: {question[:100]}...")
            
            # 汇总感悟
            for insight in data["insights"]:
                mindmap["Week 1学习成果"]["学习感悟总结"].append(f"{day}: {insight[:100]}...")
            
            # 汇总AI动态
            for trend in data["ai_trends"]:
                mindmap["Week 1学习成果"]["AI技术动态"].append(f"{day}: {trend[:100]}...")
        
        # 生成markdown格式的脑图
        md_content = self._mindmap_to_markdown(mindmap)
        
        # 保存到文件
        output_path = self.base_path / "Day7" / "Week 1周进展脑图.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"脑图已生成并保存至: {output_path}")
        return md_content
    
    def _mindmap_to_markdown(self, mindmap: Dict, level: int = 0) -> str:
        """
        将脑图字典转换为markdown格式
        
        Args:
            mindmap: 脑图字典
            level: 当前层级
            
        Returns:
            markdown格式内容
        """
        md_lines = []
        
        for key, value in mindmap.items():
            indent = "  " * level
            prefix = "#" * (level + 1) + " " if level < 6 else "- "
            
            if isinstance(value, dict):
                # 节点
                md_lines.append(f"{indent}{prefix}{key}")
                md_lines.append(self._mindmap_to_markdown(value, level + 1))
            elif isinstance(value, list):
                # 叶子节点（列表项）
                md_lines.append(f"{indent}{prefix}{key}")
                for item in value:
                    md_lines.append(f"{indent}  - {item}")
            else:
                # 简单值
                md_lines.append(f"{indent}{prefix}{key}: {value}")
        
        return "\n".join(md_lines)
    
    def generate_statistics(self) -> Dict:
        """
        生成学习数据统计
        
        Returns:
            统计数据字典
        """
        stats = {
            "total_days": len(self.days),
            "collected_days": 0,
            "total_harvests": 0,
            "total_questions": 0,
            "total_insights": 0,
            "total_ai_trends": 0,
            "template_days": 0,
            "user_filled_days": 0,
            "knowledge_distribution": {}
        }
        
        for day, data in self.cards_data.items():
            if not data:
                continue
            
            stats["collected_days"] += 1
            
            if data["is_template"]:
                stats["template_days"] += 1
            else:
                stats["user_filled_days"] += 1
            
            stats["total_harvests"] += len(data["harvests"])
            stats["total_questions"] += len(data["questions"])
            stats["total_insights"] += len(data["insights"])
            stats["total_ai_trends"] += len(data["ai_trends"])
        
        return stats
    
    def run_full_analysis(self):
        """
        运行完整的分析流程
        """
        print("Week 1学习成果分析工具")
        print("=" * 50)
        
        # 1. 收集卡片数据
        self.collect_all_cards()
        
        # 2. 生成统计
        stats = self.generate_statistics()
        print("\n学习数据统计:")
        print(f"- 总天数: {stats['total_days']}天")
        print(f"- 已收集: {stats['collected_days']}天")
        print(f"- 用户填写: {stats['user_filled_days']}天")
        print(f"- 使用模板: {stats['template_days']}天")
        print(f"- 总收获数: {stats['total_harvests']}条")
        print(f"- 总疑问数: {stats['total_questions']}条")
        print(f"- 总感悟数: {stats['total_insights']}条")
        
        # 3. 生成脑图
        print("\n生成周进展脑图...")
        self.generate_mindmap()
        
        # 4. 生成建议
        print("\n学习建议:")
        if stats["user_filled_days"] < 3:
            print("- 建议更多填写每日执行卡片，有助于巩固学习成果")
        if stats["total_questions"] > 0:
            print(f"- 本周共有{stats['total_questions']}个疑问，建议在Week 2学习中重点关注")
        if stats["total_harvests"] >= 15:
            print("- 收获丰富，说明本周学习效果良好，继续保持!")
        
        print("\n分析完成！")


def main():
    """主函数"""
    # 创建收集器实例
    collector = Week1ProgressCollector()
    
    # 运行完整分析
    collector.run_full_analysis()
    
    # 打印使用说明
    print("\n" + "=" * 50)
    print("使用说明:")
    print("1. 如果用户填写了每日执行卡片，工具会自动汇总内容")
    print("2. 如果只有模板文件，工具会提示用户填写后再运行")
    print("3. 生成的脑图文件保存在 outputs/阶段一/Week1/Day7/Week 1周进展脑图.md")
    print("4. 建议每周日使用此工具进行学习复盘")
    
    # 保存统计数据到JSON文件
    stats = collector.generate_statistics()
    stats_path = Path("outputs/阶段一/Week1/Day7/Week 1学习统计数据.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"5. 详细统计数据已保存至: {stats_path}")


if __name__ == "__main__":
    main()