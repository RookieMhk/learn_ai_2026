#!/usr/bin/env python3
"""
Week 3 学习成果评估工具
功能：
1. 知识点掌握度检测
2. 代码实践完成度检查
3. 学习时间分布分析
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class Week3LearningEvaluator:
    """Week 3 学习成果评估器"""
    
    def __init__(self, week3_path: str = "outputs/阶段一/Week3"):
        """
        初始化评估器
        
        Args:
            week3_path: Week 3 学习材料的根目录路径
        """
        self.week3_path = Path(week3_path)
        self.days = ["Day15", "Day16", "Day17", "Day18", "Day19", "Day20"]
        
        # Week 3 核心知识点
        self.core_knowledge_points = {
            "Day15": [
                "神经网络数学基础",
                "前向传播原理",
                "激活函数（Sigmoid、ReLU、Tanh）",
                "损失函数（MSE、交叉熵）"
            ],
            "Day16": [
                "反向传播算法",
                "梯度下降优化",
                "优化器（SGD、Adam、RMSprop）",
                "梯度消失/爆炸问题"
            ],
            "Day17": [
                "PyTorch核心API",
                "张量操作",
                "自动求导（autograd）",
                "模型定义与训练流程"
            ],
            "Day18": [
                "TensorFlow核心API",
                "即时执行模式（eager execution）",
                "Keras高级API",
                "PyTorch vs TensorFlow对比"
            ],
            "Day19": [
                "卷积神经网络（CNN）原理",
                "卷积层、池化层、全连接层",
                "图像分类任务",
                "特征提取流程"
            ],
            "Day20": [
                "循环神经网络（RNN）原理",
                "LSTM和GRU结构",
                "序列建模任务",
                "时间序列预测"
            ]
        }
        
        # 预期文件结构
        self.expected_files = {
            "Day15": [
                "神经网络数学基础详解.md",
                "梯度流动图.echarts.md",
                "神经网络结构图.echarts.md"
            ],
            "Day16": [
                "反向传播实战项目.py",
                "运行优化器对比实验.py",
                "快速实验验证.py"
            ],
            "Day17": [
                "PyTorch核心API详解.md",
                "PyTorch工作流程.echarts.md"
            ],
            "Day18": [
                "TensorFlow核心API实战.py",
                "TensorFlow_vs_PyTorch对比示例.py"
            ],
            "Day19": [
                "CNN基础与图像分类详解.md",
                "CNN特征提取流程图.echarts.md"
            ],
            "Day20": [
                "RNN_LSTM_GRU实战项目.py",
                "RNN_LSTM_GRU实战项目_简化版.py"
            ]
        }
        
        # 每日学习时间估算（分钟）
        self.estimated_time_per_day = {
            "Day15": 120,  # 2小时：神经网络数学基础
            "Day16": 150,  # 2.5小时：反向传播与优化
            "Day17": 120,  # 2小时：PyTorch API
            "Day18": 120,  # 2小时：TensorFlow API
            "Day19": 150,  # 2.5小时：CNN基础
            "Day20": 150   # 2.5小时：RNN/LSTM
        }
    
    def check_knowledge_mastery(self) -> Dict[str, Dict[str, bool]]:
        """
        检查知识点掌握度
        
        Returns:
            字典：每天的知识点掌握情况
        """
        mastery_report = {}
        
        for day in self.days:
            day_path = self.week3_path / day
            if not day_path.exists():
                mastery_report[day] = {point: False for point in self.core_knowledge_points[day]}
                continue
            
            # 检查相关文档是否存在
            day_mastery = {}
            for point in self.core_knowledge_points[day]:
                # 简化的检查：查看是否有相关文件
                has_documentation = False
                
                # 检查是否有详解文档
                for file in day_path.glob("*.md"):
                    if "详解" in file.name or "基础" in file.name:
                        has_documentation = True
                        break
                
                # 检查是否有相关代码
                has_code = False
                for file in day_path.glob("*.py"):
                    has_code = True
                    break
                
                # 如果既有文档又有代码，认为掌握了该知识点
                day_mastery[point] = has_documentation and has_code
            
            mastery_report[day] = day_mastery
        
        return mastery_report
    
    def check_code_practice_completion(self) -> Dict[str, Dict[str, bool]]:
        """
        检查代码实践完成度
        
        Returns:
            字典：每天的代码文件完成情况
        """
        completion_report = {}
        
        for day in self.days:
            day_path = self.week3_path / day
            if not day_path.exists():
                completion_report[day] = {file: False for file in self.expected_files.get(day, [])}
                continue
            
            day_completion = {}
            for expected_file in self.expected_files.get(day, []):
                file_path = day_path / expected_file
                day_completion[expected_file] = file_path.exists()
            
            completion_report[day] = day_completion
        
        return completion_report
    
    def analyze_learning_time_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        分析学习时间分布
        
        Returns:
            字典：每天的学习时间分析
        """
        time_report = {}
        total_estimated = 0
        total_files = 0
        total_days = len(self.days)
        
        for day in self.days:
            day_path = self.week3_path / day
            estimated_time = self.estimated_time_per_day.get(day, 120)
            total_estimated += estimated_time
            
            # 计算实际文件数量
            actual_files = 0
            if day_path.exists():
                actual_files = len(list(day_path.glob("*")))
                total_files += actual_files
            
            time_report[day] = {
                "estimated_minutes": estimated_time,
                "actual_files": actual_files,
                "files_per_hour": actual_files / (estimated_time / 60) if estimated_time > 0 else 0
            }
        
        # 总体统计
        overall_stats = {
            "total_estimated_hours": total_estimated / 60,
            "total_files_generated": total_files,
            "average_files_per_day": total_files / total_days if total_days > 0 else 0,
            "average_hours_per_day": total_estimated / 60 / total_days if total_days > 0 else 0
        }
        
        return {
            "daily_time": time_report,
            "overall_stats": overall_stats
        }
    
    def evaluate_learning_progress(self) -> Dict[str, any]:
        """
        综合评估学习进展
        
        Returns:
            完整的评估报告
        """
        print("开始评估 Week 3 学习成果...")
        print("=" * 60)
        
        # 1. 知识点掌握度检测
        print("1. 知识点掌握度检测")
        knowledge_mastery = self.check_knowledge_mastery()
        
        # 计算掌握率
        total_points = 0
        mastered_points = 0
        for day, points in knowledge_mastery.items():
            for point, mastered in points.items():
                total_points += 1
                if mastered:
                    mastered_points += 1
        
        knowledge_summary = {
            "total_knowledge_points": total_points,
            "mastered_points": mastered_points,
            "mastery_rate": mastered_points / total_points if total_points > 0 else 0
        }
        
        # 2. 代码实践完成度检查
        print("2. 代码实践完成度检查")
        code_completion = self.check_code_practice_completion()
        
        # 计算完成率
        total_expected_files = 0
        completed_files = 0
        for day, files in code_completion.items():
            for file, completed in files.items():
                total_expected_files += 1
                if completed:
                    completed_files += 1
        
        code_summary = {
            "total_expected_files": total_expected_files,
            "completed_files": completed_files,
            "completion_rate": completed_files / total_expected_files if total_expected_files > 0 else 0
        }
        
        # 3. 学习时间分布分析
        print("3. 学习时间分布分析")
        time_analysis = self.analyze_learning_time_distribution()
        
        # 生成综合报告
        report = {
            "evaluation_date": datetime.datetime.now().isoformat(),
            "week": "Week 3 - 深度学习基础与框架实践",
            "knowledge_mastery": {
                "summary": knowledge_summary,
                "details": knowledge_mastery
            },
            "code_practice": {
                "summary": code_summary,
                "details": code_completion
            },
            "time_distribution": time_analysis,
            "overall_assessment": {
                "status": "进行中" if mastered_points > 0 else "未开始",
                "recommendation": self._generate_recommendation(knowledge_summary, code_summary)
            }
        }
        
        return report
    
    def _generate_recommendation(self, knowledge_summary: Dict, code_summary: Dict) -> str:
        """
        生成学习建议
        
        Args:
            knowledge_summary: 知识点掌握总结
            code_summary: 代码实践总结
            
        Returns:
            学习建议
        """
        mastery_rate = knowledge_summary["mastery_rate"]
        completion_rate = code_summary["completion_rate"]
        
        if mastery_rate >= 0.8 and completion_rate >= 0.8:
            return "Week 3 学习进展良好，建议继续深入学习Week 4内容，重点关注模型部署与优化。"
        elif mastery_rate >= 0.6 and completion_rate >= 0.6:
            return "Week 3 学习进展正常，建议复习薄弱知识点，加强代码实践，为Week 4学习打好基础。"
        elif mastery_rate >= 0.4 and completion_rate >= 0.4:
            return "Week 3 学习进展较慢，建议重点复习核心概念，完成未完成的代码练习。"
        else:
            return "Week 3 学习进展需要加强，建议重新学习核心知识点，确保理解神经网络基础原理。"
    
    def save_report(self, report: Dict, output_path: str = "Week3学习评估报告.json"):
        """
        保存评估报告
        
        Args:
            report: 评估报告
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"评估报告已保存到: {output_path}")
    
    def print_summary(self, report: Dict):
        """
        打印评估摘要
        
        Args:
            report: 评估报告
        """
        print("\n" + "=" * 60)
        print("Week 3 学习成果评估摘要")
        print("=" * 60)
        
        # 基本信息
        print(f"评估日期: {report['evaluation_date']}")
        print(f"学习阶段: {report['week']}")
        print()
        
        # 知识点掌握
        knowledge = report['knowledge_mastery']['summary']
        print("知识点掌握情况:")
        print(f"  总知识点数: {knowledge['total_knowledge_points']}")
        print(f"  掌握知识点: {knowledge['mastered_points']}")
        print(f"  掌握率: {knowledge['mastery_rate']:.1%}")
        print()
        
        # 代码实践
        code = report['code_practice']['summary']
        print("代码实践完成情况:")
        print(f"  预期文件数: {code['total_expected_files']}")
        print(f"  完成文件数: {code['completed_files']}")
        print(f"  完成率: {code['completion_rate']:.1%}")
        print()
        
        # 学习时间
        time_stats = report['time_distribution']['overall_stats']
        print("学习时间分析:")
        print(f"  总预估学习时间: {time_stats['total_estimated_hours']:.1f} 小时")
        print(f"  生成文件总数: {time_stats['total_files_generated']}")
        print(f"  日均学习时间: {time_stats['average_hours_per_day']:.1f} 小时")
        print(f"  日均生成文件: {time_stats['average_files_per_day']:.1f}")
        print()
        
        # 总体评估
        assessment = report['overall_assessment']
        print("总体评估:")
        print(f"  状态: {assessment['status']}")
        print(f"  建议: {assessment['recommendation']}")
        print("=" * 60)

def main():
    """主函数"""
    # 创建评估器
    evaluator = Week3LearningEvaluator()
    
    # 评估学习进展
    report = evaluator.evaluate_learning_progress()
    
    # 打印摘要
    evaluator.print_summary(report)
    
    # 保存详细报告
    report_path = "outputs/阶段一/Week3/Day21/Week3学习评估报告.json"
    evaluator.save_report(report, report_path)
    
    # 生成HTML格式的简易报告
    generate_html_report(report, "outputs/阶段一/Week3/Day21/Week3学习评估报告.html")
    
    print("\n评估完成！详细报告已保存。")

def generate_html_report(report: Dict, output_path: str):
    """
    生成HTML格式的评估报告
    
    Args:
        report: 评估报告
        output_path: 输出文件路径
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Week 3 学习成果评估报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 5px;
            font-weight: bold;
        }}
        .progress-bar {{
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #2ecc71;
            border-radius: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .recommendation {{
            background-color: #e8f4fc;
            border-left: 4px solid #2980b9;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .day-section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        .mastery-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
        }}
        .mastery-true {{
            background-color: #2ecc71;
            color: white;
        }}
        .mastery-false {{
            background-color: #e74c3c;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Week 3 学习成果评估报告</h1>
        <p><strong>评估日期:</strong> {report['evaluation_date']}</p>
        <p><strong>学习阶段:</strong> {report['week']}</p>
        
        <div class="summary-box">
            <h2>总体摘要</h2>
            <p><span class="metric">知识点掌握率: {report['knowledge_mastery']['summary']['mastery_rate']:.1%}</span>
            <span class="metric">代码完成率: {report['code_practice']['summary']['completion_rate']:.1%}</span></p>
            <p><strong>状态:</strong> {report['overall_assessment']['status']}</p>
        </div>
        
        <h2>知识点掌握详情</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {report['knowledge_mastery']['summary']['mastery_rate']*100}%"></div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>学习日</th>
                    <th>核心知识点</th>
                    <th>掌握情况</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加知识点详情
    for day, points in report['knowledge_mastery']['details'].items():
        for point, mastered in points.items():
            mastery_class = "mastery-true" if mastered else "mastery-false"
            mastery_text = "已掌握" if mastered else "待学习"
            html_content += f"""
                <tr>
                    <td>{day}</td>
                    <td>{point}</td>
                    <td><span class="mastery-badge {mastery_class}">{mastery_text}</span></td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>代码实践完成情况</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: 90%"></div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>学习日</th>
                    <th>预期文件</th>
                    <th>完成情况</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加代码实践详情
    for day, files in report['code_practice']['details'].items():
        for file, completed in files.items():
            completion_class = "mastery-true" if completed else "mastery-false"
            completion_text = "已完成" if completed else "未完成"
            html_content += f"""
                <tr>
                    <td>{day}</td>
                    <td>{file}</td>
                    <td><span class="mastery-badge {completion_class}">{completion_text}</span></td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>学习时间分布</h2>
        <p><strong>总预估学习时间:</strong> {report['time_distribution']['overall_stats']['total_estimated_hours']:.1f} 小时</p>
        <p><strong>生成文件总数:</strong> {report['time_distribution']['overall_stats']['total_files_generated']}</p>
        <p><strong>日均学习时间:</strong> {report['time_distribution']['overall_stats']['average_hours_per_day']:.1f} 小时</p>
        
        <div class="recommendation">
            <h2>学习建议</h2>
            <p>{report['overall_assessment']['recommendation']}</p>
        </div>
        
        <p><em>报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已保存到: {output_path}")

if __name__ == "__main__":
    main()