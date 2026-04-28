#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 2 学习成果评估工具 v1.0
=============================
生成时间：2026-04-18
功能：评估机器学习核心概念掌握度、代码实践完成度、学习时间分布

使用方法：
    python Week2学习成果评估工具.py
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# ============================================================================
# 配置区：可根据实际情况修改
# ============================================================================

# Week 2 学习模块配置
WEEK2_MODULES = {
    "Day8": {
        "主题": "监督学习基础 + 注意力机制数学原理",
        "核心知识点": [
            "监督学习分类体系",
            "特征与标签概念",
            "训练/验证/测试集划分",
            "注意力机制(Self-Attention)",
            "Q/K/V矩阵计算",
            "缩放点积注意力"
        ],
        "代码任务": ["实现简单分类器", "注意力机制演示"],
        "建议时长": 120  # 分钟
    },
    "Day9": {
        "主题": "损失函数与优化算法进阶",
        "核心知识点": [
            "均方误差(MSE)损失",
            "交叉熵损失(Cross-Entropy)",
            "梯度下降算法",
            "随机梯度下降(SGD)",
            "Adam优化器原理",
            "学习率调度策略"
        ],
        "代码任务": ["优化器对比实验", "学习率影响分析"],
        "建议时长": 150
    },
    "Day10": {
        "主题": "模型评估指标 + 算法基础",
        "核心知识点": [
            "准确率、精确率、召回率",
            "F1-Score计算",
            "ROC-AUC曲线",
            "混淆矩阵解读",
            "偏差-方差权衡",
            "过拟合与欠拟合"
        ],
        "代码任务": ["评估指标可视化", "模型诊断实践"],
        "建议时长": 140
    },
    "Day11": {
        "主题": "特征工程 + 智能工具实践",
        "核心知识点": [
            "特征缩放方法",
            "特征选择策略",
            "编码类别特征",
            "文本向量化",
            "提示词工程基础",
            "Coze工作流搭建"
        ],
        "代码任务": ["特征工程pipeline", "AI工具集成"],
        "建议时长": 130
    },
    "Day12": {
        "主题": "综合项目实战",
        "核心知识点": [
            "项目流程规范",
            "数据预处理",
            "模型选择与训练",
            "结果分析与报告"
        ],
        "代码任务": ["端到端项目完成"],
        "建议时长": 180
    },
    "Day13": {
        "主题": "知识整理与预习",
        "核心知识点": [
            "知识体系梳理",
            "Week2知识框架整合",
            "Week3预习内容"
        ],
        "代码任务": ["知识整理输出"],
        "建议时长": 90
    }
}

# 知识点掌握度评估题目
KNOWLEDGE_QUESTIONS = [
    {
        "id": 1,
        "题目": "监督学习中，训练集、验证集、测试集的划分比例通常为？",
        "选项": ["60:20:20", "70:15:15", "80:10:10", "90:5:5"],
        "答案": "C",
        "解析": "通常采用80:10:10或70:15:15的比例，训练集需要足够大，测试集需要足够有代表性。",
        "对应模块": "Day8",
        "难度": "基础"
    },
    {
        "id": 2,
        "题目": "注意力机制中，Q、K、V矩阵分别代表什么？",
        "选项": ["Query, Key, Value", "Question, Knowledge, Value", "Query, Kernel, Vector", "Quality, Key, Variable"],
        "答案": "A",
        "解析": "Q(Query)查询向量、K(Key)键向量、V(Value)值向量，通过Q与K的相似度计算注意力权重。",
        "对应模块": "Day8",
        "难度": "基础"
    },
    {
        "id": 3,
        "题目": "交叉熵损失函数适用于以下哪种问题？",
        "选项": ["回归问题", "分类问题", "聚类问题", "降维问题"],
        "答案": "B",
        "解析": "交叉熵主要用于分类问题，特别是多分类问题，与softmax函数配合使用效果最佳。",
        "对应模块": "Day9",
        "难度": "基础"
    },
    {
        "id": 4,
        "题目": "Adam优化器结合了哪两个优化算法的优点？",
        "选项": ["SGD + AdaGrad", "AdaGrad + RMSProp", "Momentum + RMSProp", "SGD + Momentum"],
        "答案": "C",
        "解析": "Adam = Adaptive + Momentum，结合了Momentum的一阶矩估计和RMSProp的二阶矩估计。",
        "对应模块": "Day9",
        "难度": "中等"
    },
    {
        "id": 5,
        "题目": "在模型评估中，F1-Score是什么的调和平均数？",
        "选项": ["准确率和召回率", "精确率和召回率", "精确率和准确率", "召回率和AUC"],
        "答案": "B",
        "解析": "F1 = 2 × (Precision × Recall) / (Precision + Recall)，平衡精确率和召回率的贡献。",
        "对应模块": "Day10",
        "难度": "基础"
    },
    {
        "id": 6,
        "题目": "ROC-AUC值为0.5意味着什么？",
        "选项": ["完美分类", "随机猜测", "完全错误", "过拟合"],
        "答案": "B",
        "解析": "AUC=0.5表示模型没有区分能力，等同于随机猜测；AUC=1.0为完美分类。",
        "对应模块": "Day10",
        "难度": "基础"
    },
    {
        "id": 7,
        "题目": "偏差-方差分解中，高偏差会导致什么问题？",
        "选项": ["过拟合", "欠拟合", "维度灾难", "数据泄露"],
        "答案": "B",
        "解析": "高偏差意味着模型太简单，无法捕捉数据规律，导致欠拟合(欠拟合)。",
        "对应模块": "Day10",
        "难度": "中等"
    },
    {
        "id": 8,
        "题目": "特征缩放的Z-score标准化公式是什么？",
        "选项": ["(x - min) / (max - min)", "(x - μ) / σ", "x / √(x²)", "log(x)"],
        "答案": "B",
        "解析": "Z-score = (x - mean) / std，使数据服从均值为0、标准差为1的正态分布。",
        "对应模块": "Day11",
        "难度": "中等"
    },
    {
        "id": 9,
        "题目": "One-Hot编码的主要目的是什么？",
        "选项": ["减少特征数量", "将类别特征转为数值", "特征选择", "降维"],
        "答案": "B",
        "解析": "One-Hot将类别特征转换为二进制向量，使模型能正确处理非数值型数据。",
        "对应模块": "Day11",
        "难度": "基础"
    },
    {
        "id": 10,
        "题目": "过拟合的典型解决策略不包括？",
        "选项": ["增加数据量", "正则化", "减少模型复杂度", "减少验证集"],
        "答案": "D",
        "解析": "过拟合时应增加数据、正则化、简化模型；减少验证集会降低评估可靠性。",
        "对应模块": "Day10",
        "难度": "中等"
    }
]


# ============================================================================
# 评估工具核心类
# ============================================================================

class LearningAssessmentTool:
    """Week 2 学习成果评估工具"""
    
    def __init__(self):
        self.modules = WEEK2_MODULES
        self.questions = KNOWLEDGE_QUESTIONS
        self.assessment_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    # ----------------------------------------------------------------------
    # 功能1：知识点掌握度检测
    # ----------------------------------------------------------------------
    def assess_knowledge_mastery(self, answers: Dict[int, str]) -> Dict[str, Any]:
        """
        检测知识点掌握度
        
        参数:
            answers: 字典，key为题目ID，value为用户选择的选项字母
            
        返回:
            评估结果字典
        """
        print("\n" + "="*60)
        print("📊 功能1：知识点掌握度检测")
        print("="*60)
        
        results = {
            "评估时间": self.assessment_date,
            "总题数": len(self.questions),
            "作答数": len(answers),
            "正确数": 0,
            "错误数": 0,
            "正确率": 0.0,
            "模块掌握度": defaultdict(dict),
            "详细结果": []
        }
        
        module_correct = defaultdict(int)
        module_total = defaultdict(int)
        
        for q in self.questions:
            qid = q["id"]
            module = q["对应模块"]
            module_total[module] += 1
            
            if qid in answers:
                user_answer = answers[qid].upper()
                is_correct = user_answer == q["答案"]
                
                if is_correct:
                    results["正确数"] += 1
                    module_correct[module] += 1
                
                results["详细结果"].append({
                    "题目ID": qid,
                    "题目": q["题目"],
                    "用户答案": user_answer,
                    "正确答案": q["答案"],
                    "是否正确": is_correct,
                    "解析": q["解析"],
                    "难度": q["难度"]
                })
            else:
                results["详细结果"].append({
                    "题目ID": qid,
                    "题目": q["题目"],
                    "用户答案": "未作答",
                    "正确答案": q["答案"],
                    "是否正确": False,
                    "解析": q["解析"],
                    "难度": q["难度"]
                })
                results["错误数"] += 1
        
        # 计算正确率
        if len(answers) > 0:
            results["正确率"] = results["正确数"] / len(answers) * 100
        
        # 计算模块掌握度
        for module in module_total:
            correct = module_correct[module]
            total = module_total[module]
            rate = correct / total * 100 if total > 0 else 0
            
            level = "优秀" if rate >= 90 else ("良好" if rate >= 70 else ("一般" if rate >= 50 else "待提升"))
            
            results["模块掌握度"][module] = {
                "正确数": correct,
                "总题数": total,
                "掌握率": rate,
                "掌握等级": level
            }
        
        return results
    
    def print_knowledge_assessment(self, results: Dict):
        """打印知识点评估结果"""
        print(f"\n📅 评估时间: {results['评估时间']}")
        print(f"📝 总题数: {results['总题数']} | 已作答: {results['作答数']}")
        print(f"✅ 正确: {results['正确数']} | ❌ 错误: {results['错误数']}")
        print(f"📈 正确率: {results['正确率']:.1f}%")
        
        print("\n" + "-"*40)
        print("📊 模块掌握度详情:")
        print("-"*40)
        
        for module, stats in results["模块掌握度"].items():
            bar_length = int(stats["掌握率"] / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            status_emoji = "🟢" if stats["掌握等级"] == "优秀" else \
                           "🔵" if stats["掌握等级"] == "良好" else \
                           "🟡" if stats["掌握等级"] == "一般" else "🔴"
            
            print(f"  {module}: {bar} {stats['掌握率']:.0f}% {status_emoji}{stats['掌握等级']}")
    
    # ----------------------------------------------------------------------
    # 功能2：代码实践完成度检查
    # ----------------------------------------------------------------------
    def check_code_completion(self, completed_tasks: List[str]) -> Dict[str, Any]:
        """
        检查代码实践完成度
        
        参数:
            completed_tasks: 已完成的代码任务列表
            
        返回:
            完成度评估结果
        """
        print("\n" + "="*60)
        print("💻 功能2：代码实践完成度检查")
        print("="*60)
        
        all_tasks = []
        for day, info in self.modules.items():
            if "代码任务" in info:
                for task in info["代码任务"]:
                    all_tasks.append({
                        "day": day,
                        "task": task,
                        "status": "completed" if task in completed_tasks else "pending"
                    })
        
        total = len(all_tasks)
        completed = len([t for t in all_tasks if t["status"] == "completed"])
        
        results = {
            "评估时间": self.assessment_date,
            "总任务数": total,
            "已完成数": completed,
            "完成率": completed / total * 100 if total > 0 else 0,
            "详细清单": all_tasks,
            "未完成任务": [t for t in all_tasks if t["status"] == "pending"]
        }
        
        return results
    
    def print_code_completion(self, results: Dict):
        """打印代码完成度结果"""
        print(f"\n📅 评估时间: {results['评估时间']}")
        print(f"💻 总任务数: {results['总任务数']} | 已完成: {results['已完成数']}")
        print(f"📊 完成率: {results['完成率']:.1f}%")
        
        if results["未完成任务"]:
            print("\n❌ 未完成任务:")
            for task in results["未完成任务"]:
                print(f"   • {task['day']} - {task['task']}")
        
        # 按日期分组显示
        print("\n" + "-"*40)
        print("📋 任务完成清单:")
        print("-"*40)
        
        days = set(t["day"] for t in results["详细清单"]])
        for day in sorted(days):
            day_tasks = [t for t in results["详细清单"] if t["day"] == day]
            completed = len([t for t in day_tasks if t["status"] == "completed"])
            total = len(day_tasks)
            status = "✅" if completed == total else "⏳"
            print(f"\n  {day} {status}")
            for task in day_tasks:
                check = "✓" if task["status"] == "completed" else "○"
                print(f"     [{check}] {task['task']}")
    
    # ----------------------------------------------------------------------
    # 功能3：学习时间分布分析
    # ----------------------------------------------------------------------
    def analyze_time_distribution(self, study_records: List[Dict]) -> Dict[str, Any]:
        """
        分析学习时间分布
        
        参数:
            study_records: 学习记录列表，每条记录包含:
                - day: 日期/天数标识
                - actual_minutes: 实际学习时长(分钟)
                - topic: 学习主题
                
        返回:
            时间分布分析结果
        """
        print("\n" + "="*60)
        print("⏰ 功能3：学习时间分布分析")
        print("="*60)
        
        if not study_records:
            print("\n⚠️ 暂无学习记录数据，请先记录每日学习时长。")
            return {"error": "缺少学习记录数据"}
        
        # 统计数据
        total_actual = sum(r["actual_minutes"] for r in study_records)
        total_suggested = sum(self.modules.get(r["day"], {}).get("建议时长", 0) 
                             for r in study_records)
        
        day_stats = []
        for record in study_records:
            day = record["day"]
            actual = record["actual_minutes"]
            suggested = self.modules.get(day, {}).get("建议时长", 0)
            diff = actual - suggested
            ratio = actual / suggested * 100 if suggested > 0 else 0
            
            day_stats.append({
                "day": day,
                "topic": record.get("topic", ""),
                "actual": actual,
                "suggested": suggested,
                "difference": diff,
                "completion_ratio": ratio
            })
        
        # 按实际时长排序
        day_stats.sort(key=lambda x: x["actual"], reverse=True)
        
        results = {
            "评估时间": self.assessment_date,
            "记录天数": len(study_records),
            "总实际时长": total_actual,
            "总建议时长": total_suggested,
            "时间达标率": total_actual / total_suggested * 100 if total_suggested > 0 else 0,
            "日均时长": total_actual / len(study_records) if study_records else 0,
            "每日详情": day_stats,
            "建议": self._generate_time_suggestions(day_stats)
        }
        
        return results
    
    def _generate_time_suggestions(self, day_stats: List[Dict]) -> List[str]:
        """生成时间优化建议"""
        suggestions = []
        
        # 检查超标或不足
        underperforming = [d for d in day_stats if d["completion_ratio"] < 80]
        overperforming = [d for d in day_stats if d["completion_ratio"] > 120]
        
        if underperforming:
            suggestions.append(f"⚠️ 以下模块学习时间偏少，建议加强：{', '.join([d['day'] for d in underperforming])}")
        
        if overperforming:
            suggestions.append(f"💡 以下模块耗时较多，可考虑优化效率：{', '.join([d['day'] for d in overperforming])}")
        
        avg_ratio = sum(d["completion_ratio"] for d in day_stats) / len(day_stats) if day_stats else 0
        if avg_ratio < 80:
            suggestions.append("📌 本周整体学习时长偏少，建议调整日程或提高每日学习效率")
        elif avg_ratio > 110:
            suggestions.append("📌 本周学习投入充足，注意休息避免疲劳")
        else:
            suggestions.append("✅ 学习时间分配合理，保持当前节奏")
        
        return suggestions
    
    def print_time_analysis(self, results: Dict):
        """打印时间分布分析结果"""
        if "error" in results:
            return
            
        print(f"\n📅 评估时间: {results['评估时间']}")
        print(f"⏱️  记录天数: {results['记录天数']}")
        print(f"📊 总学习时长: {results['总实际时长']}分钟 ({results['总实际时长']/60:.1f}小时)")
        print(f"📈 时间达标率: {results['时间达标率']:.1f}%")
        print(f"📉 日均学习时长: {results['日均时长']:.0f}分钟 ({results['日均时长']/60:.1f}小时)")
        
        print("\n" + "-"*50)
        print("📅 每日学习时间详情:")
        print("-"*50)
        print(f"{'日期':<8} {'主题':<20} {'实际':>8} {'建议':>8} {'达标率':>8}")
        print("-"*50)
        
        for day in results["每日详情"]:
            status = "✓" if day["completion_ratio"] >= 100 else "✗"
            print(f"{day['day']:<8} {day['topic'][:18]:<18} {day['actual']:>6}min {day['suggested']:>6}min {day['completion_ratio']:>6.0f}%{status}")
        
        print("\n💡 优化建议:")
        for suggestion in results["建议"]:
            print(f"   {suggestion}")
    
    # ----------------------------------------------------------------------
    # 综合评估报告
    # ----------------------------------------------------------------------
    def generate_comprehensive_report(self, 
                                     knowledge_results: Dict,
                                     code_results: Dict,
                                     time_results: Dict) -> Dict[str, Any]:
        """生成综合评估报告"""
        
        # 计算综合得分
        knowledge_score = knowledge_results.get("正确率", 0) * 0.4
        code_score = code_results.get("完成率", 0) * 0.35
        time_score = min(time_results.get("时间达标率", 100), 100) * 0.25
        
        total_score = knowledge_score + code_score + time_score
        
        # 评级
        if total_score >= 90:
            grade = "A+"
            grade_desc = "卓越"
        elif total_score >= 85:
            grade = "A"
            grade_desc = "优秀"
        elif total_score >= 75:
            grade = "B+"
            grade_desc = "良好"
        elif total_score >= 65:
            grade = "B"
            grade_desc = "合格"
        elif total_score >= 55:
            grade = "C"
            grade_desc = "需改进"
        else:
            grade = "D"
            grade_desc = "不达标"
        
        report = {
            "评估日期": self.assessment_date,
            "Week2整体评估": {
                "综合得分": total_score,
                "评级": grade,
                "评级描述": grade_desc
            },
            "分项得分": {
                "知识点掌握度": {
                    "得分": knowledge_score,
                    "权重": "40%",
                    "正确率": knowledge_results.get("正确率", 0)
                },
                "代码实践完成度": {
                    "得分": code_score,
                    "权重": "35%",
                    "完成率": code_results.get("完成率", 0)
                },
                "学习时间投入": {
                    "得分": time_score,
                    "权重": "25%",
                    "达标率": time_results.get("时间达标率", 0) if "error" not in time_results else 0
                }
            },
            "Week3建议": self._generate_week3_suggestions(knowledge_results, code_results, time_results)
        }
        
        return report
    
    def _generate_week3_suggestions(self, 
                                    knowledge: Dict, 
                                    code: Dict, 
                                    time: Dict) -> List[str]:
        """生成Week3学习建议"""
        suggestions = []
        
        # 知识点薄弱点
        weak_modules = [m for m, s in knowledge.get("模块掌握度", {}).items() 
                       if s["掌握率"] < 70]
        if weak_modules:
            suggestions.append(f"📚 重点复习Week2薄弱模块：{', '.join(weak_modules)}")
        
        # 代码未完成
        if code.get("未完成任务"):
            suggestions.append(f"💻 补做未完成的代码任务后再进入Week3")
        
        # 时间问题
        if "error" not in time:
            if time.get("时间达标率", 100) < 80:
                suggestions.append("⏰ 建议提高每日学习时间投入至每周12小时以上")
        
        # Week3预告
        suggestions.append("🚀 Week3将进入深度学习基础阶段，重点：神经网络、反向传播、PyTorch")
        
        return suggestions
    
    def print_comprehensive_report(self, report: Dict):
        """打印综合评估报告"""
        print("\n" + "="*60)
        print("🏆 Week 2 综合评估报告")
        print("="*60)
        
        print(f"\n📅 评估日期: {report['评估日期']}")
        print(f"\n{'='*40}")
        print(f"   综合得分: {report['Week2整体评估']['综合得分']:.1f}/100")
        print(f"   评级: {report['Week2整体评估']['评级']} ({report['Week2整体评估']['评级描述']})")
        print(f"{'='*40}")
        
        print("\n📊 分项得分:")
        for item, data in report["分项得分"].items():
            bar_len = int(data["得分"] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {item}:")
            print(f"      {bar} {data['得分']:.1f}分 (权重{data['权重']})")
        
        print("\n💡 Week3学习建议:")
        for i, suggestion in enumerate(report["Week3建议"], 1):
            print(f"   {i}. {suggestion}")
    
    # ----------------------------------------------------------------------
    # 导出评估结果
    # ----------------------------------------------------------------------
    def export_report(self, report: Dict, filename: str = "Week2评估报告.json"):
        """导出评估报告为JSON文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 评估报告已保存至: {filename}")


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序入口"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         📊 Week 2 学习成果评估工具 v1.0                      ║
║                                                              ║
║         评估内容：                                            ║
║         1. 知识点掌握度检测                                    ║
║         2. 代码实践完成度检查                                 ║
║         3. 学习时间分布分析                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    tool = LearningAssessmentTool()
    
    # ========== 示例数据（请根据实际情况修改）==========
    
    # 示例1：知识点答题记录（请修改为实际作答）
    # 格式：题目ID -> 用户选择的选项字母
    user_answers = {
        1: "C",    # 监督学习划分比例
        2: "A",    # Q/K/V矩阵
        3: "B",    # 交叉熵损失
        4: "C",    # Adam优化器
        5: "B",    # F1-Score
        6: "B",    # ROC-AUC
        7: "B",    # 偏差-方差
        8: "B",    # Z-score
        9: "B",    # One-Hot
        10: "D"    # 过拟合
    }
    
    # 示例2：已完成的代码任务
    completed_code_tasks = [
        "实现简单分类器",
        "注意力机制演示",
        "优化器对比实验",
        "学习率影响分析",
        "评估指标可视化",
        "模型诊断实践"
    ]
    
    # 示例3：每日学习时长记录（分钟）
    study_records = [
        {"day": "Day8", "actual_minutes": 115, "topic": "监督学习+注意力"},
        {"day": "Day9", "actual_minutes": 145, "topic": "损失函数+优化器"},
        {"day": "Day10", "actual_minutes": 130, "topic": "模型评估+算法"},
        {"day": "Day11", "actual_minutes": 140, "topic": "特征工程+工具"},
        {"day": "Day12", "actual_minutes": 175, "topic": "综合项目"},
        {"day": "Day13", "actual_minutes": 95, "topic": "知识整理"}
    ]
    
    # ========== 执行评估 ==========
    
    # 功能1：知识点掌握度
    knowledge_results = tool.assess_knowledge_mastery(user_answers)
    tool.print_knowledge_assessment(knowledge_results)
    
    # 功能2：代码完成度
    code_results = tool.check_code_completion(completed_code_tasks)
    tool.print_code_completion(code_results)
    
    # 功能3：时间分布
    time_results = tool.analyze_time_distribution(study_records)
    tool.print_time_analysis(time_results)
    
    # 综合评估报告
    report = tool.generate_comprehensive_report(knowledge_results, code_results, time_results)
    tool.print_comprehensive_report(report)
    
    # 导出报告
    tool.export_report(report, "Week2评估报告.json")
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)


if __name__ == "__main__":
    main()
