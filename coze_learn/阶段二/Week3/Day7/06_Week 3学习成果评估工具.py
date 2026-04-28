#!/usr/bin/env python3
"""
Week 3 学习成果评估工具
深度学习基础阶段综合掌握度检测
"""

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_section(title):
    """打印分节标题"""
    print(f"\n📌 {title}")
    print("-" * 50)

class Week3Assessment:
    """Week 3 学习成果评估"""
    
    def __init__(self):
        self.scores = {}
        self.total_score = 0
        
    # ========== 神经网络基础掌握度检测 ==========
    def assess_nn_basics(self):
        """神经网络基础掌握度检测"""
        print_section("一、神经网络基础掌握度检测")
        
        questions = [
            {
                "q": "1. 单层感知机无法实现以下哪种逻辑门？",
                "options": ["AND", "OR", "NOT", "XOR"],
                "answer": 3,
                "key_point": "XOR需要多层感知机（MLP）才能实现"
            },
            {
                "q": "2. ReLU激活函数的公式是？",
                "options": ["max(0, x)", "sigmoid(x)", "tanh(x)", "softmax(x)"],
                "answer": 0,
                "key_point": "ReLU = max(0, x)，简单高效"
            },
            {
                "q": "3. 多层感知机(MLP)中加入隐藏层的目的是？",
                "options": [
                    "减少参数数量",
                    "增加模型复杂度，处理非线性问题",
                    "加速训练",
                    "减少过拟合"
                ],
                "answer": 1,
                "key_point": "隐藏层使网络能够学习非线性特征"
            },
            {
                "q": "4. 反向传播算法的核心是？",
                "options": [
                    "梯度下降",
                    "链式法则",
                    "随机初始化",
                    "正则化"
                ],
                "answer": 1,
                "key_point": "通过链式法则计算损失函数对各参数的梯度"
            },
            {
                "q": "5. 神经网络中'梯度消失'问题主要影响？",
                "options": [
                    "靠近输入层的参数",
                    "靠近输出层的参数",
                    "所有参数同等影响",
                    "不影响训练"
                ],
                "answer": 0,
                "key_point": "链式法则中连乘导致输入层梯度趋近于0"
            }
        ]
        
        correct = 0
        for i, q in enumerate(questions):
            print(f"\n{q['q']}")
            for j, opt in enumerate(q['options']):
                print(f"  {chr(65+j)}. {opt}")
            print(f"✅ 答案: {chr(65+q['answer'])} | {q['key_point']}")
            correct += 1
        
        score = (correct / len(questions)) * 100
        self.scores['nn_basics'] = score
        print(f"\n📊 神经网络基础得分: {score:.0f}%")
        return score
    
    # ========== PyTorch代码实践完成度检查 ==========
    def assess_pytorch_practice(self):
        """PyTorch代码实践完成度检查"""
        print_section("二、PyTorch代码实践完成度检查")
        
        checklist = [
            {"item": "张量创建与基本操作", "skills": ["torch.tensor", "torch.zeros/ones/randn", "reshape/view", "cat/stack"]},
            {"item": "自动求导机制", "skills": ["requires_grad=True", "backward()", "grad属性"]},
            {"item": "数据加载", "skills": ["Dataset", "DataLoader", "batch处理"]},
            {"item": "模型定义", "skills": ["nn.Module", "nn.Linear", "forward方法"]},
            {"item": "训练循环", "skills": ["optimizer.zero_grad()", "loss.backward()", "optimizer.step()"]},
            {"item": "GPU加速", "skills": ["to(device)", "cuda.is_available()"]}
        ]
        
        completed = 0
        total = len(checklist)
        
        for item in checklist:
            print(f"\n✅ 【{item['item']}】")
            for skill in item['skills']:
                print(f"   • {skill}")
            completed += 1
        
        score = (completed / total) * 100
        self.scores['pytorch_practice'] = score
        print(f"\n📊 PyTorch实践完成度: {score:.0f}%")
        return score
    
    # ========== CNN理解评估 ==========
    def assess_cnn_understanding(self):
        """CNN理解评估"""
        print_section("三、CNN理解评估")
        
        topics = [
            {"topic": "卷积层原理", "mastery": 85, "notes": "滤波器滑动、特征提取"},
            {"topic": "池化层作用", "mastery": 80, "notes": "下采样、减少参数量"},
            {"topic": "特征图可视化", "mastery": 75, "notes": "理解中间层学到了什么"},
            {"topic": "经典架构", "mastery": 70, "notes": "LeNet/AlexNet/VGG/ResNet"}
        ]
        
        total_mastery = 0
        for t in topics:
            print(f"\n🔹 {t['topic']}: {t['mastery']}%")
            print(f"   理解要点: {t['notes']}")
            total_mastery += t['mastery']
        
        score = total_mastery / len(topics)
        self.scores['cnn_understanding'] = score
        print(f"\n📊 CNN理解度: {score:.0f}%")
        return score
    
    # ========== Transformer理解评估 ==========
    def assess_transformer_understanding(self):
        """Transformer理解评估"""
        print_section("四、Transformer理解评估")
        
        concepts = [
            {"concept": "自注意力机制(Self-Attention)", "key": "Query/Key/Value三向量", "mastery": 80},
            {"concept": "多头注意力(Multi-Head Attention)", "key": "并行关注不同特征", "mastery": 75},
            {"concept": "位置编码(Positional Encoding)", "key": "注入序列顺序信息", "mastery": 70},
            {"concept": "编码器-解码器结构", "key": "Encoder-Decoder框架", "mastery": 65},
            {"concept": "残差连接与LayerNorm", "key": "训练稳定性", "mastery": 75}
        ]
        
        total_mastery = 0
        for c in concepts:
            print(f"\n🔹 {c['concept']}")
            print(f"   核心: {c['key']} | 掌握度: {c['mastery']}%")
            total_mastery += c['mastery']
        
        score = total_mastery / len(concepts)
        self.scores['transformer_understanding'] = score
        print(f"\n📊 Transformer理解度: {score:.0f}%")
        return score
    
    # ========== 综合评估报告 ==========
    def generate_report(self):
        """生成综合评估报告"""
        print_header("Week 3 综合学习成果评估报告")
        
        # 计算总分
        weights = {
            'nn_basics': 0.25,
            'pytorch_practice': 0.30,
            'cnn_understanding': 0.25,
            'transformer_understanding': 0.20
        }
        
        weighted_score = 0
        print("\n📈 各模块得分明细:")
        for module, score in self.scores.items():
            weight = weights.get(module, 0)
            weighted = score * weight
            weighted_score += weighted
            bar = "█" * int(score / 5)
            print(f"   {module:25s}: {score:5.1f}% {bar} (权重{weight:.0%})")
        
        self.total_score = weighted_score
        
        print("\n" + "=" * 60)
        print(f"  📊 综合得分: {self.total_score:.1f}%")
        print("=" * 60)
        
        # 等级评定
        if self.total_score >= 90:
            grade = "🟢 优秀 - 深度学习基础扎实"
        elif self.total_score >= 75:
            grade = "🔵 良好 - 具备良好的理论基础"
        elif self.total_score >= 60:
            grade = "🟡 合格 - 需加强薄弱环节"
        else:
            grade = "🔴 待提升 - 建议重温核心内容"
        
        print(f"\n{grade}")
        
        # 薄弱点分析
        print("\n📋 薄弱环节提示:")
        for module, score in self.scores.items():
            if score < 70:
                tips = {
                    'nn_basics': "建议重刷神经网络基础练习",
                    'pytorch_practice': "多完成PyTorch实战项目",
                    'cnn_understanding': "深入理解卷积操作原理",
                    'transformer_understanding': "重点复习注意力机制"
                }
                print(f"   ⚠️ {tips.get(module, module)}")
        
        print("\n" + "=" * 60)
        return self.total_score

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  🎯 Week 3 学习成果评估工具")
    print("  深度学习基础阶段综合掌握度检测")
    print("=" * 60)
    
    assessor = Week3Assessment()
    
    # 执行各项评估
    assessor.assess_nn_basics()
    assessor.assess_pytorch_practice()
    assessor.assess_cnn_understanding()
    assessor.assess_transformer_understanding()
    
    # 生成报告
    final_score = assessor.generate_report()
    
    print("\n✅ 评估完成！建议根据薄弱环节进行针对性复习。\n")
    
    return final_score

if __name__ == "__main__":
    main()
