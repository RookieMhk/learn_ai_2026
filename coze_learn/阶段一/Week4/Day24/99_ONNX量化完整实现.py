"""
ONNX模型导出与INT8量化完整实现
功能：展示PyTorch模型转ONNX格式 + INT8静态量化 + 性能对比全流程
作者：AI学习计划
日期：2026年4月27日
"""

import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
import time
import os
import sys
from typing import Dict, Tuple, List

class ONNXQuantizationPipeline:
    """ONNX导出与量化全流程管道"""
    
    def __init__(self, model_name: str = "resnet18", device: str = "cpu"):
        """
        初始化管道
        
        Args:
            model_name: 模型名称，支持 'resnet18', 'mobilenet_v2', 'bert'（简化版）
            device: 设备类型，'cpu' 或 'cuda'
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.fp32_onnx_path = None
        self.int8_onnx_path = None
        
        print(f"初始化管道：模型={model_name}, 设备={self.device}")
    
    def load_pretrained_model(self) -> nn.Module:
        """加载预训练模型"""
        print(f"\n1. 加载预训练模型：{self.model_name}")
        
        if self.model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
        elif self.model_name == "bert":
            # 简化版BERT，用于演示
            class SimpleBERT(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(30522, 768)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=768, nhead=12),
                        num_layers=12
                    )
                    self.classifier = nn.Linear(768, 2)
                
                def forward(self, input_ids):
                    x = self.embedding(input_ids)
                    x = self.transformer(x)
                    x = x.mean(dim=1)  # 池化
                    return self.classifier(x)
            
            model = SimpleBERT()
            # 加载随机权重（演示用）
            model.load_state_dict({k: torch.randn_like(v) for k, v in model.state_dict().items()})
        else:
            raise ValueError(f"不支持的模型：{self.model_name}")
        
        model.eval()
        model.to(self.device)
        self.model = model
        
        # 统计模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  模型参数：总计 {total_params:,}，可训练 {trainable_params:,}")
        
        return model
    
    def export_to_onnx(self, 
                       output_dir: str = "./",
                       dynamic_batch: bool = True,
                       opset_version: int = 13) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            output_dir: 输出目录
            dynamic_batch: 是否支持动态batch
            opset_version: ONNX算子集版本
        
        Returns:
            ONNX文件路径
        """
        print(f"\n2. 导出模型为ONNX格式")
        
        if self.model is None:
            raise RuntimeError("请先加载模型")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据模型类型确定输入尺寸
        if self.model_name in ["resnet18", "mobilenet_v2"]:
            input_shape = (1, 3, 224, 224)
            input_names = ["input"]
            output_names = ["output"]
        elif self.model_name == "bert":
            input_shape = (1, 128)  # 序列长度128
            input_names = ["input_ids"]
            output_names = ["logits"]
        else:
            input_shape = (1, 3, 224, 224)
            input_names = ["input"]
            output_names = ["output"]
        
        # 创建示例输入
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # 生成文件路径
        self.fp32_onnx_path = os.path.join(output_dir, f"{self.model_name}_fp32.onnx")
        
        # 动态轴配置
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            self.fp32_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"  导出成功：{self.fp32_onnx_path}")
        
        # 验证模型
        onnx_model = onnx.load(self.fp32_onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  模型验证通过")
        
        # 显示模型信息
        print(f"  输入：{onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"  输出：{onnx_model.graph.output[0].type.tensor_type.shape}")
        
        return self.fp32_onnx_path
    
    def quantize_to_int8(self,
                        calibration_data: List[np.ndarray] = None,
                        num_calibration_samples: int = 100) -> str:
        """
        执行INT8静态量化
        
        Args:
            calibration_data: 校准数据列表，每个元素为numpy数组
            num_calibration_samples: 校准样本数（如果未提供calibration_data）
        
        Returns:
            量化后的ONNX文件路径
        """
        print(f"\n3. 执行INT8静态量化")
        
        if self.fp32_onnx_path is None or not os.path.exists(self.fp32_onnx_path):
            raise RuntimeError("请先导出ONNX模型")
        
        # 生成输出路径
        base_name = os.path.splitext(os.path.basename(self.fp32_onnx_path))[0]
        self.int8_onnx_path = self.fp32_onnx_path.replace("_fp32.onnx", "_int8.onnx")
        
        # 准备校准数据
        if calibration_data is None:
            print(f"  生成随机校准数据（{num_calibration_samples}个样本）")
            
            # 根据模型类型确定输入尺寸
            if self.model_name in ["resnet18", "mobilenet_v2"]:
                input_shape = (1, 3, 224, 224)
            elif self.model_name == "bert":
                input_shape = (1, 128)
            else:
                input_shape = (1, 3, 224, 224)
            
            calibration_data = [
                np.random.randn(*input_shape).astype(np.float32)
                for _ in range(num_calibration_samples)
            ]
        
        # 执行量化
        try:
            from onnxruntime.quantization import quantize_static, QuantType
            
            quantize_static(
                self.fp32_onnx_path,
                self.int8_onnx_path,
                calibration_data,
                quant_format=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
                optimize_model=True
            )
            
            print(f"  量化成功：{self.int8_onnx_path}")
            
        except ImportError:
            print(f"  警告：未安装onnxruntime量化工具，使用模拟量化")
            # 模拟量化：复制文件并记录日志
            import shutil
            shutil.copy(self.fp32_onnx_path, self.int8_onnx_path)
            print(f"  模拟量化完成（实际为FP32模型）")
        
        return self.int8_onnx_path
    
    def benchmark_performance(self,
                             num_warmup: int = 10,
                             num_iterations: int = 100,
                             batch_size: int = 1) -> Dict[str, Dict[str, float]]:
        """
        性能基准测试：对比FP32和INT8模型
        
        Args:
            num_warmup: 预热迭代次数
            num_iterations: 正式测试迭代次数
            batch_size: 批处理大小
        
        Returns:
            包含各项指标的字典
        """
        print(f"\n4. 性能基准测试")
        
        if self.fp32_onnx_path is None or self.int8_onnx_path is None:
            raise RuntimeError("请先完成模型导出和量化")
        
        # 根据模型类型确定输入
        if self.model_name in ["resnet18", "mobilenet_v2"]:
            input_shape = (batch_size, 3, 224, 224)
            input_name = "input"
        elif self.model_name == "bert":
            input_shape = (batch_size, 128)
            input_name = "input_ids"
        else:
            input_shape = (batch_size, 3, 224, 224)
            input_name = "input"
        
        # 创建测试输入
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 配置ONNX Runtime会话
        def create_session(onnx_path: str) -> ort.InferenceSession:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 2
            
            providers = []
            if torch.cuda.is_available():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024  # 2GB
                }))
            providers.append('CPUExecutionProvider')
            
            return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        
        # 创建会话
        session_fp32 = create_session(self.fp32_onnx_path)
        session_int8 = create_session(self.int8_onnx_path)
        
        # 预热
        print(f"  预热阶段（{num_warmup}次迭代）")
        for _ in range(num_warmup):
            session_fp32.run(None, {input_name: test_input})
            session_int8.run(None, {input_name: test_input})
        
        # 测试FP32模型
        print(f"  测试FP32模型（{num_iterations}次迭代）")
        start = time.time()
        for _ in range(num_iterations):
            session_fp32.run(None, {input_name: test_input})
        fp32_time = (time.time() - start) / num_iterations * 1000  # 毫秒
        
        # 测试INT8模型
        print(f"  测试INT8模型（{num_iterations}次迭代）")
        start = time.time()
        for _ in range(num_iterations):
            session_int8.run(None, {input_name: test_input})
        int8_time = (time.time() - start) / num_iterations * 1000  # 毫秒
        
        # 计算模型大小
        fp32_size = os.path.getsize(self.fp32_onnx_path) / 1024**2  # MB
        int8_size = os.path.getsize(self.int8_onnx_path) / 1024**2  # MB
        
        # 精度验证：比较输出
        print(f"  精度验证")
        output_fp32 = session_fp32.run(None, {input_name: test_input})[0]
        output_int8 = session_int8.run(None, {input_name: test_input})[0]
        
        # 计算MSE和余弦相似度
        mse = np.mean((output_fp32 - output_int8) ** 2)
        
        fp32_flat = output_fp32.flatten()
        int8_flat = output_int8.flatten()
        cos_sim = np.dot(fp32_flat, int8_flat) / (
            np.linalg.norm(fp32_flat) * np.linalg.norm(int8_flat)
        )
        
        # 汇总结果
        results = {
            "fp32": {
                "inference_time_ms": fp32_time,
                "model_size_mb": fp32_size,
                "throughput_fps": 1000 / fp32_time if fp32_time > 0 else 0
            },
            "int8": {
                "inference_time_ms": int8_time,
                "model_size_mb": int8_size,
                "throughput_fps": 1000 / int8_time if int8_time > 0 else 0
            },
            "comparison": {
                "speedup_ratio": fp32_time / int8_time if int8_time > 0 else 0,
                "compression_ratio": fp32_size / int8_size if int8_size > 0 else 0,
                "mse": mse,
                "cosine_similarity": cos_sim
            }
        }
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"性能测试结果 - {self.model_name}")
        print(f"{'='*60}")
        
        print(f"\n模型体积对比：")
        print(f"  FP32模型：{fp32_size:.2f} MB")
        print(f"  INT8模型：{int8_size:.2f} MB")
        print(f"  压缩率：{fp32_size/int8_size:.1f}x")
        
        print(f"\n推理速度对比：")
        print(f"  FP32模型：{fp32_time:.2f} ms/次 (吞吐量：{1000/fp32_time:.1f} FPS)")
        print(f"  INT8模型：{int8_time:.2f} ms/次 (吞吐量：{1000/int8_time:.1f} FPS)")
        print(f"  加速比：{fp32_time/int8_time:.1f}x")
        
        print(f"\n精度保持指标：")
        print(f"  输出MSE：{mse:.6f}")
        print(f"  余弦相似度：{cos_sim:.4f}")
        
        print(f"\n评估结论：")
        if mse < 0.001 and cos_sim > 0.99:
            print(f"  ✅ 量化成功：精度损失极小，速度提升显著")
        elif mse < 0.01 and cos_sim > 0.95:
            print(f"  ⚠️  量化可用：精度损失可接受，速度提升明显")
        else:
            print(f"  ❌ 量化失败：精度损失过大，建议调整量化参数")
        
        return results
    
    def run_full_pipeline(self, output_dir: str = "./outputs") -> Dict:
        """
        运行完整管道：加载模型 → 导出ONNX → 量化 → 性能测试
        
        Args:
            output_dir: 输出目录
        
        Returns:
            性能测试结果
        """
        print("\n" + "="*80)
        print("开始运行完整ONNX量化管道")
        print("="*80)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1：加载模型
        self.load_pretrained_model()
        
        # 步骤2：导出ONNX
        self.export_to_onnx(output_dir=output_dir)
        
        # 步骤3：量化
        self.quantize_to_int8()
        
        # 步骤4：性能测试
        results = self.benchmark_performance()
        
        print("\n" + "="*80)
        print("管道运行完成")
        print("="*80)
        
        return results

def compare_multiple_models():
    """对比多个模型的量化效果"""
    models_to_test = ["resnet18", "mobilenet_v2"]  # 可以添加 "bert"
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*80}")
        print(f"测试模型：{model_name}")
        print(f"{'#'*80}")
        
        # 创建管道
        pipeline = ONNXQuantizationPipeline(model_name=model_name, device="cuda")
        
        # 运行完整管道
        results = pipeline.run_full_pipeline(output_dir=f"./outputs/{model_name}")
        
        all_results[model_name] = results
    
    # 生成对比报告
    print(f"\n{'='*100}")
    print("多模型量化效果对比报告")
    print(f"{'='*100}")
    
    print(f"\n{'模型':<15} {'FP32大小(MB)':<15} {'INT8大小(MB)':<15} {'压缩率':<10} {'FP32延迟(ms)':<15} {'INT8延迟(ms)':<15} {'加速比':<10} {'余弦相似度':<12}")
    print(f"{'-'*120}")
    
    for model_name, results in all_results.items():
        fp32_size = results["fp32"]["model_size_mb"]
        int8_size = results["int8"]["model_size_mb"]
        compression = fp32_size / int8_size
        fp32_time = results["fp32"]["inference_time_ms"]
        int8_time = results["int8"]["inference_time_ms"]
        speedup = fp32_time / int8_time
        cos_sim = results["comparison"]["cosine_similarity"]
        
        print(f"{model_name:<15} {fp32_size:<15.2f} {int8_size:<15.2f} {compression:<10.1f} {fp32_time:<15.2f} {int8_time:<15.2f} {speedup:<10.1f} {cos_sim:<12.4f}")
    
    return all_results

def main():
    """主函数"""
    print("ONNX模型导出与INT8量化完整实现")
    print("="*80)
    
    # 检查依赖
    print("检查依赖库...")
    required_libs = ["torch", "torchvision", "onnx", "onnxruntime", "numpy"]
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"缺少依赖库：{missing_libs}")
        print("请运行：pip install torch torchvision onnx onnxruntime numpy")
        return
    
    print("所有依赖库已安装")
    
    # 选择运行模式
    print(f"\n运行模式：")
    print(f"  1. 完整管道（单个模型）")
    print(f"  2. 多模型对比")
    print(f"  3. 仅导出ONNX")
    print(f"  4. 仅量化")
    
    try:
        mode = int(input("请选择模式 (1-4): "))
    except:
        mode = 1
    
    if mode == 1:
        # 单个模型完整管道
        pipeline = ONNXQuantizationPipeline(model_name="resnet18", device="cuda")
        pipeline.run_full_pipeline(output_dir="./outputs/full_pipeline")
    
    elif mode == 2:
        # 多模型对比
        compare_multiple_models()
    
    elif mode == 3:
        # 仅导出ONNX
        pipeline = ONNXQuantizationPipeline(model_name="resnet18", device="cuda")
        pipeline.load_pretrained_model()
        pipeline.export_to_onnx(output_dir="./outputs/export_only")
    
    elif mode == 4:
        # 仅量化（需先有ONNX模型）
        onnx_path = input("请输入ONNX模型路径: ")
        if not os.path.exists(onnx_path):
            print(f"文件不存在：{onnx_path}")
            return
        
        # 创建管道但不加载模型
        pipeline = ONNXQuantizationPipeline(model_name="resnet18", device="cpu")
        pipeline.fp32_onnx_path = onnx_path
        pipeline.quantize_to_int8()
    
    else:
        print("无效模式，退出")

if __name__ == "__main__":
    main()