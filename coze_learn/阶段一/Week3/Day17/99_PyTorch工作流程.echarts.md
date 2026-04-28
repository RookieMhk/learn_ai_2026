```echarts
{
  "title": {
    "text": "PyTorch深度学习工作全流程",
    "subtext": "从数据加载到模型部署的完整流程示意图",
    "left": "center",
    "top": "10",
    "textStyle": {
      "fontSize": 20,
      "fontWeight": "bold"
    }
  },
  "tooltip": {
    "trigger": "item",
    "formatter": "{b}: {c}"
  },
  "legend": {
    "data": ["数据准备", "模型构建", "训练优化", "评估部署"],
    "top": "50",
    "left": "center",
    "itemWidth": 20,
    "itemHeight": 14
  },
  "series": [
    {
      "type": "graph",
      "layout": "none",
      "symbolSize": 70,
      "roam": true,
      "edgeSymbol": ["circle", "arrow"],
      "edgeSymbolSize": [4, 10],
      "edgeLabel": {
        "fontSize": 12
      },
      "force": {
        "repulsion": 200,
        "edgeLength": 150
      },
      "draggable": true,
      "lineStyle": {
        "color": "source",
        "curveness": 0.3,
        "width": 2
      },
      "label": {
        "show": true,
        "position": "inside",
        "fontSize": 12,
        "fontWeight": "bold"
      },
      "categories": [
        {
          "name": "数据准备",
          "itemStyle": {
            "color": "#5470c6"
          }
        },
        {
          "name": "模型构建",
          "itemStyle": {
            "color": "#91cc75"
          }
        },
        {
          "name": "训练优化",
          "itemStyle": {
            "color": "#fac858"
          }
        },
        {
          "name": "评估部署",
          "itemStyle": {
            "color": "#ee6666"
          }
        }
      ],
      "nodes": [
        {
          "id": "0",
          "name": "数据采集",
          "x": 100,
          "y": 300,
          "symbolSize": 60,
          "category": 0,
          "label": {
            "formatter": "数据采集\n(图像/文本/音频)"
          }
        },
        {
          "id": "1",
          "name": "数据预处理",
          "x": 250,
          "y": 300,
          "symbolSize": 60,
          "category": 0,
          "label": {
            "formatter": "数据预处理\n(清洗/归一化/增强)"
          }
        },
        {
          "id": "2",
          "name": "Dataset定义",
          "x": 400,
          "y": 300,
          "symbolSize": 60,
          "category": 0,
          "label": {
            "formatter": "Dataset定义\n(__len__, __getitem__)"
          }
        },
        {
          "id": "3",
          "name": "DataLoader",
          "x": 550,
          "y": 300,
          "symbolSize": 60,
          "category": 0,
          "label": {
            "formatter": "DataLoader\n(批量加载/打乱/多进程)"
          }
        },
        {
          "id": "4",
          "name": "模型架构",
          "x": 400,
          "y": 150,
          "symbolSize": 60,
          "category": 1,
          "label": {
            "formatter": "模型架构设计\n(nn.Module子类)"
          }
        },
        {
          "id": "5",
          "name": "层定义",
          "x": 550,
          "y": 150,
          "symbolSize": 60,
          "category": 1,
          "label": {
            "formatter": "层定义\n(Linear/Conv2d/LSTM)"
          }
        },
        {
          "id": "6",
          "name": "前向传播",
          "x": 700,
          "y": 150,
          "symbolSize": 60,
          "category": 1,
          "label": {
            "formatter": "前向传播\n(forward方法实现)"
          }
        },
        {
          "id": "7",
          "name": "损失函数",
          "x": 700,
          "y": 300,
          "symbolSize": 60,
          "category": 2,
          "label": {
            "formatter": "损失函数\n(MSELoss/CrossEntropyLoss)"
          }
        },
        {
          "id": "8",
          "name": "优化器",
          "x": 700,
          "y": 450,
          "symbolSize": 60,
          "category": 2,
          "label": {
            "formatter": "优化器\n(Adam/SGD/RMSprop)"
          }
        },
        {
          "id": "9",
          "name": "训练循环",
          "x": 850,
          "y": 300,
          "symbolSize": 70,
          "category": 2,
          "label": {
            "formatter": "训练循环\n(epoch迭代/梯度计算)"
          }
        },
        {
          "id": "10",
          "name": "验证评估",
          "x": 1000,
          "y": 300,
          "symbolSize": 60,
          "category": 3,
          "label": {
            "formatter": "验证评估\n(准确率/F1分数/AUC)"
          }
        },
        {
          "id": "11",
          "name": "模型保存",
          "x": 1000,
          "y": 150,
          "symbolSize": 60,
          "category": 3,
          "label": {
            "formatter": "模型保存\n(state_dict/checkpoint)"
          }
        },
        {
          "id": "12",
          "name": "部署推理",
          "x": 1000,
          "y": 450,
          "symbolSize": 60,
          "category": 3,
          "label": {
            "formatter": "部署推理\n(ONNX/TorchServe)"
          }
        }
      ],
      "links": [
        {
          "source": "0",
          "target": "1",
          "label": {
            "show": true,
            "formatter": "原始数据"
          }
        },
        {
          "source": "1",
          "target": "2",
          "label": {
            "show": true,
            "formatter": "标准化数据"
          }
        },
        {
          "source": "2",
          "target": "3",
          "label": {
            "show": true,
            "formatter": "数据接口"
          }
        },
        {
          "source": "4",
          "target": "5",
          "label": {
            "show": true,
            "formatter": "层配置"
          }
        },
        {
          "source": "5",
          "target": "6",
          "label": {
            "show": true,
            "formatter": "计算图"
          }
        },
        {
          "source": "3",
          "target": "9",
          "label": {
            "show": true,
            "formatter": "批量数据"
          }
        },
        {
          "source": "6",
          "target": "9",
          "label": {
            "show": true,
            "formatter": "模型输出"
          }
        },
        {
          "source": "9",
          "target": "7",
          "label": {
            "show": true,
            "formatter": "预测值"
          }
        },
        {
          "source": "7",
          "target": "9",
          "label": {
            "show": true,
            "formatter": "损失值"
          }
        },
        {
          "source": "8",
          "target": "9",
          "label": {
            "show": true,
            "formatter": "参数更新"
          }
        },
        {
          "source": "9",
          "target": "8",
          "label": {
            "show": true,
            "formatter": "梯度"
          }
        },
        {
          "source": "9",
          "target": "10",
          "label": {
            "show": true,
            "formatter": "训练完成"
          }
        },
        {
          "source": "10",
          "target": "11",
          "label": {
            "show": true,
            "formatter": "性能达标"
          }
        },
        {
          "source": "10",
          "target": "12",
          "label": {
            "show": true,
            "formatter": "生产就绪"
          }
        },
        {
          "source": "11",
          "target": "12",
          "label": {
            "show": true,
            "formatter": "加载模型"
          }
        }
      ]
    }
  ]
}
```