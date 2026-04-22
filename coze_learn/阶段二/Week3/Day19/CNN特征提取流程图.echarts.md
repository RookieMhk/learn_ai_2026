```echarts
{
  "title": {
    "text": "CNN特征提取流程图",
    "subtext": "从输入图像到分类输出的完整流程",
    "left": "center",
    "textStyle": {
      "fontSize": 18,
      "fontWeight": "bold"
    }
  },
  "tooltip": {
    "trigger": "item",
    "formatter": "{b}: {c}"
  },
  "legend": {
    "data": ["输入层", "卷积层", "激活层", "池化层", "全连接层", "输出层"],
    "top": "10%",
    "textStyle": {
      "color": "#666"
    }
  },
  "series": [
    {
      "type": "graph",
      "layout": "none",
      "symbolSize": 60,
      "roam": false,
      "label": {
        "show": true,
        "fontSize": 12,
        "fontWeight": "bold"
      },
      "edgeSymbol": ["circle", "arrow"],
      "edgeSymbolSize": [4, 10],
      "edgeLabel": {
        "fontSize": 10
      },
      "data": [
        {
          "name": "输入图像",
          "x": 100,
          "y": 300,
          "symbol": "rect",
          "itemStyle": {
            "color": "#91cc75"
          },
          "label": {
            "formatter": "输入图像\n(224×224×3)"
          }
        },
        {
          "name": "卷积层1",
          "x": 250,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#5470c6"
          },
          "label": {
            "formatter": "卷积层1\n64个3×3卷积核"
          }
        },
        {
          "name": "ReLU激活",
          "x": 400,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#ee6666"
          },
          "label": {
            "formatter": "ReLU激活\nf(x)=max(0,x)"
          }
        },
        {
          "name": "池化层1",
          "x": 550,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#fac858"
          },
          "label": {
            "formatter": "最大池化\n2×2, stride=2"
          }
        },
        {
          "name": "卷积层2",
          "x": 700,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#5470c6"
          },
          "label": {
            "formatter": "卷积层2\n128个3×3卷积核"
          }
        },
        {
          "name": "ReLU激活2",
          "x": 850,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#ee6666"
          },
          "label": {
            "formatter": "ReLU激活\n非线性变换"
          }
        },
        {
          "name": "池化层2",
          "x": 1000,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#fac858"
          },
          "label": {
            "formatter": "最大池化\n2×2, stride=2"
          }
        },
        {
          "name": "卷积层3",
          "x": 1150,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#5470c6"
          },
          "label": {
            "formatter": "卷积层3\n256个3×3卷积核"
          }
        },
        {
          "name": "ReLU激活3",
          "x": 1300,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#ee6666"
          },
          "label": {
            "formatter": "ReLU激活\n特征增强"
          }
        },
        {
          "name": "全局池化",
          "x": 1450,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#73c0de"
          },
          "label": {
            "formatter": "全局平均池化\n空间维度压缩"
          }
        },
        {
          "name": "全连接层",
          "x": 1600,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#9a60b4"
          },
          "label": {
            "formatter": "全连接层\n1024个神经元"
          }
        },
        {
          "name": "Dropout",
          "x": 1750,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#fc8452"
          },
          "label": {
            "formatter": "Dropout\np=0.5防止过拟合"
          }
        },
        {
          "name": "输出层",
          "x": 1900,
          "y": 300,
          "symbol": "roundRect",
          "itemStyle": {
            "color": "#3ba272"
          },
          "label": {
            "formatter": "Softmax输出\n1000个类别概率"
          }
        }
      ],
      "links": [
        {
          "source": "输入图像",
          "target": "卷积层1",
          "label": {
            "show": true,
            "formatter": "卷积运算\n局部特征提取"
          },
          "lineStyle": {
            "color": "#5470c6",
            "width": 2
          }
        },
        {
          "source": "卷积层1",
          "target": "ReLU激活",
          "label": {
            "show": true,
            "formatter": "非线性激活"
          },
          "lineStyle": {
            "color": "#ee6666",
            "width": 2
          }
        },
        {
          "source": "ReLU激活",
          "target": "池化层1",
          "label": {
            "show": true,
            "formatter": "下采样\n特征不变性"
          },
          "lineStyle": {
            "color": "#fac858",
            "width": 2
          }
        },
        {
          "source": "池化层1",
          "target": "卷积层2",
          "label": {
            "show": true,
            "formatter": "更深层特征"
          },
          "lineStyle": {
            "color": "#5470c6",
            "width": 2
          }
        },
        {
          "source": "卷积层2",
          "target": "ReLU激活2",
          "lineStyle": {
            "color": "#ee6666",
            "width": 2
          }
        },
        {
          "source": "ReLU激活2",
          "target": "池化层2",
          "lineStyle": {
            "color": "#fac858",
            "width": 2
          }
        },
        {
          "source": "池化层2",
          "target": "卷积层3",
          "lineStyle": {
            "color": "#5470c6",
            "width": 2
          }
        },
        {
          "source": "卷积层3",
          "target": "ReLU激活3",
          "lineStyle": {
            "color": "#ee6666",
            "width": 2
          }
        },
        {
          "source": "ReLU激活3",
          "target": "全局池化",
          "label": {
            "show": true,
            "formatter": "空间压缩"
          },
          "lineStyle": {
            "color": "#73c0de",
            "width": 2
          }
        },
        {
          "source": "全局池化",
          "target": "全连接层",
          "label": {
            "show": true,
            "formatter": "特征向量化"
          },
          "lineStyle": {
            "color": "#9a60b4",
            "width": 2
          }
        },
        {
          "source": "全连接层",
          "target": "Dropout",
          "lineStyle": {
            "color": "#fc8452",
            "width": 2
          }
        },
        {
          "source": "Dropout",
          "target": "输出层",
          "label": {
            "show": true,
            "formatter": "分类决策"
          },
          "lineStyle": {
            "color": "#3ba272",
            "width": 2
          }
        }
      ],
      "lineStyle": {
        "opacity": 0.8,
        "width": 2,
        "curveness": 0.1
      }
    }
  ]
}
```