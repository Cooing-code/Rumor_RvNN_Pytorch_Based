# 基于树结构递归神经网络（RvNN）的谣言检测模型

本项目是基于树结构递归神经网络（RvNN）的谣言检测方法的PyTorch实现。该模型利用社交媒体上谣言传播的树状结构特征，通过自底向上（BU-RvNN）和自顶向下（TD-RvNN）两种递归神经网络结构来捕捉谣言传播模式。

## 模型结构

该模型包含以下主要组件：

1. **自底向上RvNN (BU-RvNN)**：从叶节点到根节点递归聚合信息，捕捉局部特征形成全局表示。
2. **自顶向下RvNN (TD-RvNN)**：从根节点到叶节点传递信息，强化节点特征。
3. **GAN架构**：包含两个生成器和一个判别器，通过对抗训练提高模型性能。

## 文件结构

```
RumorRvNN/
├── data/               # 数据处理相关代码
├── models/             # 模型定义
│   ├── bu_rvnn.py      # 自底向上RvNN模型
│   ├── td_rvnn.py      # 自顶向下RvNN模型
│   ├── gan_model.py    # GAN模型
│   └── base_model.py   # 基础模型类
├── utils/              # 工具函数
│   ├── data_utils.py   # 数据处理工具
│   └── eval_utils.py   # 评估工具
├── train/              # 训练相关代码
│   ├── train_rvnn.py   # RvNN模型训练
│   └── train_gan.py    # GAN模型训练
└── main.py             # 主程序入口
```

## 使用方法

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- scikit-learn

### 训练模型

```bash
python main.py --mode train --model bu_rvnn --data_path path/to/data
```

### 评估模型

```bash
python main.py --mode test --model td_rvnn --checkpoint path/to/checkpoint
```

## 引用

如果您使用了本代码，请引用原论文：

```
@article{ma2018rumor,
  title={Rumor Detection on Twitter with Tree-structured Recursive Neural Networks},
  author={Ma, Jing and Gao, Wei and Wong, Kam-Fai},
  journal={Association for Computational Linguistics},
  year={2018}
}
``` 