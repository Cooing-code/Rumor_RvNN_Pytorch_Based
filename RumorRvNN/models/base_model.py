import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class BaseModel(nn.Module):
    """
    所有模型的基类，提供通用功能
    """
    def __init__(self, vocab_size, hidden_size, num_classes):
        """
        初始化基础模型
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
            num_classes (int): 类别数量
        """
        super(BaseModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        前向传播，需要在子类中实现
        
        参数:
            x: 输入数据
            
        返回:
            输出预测
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def save_model(self, path):
        """
        保存模型参数
        
        参数:
            path (str): 保存路径
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载模型参数
        
        参数:
            path (str): 加载路径
            
        返回:
            加载后的模型
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件 {path} 不存在")
            
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")
        return self
    
    def count_parameters(self):
        """
        计算模型参数数量
        
        返回:
            参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 