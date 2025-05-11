import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class BURvNN(BaseModel):
    """
    自底向上递归神经网络 (Bottom-Up RvNN)
    从叶节点到根节点递归聚合信息
    """
    def __init__(self, vocab_size, hidden_size, num_classes, dropout=0.5):
        """
        初始化BU-RvNN模型
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
            num_classes (int): 类别数量
            dropout (float): Dropout概率
        """
        super(BURvNN, self).__init__(vocab_size, hidden_size, num_classes)
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # GRU参数
        # 重置门参数
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        
        # 更新门参数
        self.W_z = nn.Linear(hidden_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        
        # 候选隐藏状态参数
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def node_forward(self, x, child_h):
        """
        单个节点的前向传播
        
        参数:
            x: 节点输入向量
            child_h: 子节点隐藏状态的总和
            
        返回:
            节点的隐藏状态
        """
        # 嵌入输入
        x_emb = self.embedding(x) if isinstance(x, torch.LongTensor) else x
        
        # GRU计算
        r = torch.sigmoid(self.W_r(x_emb) + self.U_r(child_h))
        z = torch.sigmoid(self.W_z(x_emb) + self.U_z(child_h))
        h_tilde = torch.tanh(self.W_h(x_emb) + self.U_h(child_h * r))
        h = (1 - z) * child_h + z * h_tilde
        
        return h
    
    def forward(self, tree):
        """
        前向传播
        
        参数:
            tree: 输入树结构
            
        返回:
            类别预测概率
        """
        # 自底向上遍历树
        node_hidden = {}
        
        # 后序遍历树（自底向上）
        for node_id in tree.post_order():
            node = tree.get_node(node_id)
            
            if node.is_leaf():
                # 叶节点
                node_hidden[node_id] = self.node_forward(
                    node.data, 
                    torch.zeros(self.hidden_size, device=self.embedding.weight.device)
                )
            else:
                # 非叶节点，聚合子节点信息
                child_h_sum = torch.zeros(self.hidden_size, device=self.embedding.weight.device)
                for child_id in node.children:
                    child_h_sum += node_hidden[child_id]
                
                node_hidden[node_id] = self.node_forward(node.data, child_h_sum)
        
        # 使用根节点的隐藏状态进行分类
        root_id = tree.root
        root_hidden = node_hidden[root_id]
        root_hidden = self.dropout(root_hidden)
        
        # 输出层
        logits = self.output_layer(root_hidden)
        return F.softmax(logits, dim=-1)
    
    def compute_loss(self, tree, labels):
        """
        计算损失
        
        参数:
            tree: 输入树结构
            labels: 真实标签
            
        返回:
            损失值
        """
        predictions = self.forward(tree)
        loss = F.cross_entropy(predictions, labels)
        return loss 