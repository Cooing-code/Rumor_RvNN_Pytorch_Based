import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class TDRvNN(BaseModel):
    """
    自顶向下递归神经网络 (Top-Down RvNN)
    从根节点到叶节点传递信息
    """
    def __init__(self, vocab_size, hidden_size, num_classes, dropout=0.5):
        """
        初始化TD-RvNN模型
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
            num_classes (int): 类别数量
            dropout (float): Dropout概率
        """
        super(TDRvNN, self).__init__(vocab_size, hidden_size, num_classes)
        
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
    
    def node_forward(self, x, parent_h):
        """
        单个节点的前向传播
        
        参数:
            x: 节点输入向量
            parent_h: 父节点隐藏状态
            
        返回:
            节点的隐藏状态
        """
        # 嵌入输入
        x_emb = self.embedding(x) if isinstance(x, torch.LongTensor) else x
        
        # GRU计算
        r = torch.sigmoid(self.W_r(x_emb) + self.U_r(parent_h))
        z = torch.sigmoid(self.W_z(x_emb) + self.U_z(parent_h))
        h_tilde = torch.tanh(self.W_h(x_emb) + self.U_h(parent_h * r))
        h = (1 - z) * parent_h + z * h_tilde
        
        return h
    
    def forward(self, tree):
        """
        前向传播
        
        参数:
            tree: 输入树结构
            
        返回:
            类别预测概率
        """
        # 自顶向下遍历树
        node_hidden = {}
        leaf_hidden = []
        
        # 前序遍历树（自顶向下）
        for node_id in tree.pre_order():
            node = tree.get_node(node_id)
            
            if node_id == tree.root:
                # 根节点
                root_data = node.data
                root_emb = self.embedding(root_data) if isinstance(root_data, torch.LongTensor) else root_data
                node_hidden[node_id] = root_emb
            else:
                # 非根节点，使用父节点信息
                parent_id = tree.get_parent(node_id)
                parent_h = node_hidden[parent_id]
                
                node_hidden[node_id] = self.node_forward(node.data, parent_h)
                
                # 收集叶节点隐藏状态
                if node.is_leaf():
                    leaf_hidden.append(node_hidden[node_id])
        
        # 如果没有叶节点，使用根节点
        if not leaf_hidden:
            leaf_hidden.append(node_hidden[tree.root])
            
        # 最大池化获取最终表示
        leaf_hidden_tensor = torch.stack(leaf_hidden)
        pooled_hidden, _ = torch.max(leaf_hidden_tensor, dim=0)
        pooled_hidden = self.dropout(pooled_hidden)
        
        # 输出层
        logits = self.output_layer(pooled_hidden)
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