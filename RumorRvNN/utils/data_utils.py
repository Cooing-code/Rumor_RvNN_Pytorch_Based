import torch
import numpy as np
import os
import json
from collections import defaultdict

class TreeNode:
    """树节点类"""
    def __init__(self, node_id, data=None):
        self.id = node_id
        self.data = data
        self.parent = None
        self.children = []
        
    def is_leaf(self):
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def add_child(self, child_id):
        """添加子节点"""
        self.children.append(child_id)
        
    def __repr__(self):
        return f"TreeNode(id={self.id}, children={len(self.children)})"


class Tree:
    """树结构类"""
    def __init__(self, root_id=None):
        self.nodes = {}
        self.root = root_id
        
    def add_node(self, node_id, data=None):
        """添加节点"""
        if node_id not in self.nodes:
            self.nodes[node_id] = TreeNode(node_id, data)
        else:
            self.nodes[node_id].data = data
        return self.nodes[node_id]
    
    def get_node(self, node_id):
        """获取节点"""
        return self.nodes.get(node_id)
    
    def add_edge(self, parent_id, child_id):
        """添加边"""
        if parent_id not in self.nodes:
            self.add_node(parent_id)
        if child_id not in self.nodes:
            self.add_node(child_id)
            
        self.nodes[parent_id].add_child(child_id)
        self.nodes[child_id].parent = parent_id
        
        # 如果没有设置根节点，则将没有父节点的节点设为根节点
        if self.root is None and self.nodes[parent_id].parent is None:
            self.root = parent_id
    
    def get_parent(self, node_id):
        """获取父节点ID"""
        node = self.get_node(node_id)
        return node.parent if node else None
    
    def post_order(self):
        """后序遍历（自底向上）"""
        result = []
        
        def _post_order(node_id):
            node = self.get_node(node_id)
            for child_id in node.children:
                _post_order(child_id)
            result.append(node_id)
            
        if self.root:
            _post_order(self.root)
        return result
    
    def pre_order(self):
        """前序遍历（自顶向下）"""
        result = []
        
        def _pre_order(node_id):
            result.append(node_id)
            node = self.get_node(node_id)
            for child_id in node.children:
                _pre_order(child_id)
                
        if self.root:
            _pre_order(self.root)
        return result
    
    def __len__(self):
        """树的节点数量"""
        return len(self.nodes)


def load_vocab(vocab_path, max_size=5000):
    """
    加载词汇表
    
    参数:
        vocab_path (str): 词汇表文件路径
        max_size (int): 最大词汇表大小
        
    返回:
        word2idx (dict): 词到索引的映射
        idx2word (dict): 索引到词的映射
    """
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_size - 2:  # 减去PAD和UNK
                break
            word = line.strip().split()[0]
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
    
    return word2idx, idx2word


def text_to_tensor(text, word2idx, max_len=None):
    """
    将文本转换为张量
    
    参数:
        text (str): 输入文本
        word2idx (dict): 词到索引的映射
        max_len (int): 最大长度
        
    返回:
        tensor: 文本张量
    """
    words = text.strip().split()
    if max_len:
        words = words[:max_len]
    
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    
    if max_len:
        # 填充到最大长度
        indices = indices + [word2idx['<PAD>']] * (max_len - len(indices))
    
    return torch.tensor(indices, dtype=torch.long)


def build_tree_from_json(json_data, word2idx):
    """
    从JSON数据构建树
    
    参数:
        json_data (dict): JSON格式的树数据
        word2idx (dict): 词到索引的映射
        
    返回:
        tree: 构建的树结构
    """
    tree = Tree()
    
    def process_node(node_data, parent_id=None):
        node_id = node_data['id']
        text = node_data.get('text', '')
        
        # 将文本转换为张量
        text_tensor = text_to_tensor(text, word2idx)
        
        # 添加节点
        tree.add_node(node_id, text_tensor)
        
        # 添加边
        if parent_id is not None:
            tree.add_edge(parent_id, node_id)
        elif tree.root is None:
            tree.root = node_id
        
        # 处理子节点
        for child in node_data.get('children', []):
            process_node(child, node_id)
    
    # 处理根节点
    process_node(json_data)
    
    return tree


def load_data(data_path, label_path, vocab_path, max_vocab_size=5000):
    """
    加载数据
    
    参数:
        data_path (str): 数据文件路径
        label_path (str): 标签文件路径
        vocab_path (str): 词汇表文件路径
        max_vocab_size (int): 最大词汇表大小
        
    返回:
        data_list: 数据列表
        labels: 标签列表
        word2idx: 词到索引的映射
    """
    # 加载词汇表
    word2idx, _ = load_vocab(vocab_path, max_vocab_size)
    
    # 加载标签
    labels_dict = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                event_id, label = parts[0], parts[1]
                # 转换标签为one-hot向量
                if label in ['true', 'non-rumour']:
                    labels_dict[event_id] = torch.tensor([1, 0], dtype=torch.float)
                else:  # 'false', 'rumour'
                    labels_dict[event_id] = torch.tensor([0, 1], dtype=torch.float)
    
    # 加载数据
    data_list = []
    labels = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            event_id = line.strip()
            if event_id in labels_dict:
                # 加载事件数据
                event_file = os.path.join(os.path.dirname(data_path), 'trees', f"{event_id}.json")
                if os.path.exists(event_file):
                    with open(event_file, 'r', encoding='utf-8') as ef:
                        tree_data = json.load(ef)
                        tree = build_tree_from_json(tree_data, word2idx)
                        data_list.append(tree)
                        labels.append(labels_dict[event_id])
    
    return data_list, labels, word2idx


def batch_trees(trees, labels, batch_size=32):
    """
    将树数据分批
    
    参数:
        trees: 树列表
        labels: 标签列表
        batch_size: 批大小
        
    返回:
        批次列表
    """
    indices = list(range(len(trees)))
    np.random.shuffle(indices)
    
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_trees = [trees[idx] for idx in batch_indices]
        batch_labels = [labels[idx] for idx in batch_indices]
        batches.append((batch_trees, torch.stack(batch_labels)))
    
    return batches


def matrix_to_tensor(matrix_data, vocab_size):
    """
    将矩阵数据转换为张量
    
    参数:
        matrix_data: 矩阵数据
        vocab_size: 词汇表大小
        
    返回:
        tensor: 转换后的张量
    """
    batch_size = len(matrix_data)
    seq_len = max(len(seq) for seq in matrix_data)
    
    # 创建全零张量
    tensor = torch.zeros(batch_size, seq_len, vocab_size)
    
    for i, seq in enumerate(matrix_data):
        for j, vec in enumerate(seq):
            tensor[i, j] = torch.tensor(vec)
    
    return tensor 