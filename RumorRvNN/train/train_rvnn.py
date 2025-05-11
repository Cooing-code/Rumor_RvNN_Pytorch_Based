import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import sys
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bu_rvnn import BURvNN
from models.td_rvnn import TDRvNN
from utils.data_utils import load_data, batch_trees
from utils.eval_utils import evaluate_model, print_metrics

def train_rvnn(args):
    """
    训练RvNN模型
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_data, train_labels, word2idx = load_data(
        args.train_path, 
        args.label_path, 
        args.vocab_path, 
        args.vocab_size
    )
    
    val_data, val_labels, _ = load_data(
        args.val_path, 
        args.label_path, 
        args.vocab_path, 
        args.vocab_size
    )
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"词汇表大小: {len(word2idx)}")
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    if args.model_type == 'bu_rvnn':
        model = BURvNN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
    elif args.model_type == 'td_rvnn':
        model = TDRvNN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model = model.to(device)
    print(f"模型参数数量: {model.count_parameters()}")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 训练循环
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_metrics_history = []
    
    print("开始训练...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 创建批次
        train_batches = batch_trees(train_data, train_labels, args.batch_size)
        
        # 进度条
        train_iterator = tqdm(train_batches, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_trees, batch_labels in train_iterator:
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # 处理每棵树
            for i, tree in enumerate(batch_trees):
                # 将树的数据移到设备上
                for node_id, node in tree.nodes.items():
                    if node.data is not None:
                        node.data = node.data.to(device)
                
                # 前向传播
                prediction = model(tree)
                
                # 计算损失
                loss = criterion(prediction.unsqueeze(0), batch_labels[i].unsqueeze(0).to(device))
                batch_loss += loss
            
            # 平均批次损失
            batch_loss /= len(batch_trees)
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            epoch_loss += batch_loss.item()
            batch_count += 1
            
            # 更新进度条
            train_iterator.set_postfix(loss=batch_loss.item())
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        print(f"验证中...")
        val_batches = batch_trees(val_data, val_labels, args.batch_size)
        val_metrics = evaluate_model(model, val_batches, device)
        val_metrics_history.append(val_metrics)
        
        # 打印指标
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证指标:")
        print_metrics(val_metrics)
        
        # 更新学习率
        scheduler.step(val_metrics['accuracy'])
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            
            # 保存模型
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            
            model_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_best.pt")
            model.save_model(model_path)
            print(f"  保存最佳模型到 {model_path}")
        
        print(f"  当前最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return model, train_losses, val_metrics_history

def main():
    parser = argparse.ArgumentParser(description='训练RvNN模型')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, required=True, help='训练集路径')
    parser.add_argument('--val_path', type=str, required=True, help='验证集路径')
    parser.add_argument('--label_path', type=str, required=True, help='标签文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--vocab_size', type=int, default=5000, help='词汇表大小')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, choices=['bu_rvnn', 'td_rvnn'], default='bu_rvnn', help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=100, help='隐藏层大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    
    args = parser.parse_args()
    
    # 训练模型
    train_rvnn(args)

if __name__ == '__main__':
    main() 