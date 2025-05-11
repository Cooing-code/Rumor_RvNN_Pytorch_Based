import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import sys
import argparse
import random
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gan_model import RumorGAN
from utils.data_utils import load_data, batch_trees, matrix_to_tensor
from utils.eval_utils import evaluate_model, print_metrics, compute_rmse

def split_data_by_label(data, labels):
    """
    按标签分割数据
    
    参数:
        data: 数据列表
        labels: 标签列表
        
    返回:
        non_rumor_data: 非谣言数据
        non_rumor_labels: 非谣言标签
        rumor_data: 谣言数据
        rumor_labels: 谣言标签
    """
    non_rumor_indices = []
    rumor_indices = []
    
    for i, label in enumerate(labels):
        if label[0] > label[1]:  # 非谣言 [1, 0]
            non_rumor_indices.append(i)
        else:  # 谣言 [0, 1]
            rumor_indices.append(i)
    
    non_rumor_data = [data[i] for i in non_rumor_indices]
    non_rumor_labels = [labels[i] for i in non_rumor_indices]
    
    rumor_data = [data[i] for i in rumor_indices]
    rumor_labels = [labels[i] for i in rumor_indices]
    
    return non_rumor_data, non_rumor_labels, rumor_data, rumor_labels

def pre_train_discriminator(model, train_data, train_labels, val_data, val_labels, args):
    """
    预训练判别器
    
    参数:
        model: GAN模型
        train_data: 训练数据
        train_labels: 训练标签
        val_data: 验证数据
        val_labels: 验证标签
        args: 命令行参数
        
    返回:
        预训练后的模型
    """
    print("预训练判别器...")
    device = next(model.parameters()).device
    
    # 定义优化器
    optimizer = optim.Adam(model.discriminator.parameters(), lr=args.lr_d)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.pre_epochs_d):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 创建批次
        train_batches = batch_trees(train_data, train_labels, args.batch_size)
        
        # 进度条
        train_iterator = tqdm(train_batches, desc=f"Epoch {epoch+1}/{args.pre_epochs_d} [Pre-Train D]")
        
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
                prediction = model(tree, None, 'discriminate')
                
                # 计算损失
                loss = criterion(prediction.unsqueeze(0), batch_labels[i].unsqueeze(0).to(device))
                batch_loss += loss
            
            # 平均批次损失
            batch_loss /= len(batch_trees)
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), args.clip)
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            epoch_loss += batch_loss.item()
            batch_count += 1
            
            # 更新进度条
            train_iterator.set_postfix(loss=batch_loss.item())
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / batch_count
        
        # 验证阶段
        if (epoch + 1) % args.eval_interval == 0:
            print(f"验证中...")
            val_batches = batch_trees(val_data, val_labels, args.batch_size)
            val_metrics = evaluate_model(model, val_batches, device)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{args.pre_epochs_d}:")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  验证指标:")
            print_metrics(val_metrics)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                
                # 保存模型
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                
                model_path = os.path.join(args.checkpoint_dir, "gan_discriminator_best.pt")
                torch.save({
                    'model_state_dict': model.discriminator.state_dict(),
                }, model_path)
                print(f"  保存最佳判别器到 {model_path}")
            
            print(f"  当前最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    print(f"判别器预训练完成! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return model

def pre_train_generator(model, data, labels, seq_lengths, target_labels, is_rumor, args):
    """
    预训练生成器
    
    参数:
        model: GAN模型
        data: 训练数据
        labels: 训练标签
        seq_lengths: 序列长度
        target_labels: 目标标签
        is_rumor: 是否为谣言样本
        args: 命令行参数
        
    返回:
        预训练后的模型
    """
    generator_type = "RN" if is_rumor else "NR"
    print(f"预训练生成器 {generator_type}...")
    device = next(model.parameters()).device
    
    # 定义优化器
    if is_rumor:
        optimizer = optim.Adam(model.generator_rn.parameters(), lr=args.lr_g)
    else:
        optimizer = optim.Adam(model.generator_nr.parameters(), lr=args.lr_g)
    
    # 训练循环
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.pre_epochs_g):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 创建批次
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        # 进度条
        train_iterator = tqdm(range(0, len(indices), args.batch_size), 
                             desc=f"Epoch {epoch+1}/{args.pre_epochs_g} [Pre-Train G-{generator_type}]")
        
        for i in train_iterator:
            batch_indices = indices[i:i+args.batch_size]
            batch_data = [data[idx] for idx in batch_indices]
            batch_seq_lengths = [seq_lengths[idx] for idx in batch_indices]
            batch_target_labels = [target_labels[idx] for idx in batch_indices]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # 处理每个样本
            for j, (x, seq_len, target) in enumerate(zip(batch_data, batch_seq_lengths, batch_target_labels)):
                # 将数据移到设备上
                x = x.to(device)
                target = target.to(device)
                
                # 前向传播
                if is_rumor:
                    # 谣言->非谣言
                    x_gen = model(x, seq_len, 'generate_rn')
                    pred = model.discriminator(x_gen)
                else:
                    # 非谣言->谣言
                    x_gen = model(x, seq_len, 'generate_nr')
                    pred = model.discriminator(x_gen)
                
                # 计算损失
                loss = torch.mean((pred - target) ** 2)
                batch_loss += loss
            
            # 平均批次损失
            batch_loss /= len(batch_indices)
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪
            if is_rumor:
                torch.nn.utils.clip_grad_norm_(model.generator_rn.parameters(), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.generator_nr.parameters(), args.clip)
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            epoch_loss += batch_loss.item()
            batch_count += 1
            
            # 更新进度条
            train_iterator.set_postfix(loss=batch_loss.item())
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / batch_count
        
        print(f"Epoch {epoch+1}/{args.pre_epochs_g}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        
        # 保存最佳模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_epoch = epoch + 1
            
            # 保存模型
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            
            model_path = os.path.join(args.checkpoint_dir, f"gan_generator_{generator_type}_best.pt")
            if is_rumor:
                torch.save({
                    'model_state_dict': model.generator_rn.state_dict(),
                }, model_path)
            else:
                torch.save({
                    'model_state_dict': model.generator_nr.state_dict(),
                }, model_path)
            print(f"  保存最佳生成器到 {model_path}")
        
        print(f"  当前最佳损失: {best_loss:.4f} (Epoch {best_epoch})")
    
    print(f"生成器 {generator_type} 预训练完成! 最佳损失: {best_loss:.4f} (Epoch {best_epoch})")
    
    return model

def train_gan(args):
    """
    训练GAN模型
    
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
    
    # 分割数据
    non_rumor_data, non_rumor_labels, rumor_data, rumor_labels = split_data_by_label(train_data, train_labels)
    print(f"非谣言样本数: {len(non_rumor_data)}")
    print(f"谣言样本数: {len(rumor_data)}")
    
    # 创建模型
    print("创建GAN模型...")
    model = RumorGAN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
    model = model.to(device)
    print(f"模型参数数量: {model.count_parameters()}")
    
    # 预处理序列长度
    train_seq_lengths = [len(tree.nodes) for tree in train_data]
    non_rumor_seq_lengths = [len(tree.nodes) for tree in non_rumor_data]
    rumor_seq_lengths = [len(tree.nodes) for tree in rumor_data]
    
    # 创建目标标签
    non_rumor_target_labels = [torch.tensor([0, 1], dtype=torch.float) for _ in non_rumor_labels]  # 非谣言->谣言
    rumor_target_labels = [torch.tensor([1, 0], dtype=torch.float) for _ in rumor_labels]  # 谣言->非谣言
    
    # 预训练判别器
    if args.pre_train_d:
        model = pre_train_discriminator(model, train_data, train_labels, val_data, val_labels, args)
    
    # 预训练生成器
    if args.pre_train_g:
        # 预训练非谣言->谣言生成器
        model = pre_train_generator(model, non_rumor_data, non_rumor_labels, 
                                   non_rumor_seq_lengths, non_rumor_target_labels, False, args)
        
        # 预训练谣言->非谣言生成器
        model = pre_train_generator(model, rumor_data, rumor_labels, 
                                   rumor_seq_lengths, rumor_target_labels, True, args)
    
    # 定义优化器
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d)
    optimizer_g = optim.Adam([
        {'params': model.generator_nr.parameters()},
        {'params': model.generator_rn.parameters()}
    ], lr=args.lr_g)
    
    # 训练循环
    best_val_acc = 0.0
    best_epoch = 0
    
    print("开始GAN训练...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        d_losses = []
        g_losses = []
        
        # 创建批次索引
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        # 进度条
        train_iterator = tqdm(range(0, len(indices), args.batch_size), 
                             desc=f"Epoch {epoch+1}/{args.epochs} [GAN Train]")
        
        for i in train_iterator:
            batch_indices = indices[i:i+args.batch_size]
            batch_data = [train_data[idx] for idx in batch_indices]
            batch_labels = [train_labels[idx] for idx in batch_indices]
            batch_seq_lengths = [train_seq_lengths[idx] for idx in batch_indices]
            
            # 分割批次为谣言和非谣言
            batch_non_rumor_indices = []
            batch_rumor_indices = []
            
            for j, label in enumerate(batch_labels):
                if label[0] > label[1]:  # 非谣言 [1, 0]
                    batch_non_rumor_indices.append(j)
                else:  # 谣言 [0, 1]
                    batch_rumor_indices.append(j)
            
            # 训练判别器
            for _ in range(args.d_steps):
                optimizer_d.zero_grad()
                d_loss = 0.0
                
                # 真实样本损失
                for j, tree in enumerate(batch_data):
                    # 将树的数据移到设备上
                    for node_id, node in tree.nodes.items():
                        if node.data is not None:
                            node.data = node.data.to(device)
                    
                    # 前向传播
                    pred_real = model(tree, None, 'discriminate')
                    loss_real = torch.mean((pred_real - batch_labels[j].to(device)) ** 2)
                    d_loss += loss_real / len(batch_data)
                
                # 生成样本损失
                for j in batch_non_rumor_indices:
                    tree = batch_data[j]
                    seq_len = batch_seq_lengths[j]
                    
                    # 非谣言->谣言
                    x_nr = model(tree, seq_len, 'generate_nr')
                    pred_fake = model.discriminator(x_nr)
                    target_fake = torch.tensor([0, 1], dtype=torch.float).to(device)  # 谣言标签
                    loss_fake = torch.mean((pred_fake - target_fake) ** 2)
                    d_loss += loss_fake / len(batch_indices)
                
                for j in batch_rumor_indices:
                    tree = batch_data[j]
                    seq_len = batch_seq_lengths[j]
                    
                    # 谣言->非谣言
                    x_rn = model(tree, seq_len, 'generate_rn')
                    pred_fake = model.discriminator(x_rn)
                    target_fake = torch.tensor([1, 0], dtype=torch.float).to(device)  # 非谣言标签
                    loss_fake = torch.mean((pred_fake - target_fake) ** 2)
                    d_loss += loss_fake / len(batch_indices)
                
                # 反向传播
                d_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), args.clip)
                
                # 更新参数
                optimizer_d.step()
                
                d_losses.append(d_loss.item())
            
            # 训练生成器
            for _ in range(args.g_steps):
                optimizer_g.zero_grad()
                g_loss = 0.0
                
                # 非谣言->谣言->非谣言
                for j in batch_non_rumor_indices:
                    tree = batch_data[j]
                    seq_len = batch_seq_lengths[j]
                    
                    # 非谣言->谣言
                    x_nr = model(tree, seq_len, 'generate_nr')
                    pred_nr = model.discriminator(x_nr)
                    target_nr = torch.tensor([0, 1], dtype=torch.float).to(device)  # 谣言标签
                    loss_nr = torch.mean((pred_nr - target_nr) ** 2)
                    
                    # 谣言->非谣言
                    x_nrn = model.generator_rn(x_nr, seq_len)
                    
                    # 循环一致性损失
                    for node_id, node in tree.nodes.items():
                        if node.data is not None:
                            node_data = node.data.to(device)
                            loss_cycle = compute_rmse(x_nrn, node_data)
                            g_loss += (loss_nr + args.cycle_weight * loss_cycle) / len(batch_non_rumor_indices)
                
                # 谣言->非谣言->谣言
                for j in batch_rumor_indices:
                    tree = batch_data[j]
                    seq_len = batch_seq_lengths[j]
                    
                    # 谣言->非谣言
                    x_rn = model(tree, seq_len, 'generate_rn')
                    pred_rn = model.discriminator(x_rn)
                    target_rn = torch.tensor([1, 0], dtype=torch.float).to(device)  # 非谣言标签
                    loss_rn = torch.mean((pred_rn - target_rn) ** 2)
                    
                    # 非谣言->谣言
                    x_rnr = model.generator_nr(x_rn, seq_len)
                    
                    # 循环一致性损失
                    for node_id, node in tree.nodes.items():
                        if node.data is not None:
                            node_data = node.data.to(device)
                            loss_cycle = compute_rmse(x_rnr, node_data)
                            g_loss += (loss_rn + args.cycle_weight * loss_cycle) / len(batch_rumor_indices)
                
                # 反向传播
                g_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.generator_nr.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(model.generator_rn.parameters(), args.clip)
                
                # 更新参数
                optimizer_g.step()
                
                g_losses.append(g_loss.item())
            
            # 更新进度条
            train_iterator.set_postfix(d_loss=d_losses[-1], g_loss=g_losses[-1])
        
        # 计算平均损失
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        
        # 验证阶段
        if (epoch + 1) % args.eval_interval == 0:
            print(f"验证中...")
            val_batches = batch_trees(val_data, val_labels, args.batch_size)
            val_metrics = evaluate_model(model, val_batches, device)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  判别器损失: {avg_d_loss:.4f}")
            print(f"  生成器损失: {avg_g_loss:.4f}")
            print(f"  验证指标:")
            print_metrics(val_metrics)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                
                # 保存模型
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                
                model_path = os.path.join(args.checkpoint_dir, "gan_model_best.pt")
                model.save_model(model_path)
                print(f"  保存最佳模型到 {model_path}")
            
            print(f"  当前最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='训练GAN模型')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, required=True, help='训练集路径')
    parser.add_argument('--val_path', type=str, required=True, help='验证集路径')
    parser.add_argument('--label_path', type=str, required=True, help='标签文件路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件路径')
    parser.add_argument('--vocab_size', type=int, default=5000, help='词汇表大小')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=100, help='隐藏层大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    
    # 预训练参数
    parser.add_argument('--pre_train_d', action='store_true', help='是否预训练判别器')
    parser.add_argument('--pre_train_g', action='store_true', help='是否预训练生成器')
    parser.add_argument('--pre_epochs_d', type=int, default=50, help='判别器预训练轮数')
    parser.add_argument('--pre_epochs_g', type=int, default=30, help='生成器预训练轮数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr_d', type=float, default=0.001, help='判别器学习率')
    parser.add_argument('--lr_g', type=float, default=0.001, help='生成器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--d_steps', type=int, default=1, help='每批次判别器更新步数')
    parser.add_argument('--g_steps', type=int, default=1, help='每批次生成器更新步数')
    parser.add_argument('--cycle_weight', type=float, default=10.0, help='循环一致性损失权重')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--eval_interval', type=int, default=5, help='评估间隔')
    
    args = parser.parse_args()
    
    # 训练模型
    train_gan(args)

if __name__ == '__main__':
    main() 