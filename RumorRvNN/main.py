import argparse
import os
import sys
import torch

from models.bu_rvnn import BURvNN
from models.td_rvnn import TDRvNN
from models.gan_model import RumorGAN
from train.train_rvnn import train_rvnn
from train.train_gan import train_gan
from utils.data_utils import load_data, batch_trees
from utils.eval_utils import evaluate_model, print_metrics

def main():
    parser = argparse.ArgumentParser(description='基于树结构递归神经网络的谣言检测')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='运行模式')
    parser.add_argument('--model', type=str, choices=['bu_rvnn', 'td_rvnn', 'gan'], default='bu_rvnn', help='模型类型')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data', help='数据目录')
    parser.add_argument('--train_path', type=str, help='训练集路径')
    parser.add_argument('--val_path', type=str, help='验证集路径')
    parser.add_argument('--test_path', type=str, help='测试集路径')
    parser.add_argument('--label_path', type=str, help='标签文件路径')
    parser.add_argument('--vocab_path', type=str, help='词汇表文件路径')
    parser.add_argument('--vocab_size', type=int, default=5000, help='词汇表大小')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=100, help='隐藏层大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lr_d', type=float, default=0.001, help='判别器学习率')
    parser.add_argument('--lr_g', type=float, default=0.001, help='生成器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    
    # GAN特定参数
    parser.add_argument('--pre_train_d', action='store_true', help='是否预训练判别器')
    parser.add_argument('--pre_train_g', action='store_true', help='是否预训练生成器')
    parser.add_argument('--pre_epochs_d', type=int, default=50, help='判别器预训练轮数')
    parser.add_argument('--pre_epochs_g', type=int, default=30, help='生成器预训练轮数')
    parser.add_argument('--d_steps', type=int, default=1, help='每批次判别器更新步数')
    parser.add_argument('--g_steps', type=int, default=1, help='每批次生成器更新步数')
    parser.add_argument('--cycle_weight', type=float, default=10.0, help='循环一致性损失权重')
    parser.add_argument('--eval_interval', type=int, default=5, help='评估间隔')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置默认路径
    if args.train_path is None:
        args.train_path = os.path.join(args.data_path, 'train.txt')
    if args.val_path is None:
        args.val_path = os.path.join(args.data_path, 'val.txt')
    if args.test_path is None:
        args.test_path = os.path.join(args.data_path, 'test.txt')
    if args.label_path is None:
        args.label_path = os.path.join(args.data_path, 'labels.txt')
    if args.vocab_path is None:
        args.vocab_path = os.path.join(args.data_path, 'vocab.txt')
    
    # 训练模式
    if args.mode == 'train':
        if args.model == 'bu_rvnn' or args.model == 'td_rvnn':
            # 训练RvNN模型
            train_rvnn(args)
        elif args.model == 'gan':
            # 训练GAN模型
            train_gan(args)
    
    # 测试模式
    elif args.mode == 'test':
        # 加载数据
        print("加载测试数据...")
        test_data, test_labels, word2idx = load_data(
            args.test_path, 
            args.label_path, 
            args.vocab_path, 
            args.vocab_size
        )
        
        print(f"测试集大小: {len(test_data)}")
        print(f"词汇表大小: {len(word2idx)}")
        
        # 加载模型
        print(f"加载模型 {args.model}...")
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
        
        if args.model == 'bu_rvnn':
            model = BURvNN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
        elif args.model == 'td_rvnn':
            model = TDRvNN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
        elif args.model == 'gan':
            model = RumorGAN(len(word2idx), args.hidden_size, args.num_classes, args.dropout)
        
        model = model.to(device)
        model.load_model(args.checkpoint)
        
        # 评估模型
        print("评估模型...")
        test_batches = batch_trees(test_data, test_labels, args.batch_size)
        test_metrics = evaluate_model(model, test_batches, device)
        
        # 打印结果
        print("测试结果:")
        print_metrics(test_metrics)

if __name__ == '__main__':
    main() 