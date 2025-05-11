import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class Generator(nn.Module):
    """
    生成器模型，基于GRU的编码器-解码器结构
    """
    def __init__(self, vocab_size, hidden_size):
        """
        初始化生成器
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
        """
        super(Generator, self).__init__()
        
        # 嵌入层
        self.encoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embedding = nn.Linear(hidden_size, vocab_size)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # 编码器GRU
        self.encoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # 解码器GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, seq_len):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, vocab_size]
            seq_len: 序列长度
            
        返回:
            生成的序列 [batch_size, seq_len, vocab_size]
        """
        batch_size = x.size(0)
        device = x.device
        
        # 编码阶段
        x_emb = self.encoder_embedding(x) if x.dtype == torch.long else x @ self.encoder_embedding.weight.t()
        
        _, h_n = self.encoder_gru(x_emb)
        
        # 解码阶段
        # 初始输入
        decoder_input = F.relu(self.decoder_embedding(h_n.squeeze(0)) + self.decoder_bias)
        decoder_input = decoder_input.unsqueeze(1)  # [batch_size, 1, vocab_size]
        
        # 初始隐藏状态
        decoder_hidden = h_n
        
        # 存储所有输出
        outputs = [decoder_input.squeeze(1)]  # 第一个输出
        
        # 逐步解码
        for t in range(1, seq_len):
            decoder_emb = decoder_input @ self.encoder_embedding.weight.t()
            _, decoder_hidden = self.decoder_gru(decoder_emb, decoder_hidden)
            
            # 生成下一个词
            decoder_output = F.relu(self.decoder_embedding(decoder_hidden.squeeze(0)) + self.decoder_bias)
            decoder_input = decoder_output.unsqueeze(1)
            
            outputs.append(decoder_output)
        
        # 拼接所有输出 [batch_size, seq_len, vocab_size]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs


class Discriminator(nn.Module):
    """
    判别器模型，基于GRU的序列分类器
    """
    def __init__(self, vocab_size, hidden_size, num_classes):
        """
        初始化判别器
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
            num_classes (int): 类别数量
        """
        super(Discriminator, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, vocab_size]
            
        返回:
            类别预测概率 [batch_size, num_classes]
        """
        # 嵌入输入
        x_emb = self.embedding(x) if x.dtype == torch.long else x @ self.embedding.weight.t()
        
        # GRU处理序列
        _, h_n = self.gru(x_emb)
        
        # 输出层
        logits = self.output_layer(h_n.squeeze(0))
        return F.softmax(logits, dim=-1)
    
    def content_similarity(self, x1, x2):
        """
        计算内容相似度
        
        参数:
            x1: 第一个序列 [batch_size, seq_len, vocab_size]
            x2: 第二个序列 [batch_size, seq_len, vocab_size]
            
        返回:
            相似度得分
        """
        return torch.mean((x1 - x2) ** 2)


class RumorGAN(BaseModel):
    """
    谣言检测GAN模型，包含两个生成器和一个判别器
    """
    def __init__(self, vocab_size, hidden_size, num_classes, dropout=0.5):
        """
        初始化GAN模型
        
        参数:
            vocab_size (int): 词汇表大小
            hidden_size (int): 隐藏层大小
            num_classes (int): 类别数量
            dropout (float): Dropout概率
        """
        super(RumorGAN, self).__init__(vocab_size, hidden_size, num_classes)
        
        # 生成器NR (Non-Rumor to Rumor)
        self.generator_nr = Generator(vocab_size, hidden_size)
        
        # 生成器RN (Rumor to Non-Rumor)
        self.generator_rn = Generator(vocab_size, hidden_size)
        
        # 判别器
        self.discriminator = Discriminator(vocab_size, hidden_size, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, seq_len, mode='discriminate'):
        """
        前向传播
        
        参数:
            x: 输入序列
            seq_len: 序列长度
            mode: 运行模式 ('discriminate', 'generate_nr', 'generate_rn')
            
        返回:
            根据模式返回不同的输出
        """
        if mode == 'discriminate':
            return self.discriminator(x)
        elif mode == 'generate_nr':
            return self.generator_nr(x, seq_len)
        elif mode == 'generate_rn':
            return self.generator_rn(x, seq_len)
        elif mode == 'cycle_nr':
            # 非谣言->谣言->非谣言
            x_nr = self.generator_nr(x, seq_len)
            x_nrn = self.generator_rn(x_nr, seq_len)
            return x_nrn
        elif mode == 'cycle_rn':
            # 谣言->非谣言->谣言
            x_rn = self.generator_rn(x, seq_len)
            x_rnr = self.generator_nr(x_rn, seq_len)
            return x_rnr
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def compute_generator_loss(self, x, seq_len, target_labels, is_rumor=False):
        """
        计算生成器损失
        
        参数:
            x: 输入序列
            seq_len: 序列长度
            target_labels: 目标标签
            is_rumor: 是否为谣言样本
            
        返回:
            生成器损失
        """
        if is_rumor:
            # 谣言->非谣言
            x_rn = self.generator_rn(x, seq_len)
            # 判别器损失
            d_rn = self.discriminator(x_rn)
            loss_d = F.mse_loss(d_rn, target_labels)
            
            # 循环一致性损失
            x_rnr = self.generator_nr(x_rn, seq_len)
            loss_cycle = self.discriminator.content_similarity(x, x_rnr)
            
            return loss_d, loss_cycle, loss_d + loss_cycle
        else:
            # 非谣言->谣言
            x_nr = self.generator_nr(x, seq_len)
            # 判别器损失
            d_nr = self.discriminator(x_nr)
            loss_d = F.mse_loss(d_nr, target_labels)
            
            # 循环一致性损失
            x_nrn = self.generator_rn(x_nr, seq_len)
            loss_cycle = self.discriminator.content_similarity(x, x_nrn)
            
            return loss_d, loss_cycle, loss_d + loss_cycle
    
    def compute_discriminator_loss(self, x_real, real_labels, x_fake=None, fake_labels=None):
        """
        计算判别器损失
        
        参数:
            x_real: 真实序列
            real_labels: 真实标签
            x_fake: 生成序列
            fake_labels: 生成序列标签
            
        返回:
            判别器损失
        """
        # 真实样本损失
        d_real = self.discriminator(x_real)
        loss_real = F.cross_entropy(d_real, real_labels)
        
        # 生成样本损失
        if x_fake is not None and fake_labels is not None:
            d_fake = self.discriminator(x_fake)
            loss_fake = F.cross_entropy(d_fake, fake_labels)
            return loss_real + loss_fake
        
        return loss_real 