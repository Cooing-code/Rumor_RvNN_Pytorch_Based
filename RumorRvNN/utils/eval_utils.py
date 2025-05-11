import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_classification(predictions, labels):
    """
    评估分类性能
    
    参数:
        predictions: 预测结果 [batch_size, num_classes]
        labels: 真实标签 [batch_size, num_classes]
        
    返回:
        metrics: 评估指标字典
    """
    # 转换为类别索引
    pred_indices = torch.argmax(predictions, dim=1).cpu().numpy()
    label_indices = torch.argmax(labels, dim=1).cpu().numpy()
    
    # 计算指标
    accuracy = accuracy_score(label_indices, pred_indices)
    
    # 计算每个类别的精确率、召回率和F1
    precision = precision_score(label_indices, pred_indices, average=None)
    recall = recall_score(label_indices, pred_indices, average=None)
    f1 = f1_score(label_indices, pred_indices, average=None)
    
    # 计算混淆矩阵
    cm = confusion_matrix(label_indices, pred_indices)
    
    # 组织结果
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics

def print_metrics(metrics, class_names=None):
    """
    打印评估指标
    
    参数:
        metrics: 评估指标字典
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ['非谣言', '谣言']
    
    print(f"准确率: {metrics['accuracy']:.4f}")
    print("\n类别指标:")
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    精确率: {metrics['precision'][i]:.4f}")
        print(f"    召回率: {metrics['recall'][i]:.4f}")
        print(f"    F1分数: {metrics['f1'][i]:.4f}")
    
    print("\n混淆矩阵:")
    cm = metrics['confusion_matrix']
    cm_str = ""
    for i in range(len(class_names)):
        cm_str += f"{class_names[i]}\t"
    print(f"\t{cm_str}")
    
    for i, row in enumerate(cm):
        row_str = ""
        for j in row:
            row_str += f"{j}\t"
        print(f"{class_names[i]}\t{row_str}")

def evaluate_model(model, data_loader, device):
    """
    评估模型
    
    参数:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        
    返回:
        metrics: 评估指标字典
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_trees, batch_labels in data_loader:
            batch_predictions = []
            
            for tree in batch_trees:
                # 将树的数据移到设备上
                for node_id, node in tree.nodes.items():
                    if node.data is not None:
                        node.data = node.data.to(device)
                
                # 前向传播
                prediction = model(tree)
                batch_predictions.append(prediction)
            
            # 收集预测和标签
            batch_predictions = torch.stack(batch_predictions)
            all_predictions.append(batch_predictions)
            all_labels.append(batch_labels.to(device))
    
    # 合并所有批次的结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    metrics = evaluate_classification(all_predictions, all_labels)
    
    return metrics

def compute_rmse(predictions, labels):
    """
    计算均方根误差
    
    参数:
        predictions: 预测结果 [batch_size, num_classes]
        labels: 真实标签 [batch_size, num_classes]
        
    返回:
        rmse: 均方根误差
    """
    mse = torch.mean((predictions - labels) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item() 