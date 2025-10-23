"""
银行卡号码识别CRNN模型训练脚本
使用CTC损失函数进行端到端的序列识别训练
"""

import os
import csv
import torch
from config import *
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
from evaluate import evaluate
from torch.utils.data import DataLoader
from .dataset import CardDataset, cardnumber_collate_fn


def train_batch(crnn, data, optimizer, criterion, device):
    """
    训练单个batch的数据
    
    Args:
        crnn: CRNN模型实例
        data: 包含(images, targets, target_lengths)的训练数据
        optimizer: 优化器
        criterion: CTC损失函数
        device: 计算设备(cpu/gpu)
    
    Returns:
        float: 当前batch的平均损失值
    """
    # 设置模型为训练模式
    crnn.train()
    
    # 将数据移动到指定设备
    images, targets, target_lengths = [d.to(device) for d in data]

    # 前向传播：通过CRNN模型获取预测结果
    logits = crnn(images)
    # 对输出进行log_softmax处理，用于CTC损失计算
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    # 获取batch大小
    batch_size = images.size(0)
    # 创建输入序列长度张量(所有样本的序列长度相同)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    # 展平目标长度张量
    target_lengths = torch.flatten(target_lengths)

    # 计算CTC损失
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新模型参数
    
    return loss.item()  # 返回损失值

def main():
    """
    主训练函数
    负责初始化模型、数据加载器、优化器等，并执行完整的训练流程
    """
    # ==================== 设备配置 ====================
    # 自动选择可用的计算设备(优先使用GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # ==================== 数据加载 ====================
    # 创建训练数据集
    train_dataset = CardDataset(
        image_dir=data_dir+'/train', 
        mode='train',
        img_height=img_height, 
        img_width=img_width
    )
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,                    # 每个epoch随机打乱数据
        num_workers=num_workers,         # 数据加载的并行进程数
        collate_fn=cardnumber_collate_fn # 自定义的批处理函数
    )
    
    # ==================== 模型初始化 ====================
    # 计算类别数量(字符数 + 1个空白符)
    num_class = len(CardDataset.LABEL2CHAR) + 1
    
    # 创建CRNN模型实例
    crnn = CRNN(
        img_channel=1,                    # 输入图像通道数(灰度图)
        img_height=img_height,            # 图像高度
        img_width=img_width,              # 图像宽度
        num_class=num_class,              # 输出类别数
        map_to_seq_hidden=map_to_seq_hidden,  # CNN到RNN的映射层隐藏单元数
        rnn_hidden=rnn_hidden,            # RNN隐藏单元数
        leaky_relu=leaky_relu,            # 是否使用LeakyReLU激活函数
        backbone=backbone                 # 骨干网络类型(LCNet/ResNet/MobileNet)
    )
    print('CRNN模型结构:')
    print(crnn)

    # 如果指定了预训练模型路径，则加载预训练权重
    if reload_checkpoint:
        print(f'加载预训练模型: {reload_checkpoint}')
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    
    # 将模型移动到指定设备
    crnn.to(device)
    
    # ==================== 优化器和损失函数 ====================
    # 根据配置选择优化器
    if optim_config == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=lr)
        print(f'使用Adam优化器，学习率: {lr}')
    elif optim_config == 'sgd':
        optimizer = optim.SGD(crnn.parameters(), lr=lr)
        print(f'使用SGD优化器，学习率: {lr}')
    elif optim_config == 'rmsprop':
        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
        print(f'使用RMSprop优化器，学习率: {lr}')
    
    # 创建CTC损失函数
    criterion = CTCLoss(reduction='sum')  # 使用sum reduction计算总损失
    criterion.to(device)

    # ==================== 训练状态初始化 ====================
    best_accuracy = -1      # 记录最佳准确率
    best_epoch = None       # 记录最佳准确率对应的epoch
    data = []               # 存储训练过程中的指标数据
    
    # ==================== 创建保存目录 ====================
    # 创建主保存目录
    if not os.path.exists('./runs/recognition'):
        os.mkdir('./runs/recognition')
    
    # 自动生成新的运行编号，避免覆盖之前的训练结果
    run = 1
    while os.path.exists('./runs/recognition/run'+str(run)):
        run += 1
    
    # 创建当前运行的保存目录
    os.mkdir('./runs/recognition/run'+str(run))
    os.mkdir('./runs/recognition/run'+str(run)+'/checkpoints')  # 模型检查点保存目录
    save_path = './runs/recognition/run'+str(run)
    print(f'训练结果将保存到: {save_path}')

    # ==================== 开始训练循环 ====================
    print(f'开始训练，总epoch数: {epochs}')
    for epoch in range(1, epochs + 1):
        print(f'\n========== Epoch {epoch}/{epochs} ==========')
        
        # 初始化当前epoch的统计变量
        total_train_loss = 0.      # 累计训练损失
        total_train_count = 0      # 累计训练样本数
        index = 1                  # 当前batch索引
        length = len(train_loader) # 总batch数
        
        # ==================== 训练一个epoch ====================
        print('开始训练...')
        for train_data in train_loader: 
            # 训练当前batch
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            
            # 统计损失和样本数
            train_size = train_data[0].size(0)  # 当前batch的样本数
            total_train_loss += loss
            total_train_count += train_size
            
            # 显示当前batch的训练进度和损失
            print(f'训练进度 [{index:3d}/{length:3d}] - 平均损失: {loss/train_size:.6f}', end="\r")
            index += 1
        
        # 计算并显示当前epoch的平均训练损失
        avg_train_loss = total_train_loss / total_train_count
        print(f'\nEpoch {epoch} 平均训练损失: {avg_train_loss:.6f}')
        
        # ==================== 准备保存训练数据 ====================
        # 初始化当前epoch的指标记录
        temp = []
        temp.append(epoch)                    # epoch编号
        temp.append(avg_train_loss)           # 平均训练损失

        # ==================== 保存当前模型 ====================
        # 保存当前epoch的模型权重(用于恢复训练)
        torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn last.pt')
        print(f'已保存当前模型到: {save_path}/checkpoints/crnn last.pt')
        
        # ==================== 模型评估 ====================
        print('开始评估模型性能...')
        # 在验证集和测试集上评估模型
        test_loss, accuracy, val_loss, val_accu = evaluate(crnn, data_dir)
        
        # 记录评估指标
        temp.append(val_loss)    # 验证集损失
        temp.append(val_accu)    # 验证集准确率
        temp.append(test_loss)   # 测试集损失
        temp.append(accuracy)    # 测试集准确率
        data.append(temp)        # 添加到总数据中
        
        # 显示评估结果
        print('========== 评估结果 ==========')
        print(f'验证集损失: {val_loss:.6f}')
        print(f'验证集准确率: {val_accu:.4f}')
        print(f'测试集损失: {test_loss:.6f}')
        print(f'测试集准确率: {accuracy:.4f}')

        # ==================== 保存训练记录 ====================
        # 将训练指标保存到CSV文件中
        with open(save_path + '/results.csv', 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss', 'val_accu', 'test_loss', 'accuracy'])
            writer.writerows(data)
        print(f'训练记录已保存到: {save_path}/results.csv')
        
        # ==================== 模型保存策略 ====================
        # 如果当前模型在测试集上的准确率超过历史最佳，则保存为最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn best.pt')
            print(f'🎉 发现更好的模型! 准确率: {accuracy:.4f}')
            print(f'最佳模型已保存到: {save_path}/checkpoints/crnn best.pt')
        else:
            print(f'当前准确率: {accuracy:.4f}, 历史最佳: {best_accuracy:.4f}')
        
        # ==================== 早停策略 ====================
        # 如果连续多个epoch没有改善，则提前停止训练
        if epoch - best_epoch > early_stop:
            print(f'⏹️  早停触发! 连续 {early_stop} 个epoch没有改善')
            print(f'最佳epoch: {best_epoch}, 当前epoch: {epoch}')
            break

    # ==================== 训练完成 ====================
    print('\n========== 训练完成 ==========')
    print(f'最佳epoch: {best_epoch}')
    print(f'最佳准确率: {best_accuracy:.4f}')
    print(f'训练结果保存在: {save_path}')


if __name__ == '__main__':
    """
    程序入口点
    当直接运行此脚本时，执行主训练函数
    """
    main()