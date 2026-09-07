import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵工具
from utils import prepare_datasets, CustomLRScheduler
from config import *
from net import FileFormatAnalyzer
from torch.optim import AdamW   
import math
import torch.nn as nn
import time
from tqdm import tqdm
from config import labels
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    plt.figure(figsize=(40, 30))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        cbar_kws={'label': 'Percentage (%)'},
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=100
    )
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Confusion matrix saved to: {save_path}")

def load_checkpoint(model, optimizer, scheduler, ema, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])  
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, ema, epoch

def save_checkpoint(model, optimizer, scheduler, ema, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),       
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)

def train_epoch(model, optimizer, scheduler, train_loader, criterion, device, ema):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (data_ngram, target, raw_data) in enumerate(pbar):
        target, raw_data = target.to(device), raw_data.to(device)
        for k, v in data_ngram.items():
            data_ngram[k] = v.to(device)
        optimizer.zero_grad()
        output = model(data_ngram, raw_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        ema.update_parameters(model)

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        current_loss = loss.item()
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_true = []  # 存储所有真实标签
    all_pred = []  # 存储所有预测标签

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for data_ngram, target, raw_data in pbar:
            target, raw_data = target.to(device), raw_data.to(device)
            for k, v in data_ngram.items():
                data_ngram[k] = v.to(device)
            output = model(data_ngram, raw_data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_true.extend(target.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())            
            # 更新进度条
            current_loss = loss.item()
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    return total_loss / len(val_loader), 100. * correct / total, all_true, all_pred

def train(model, optimizer, scheduler, train_loader, val_loader, epoch, num_epochs, device, checkpoint_path, num_classes):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    patience = 0
    class_names = labels[str(SCENARIO)]
    # 只保存最佳模型的混淆矩阵（避免频繁保存）
    best_cm_saved = False  

    ema = AveragedModel(model)    

    for epc in range(epoch, num_epochs):
        print(f'\nEpoch {epc+1}/{num_epochs}')
        print('=' * 60)
        train_loss, train_acc = train_epoch(model, optimizer, scheduler, train_loader, criterion, device, ema)
        val_loss, val_acc, all_true, all_pred = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            save_checkpoint(model, optimizer, scheduler, ema, epoch + 1, checkpoint_file)
            print(f'保存新的最佳模型，验证准确率: {best_acc:.2f}%')
            # 仅在最佳模型时保存一次混淆矩阵（覆盖旧文件）
            cm_save_path = os.path.join(CHECKPOINT_PATH, 'best_confusion_matrix.png')
            plot_confusion_matrix(all_true, all_pred, class_names, cm_save_path)
            best_cm_saved = True
        else:
            patience += 1
            if patience > PATIENCE:
                print(f'早停触发，{patience}个epoch无改善')
                break
    
    return best_acc

if __name__ == "__main__":
    
    print(f'使用设备: {DEVICE}')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = prepare_datasets(root=DATASET_PATH, scenario=SCENARIO, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, dim=DIM)
    model = FileFormatAnalyzer(num_classes=MAX_CLASSES, dim=DIM)
    model = nn.DataParallel(model).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=TOP_LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
    # scheduler = CustomLRScheduler(optimizer, HEATING_EPOCHS, COOLING_EPOCHS, TOP_LR, BOTTOM_LR, EPOCHS)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,        
        T_mult=2,      
        eta_min=1e-6
    )
    
    epoch = 0
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if os.path.exists(checkpoint_file) and CONTINUE_FROM_CHECKPOINT:
        print(f'Loading checkpoint from {checkpoint_file}')
        model, optimizer, scheduler, ema, epoch = load_checkpoint(model, optimizer, scheduler, ema, checkpoint_file)


    # 开始训练

    print(f'开始训练，总epoch数: {EPOCHS}')
    print(f'当前epoch: {epoch}')
    print(f'学习率范围: {TOP_LR} -> {BOTTOM_LR}')
    print(f'批次大小: {BATCH_SIZE}')
    print(f'类别数: {MAX_CLASSES}')
    print(f'早停耐心值: {PATIENCE}')
    
    start_time = time.time()
    best_acc = train(model, optimizer, scheduler, train_loader, val_loader, epoch, EPOCHS, DEVICE, checkpoint_file, MAX_CLASSES)
    end_time = time.time()
    
    print(f'\n训练完成!')
    print(f'最佳验证准确率: {best_acc:.2f}%')
    print(f'总训练时间: {(end_time - start_time)/3600:.2f} 小时')