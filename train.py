#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/13 10:18 
"""

import glob
import warnings

import numpy as np
import tensorboardX as tb
import torch
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from config.config import Config, seed_it
from dataset.dataset import MyDataset, get_train_transforms, get_val_transforms
from model.model import UnetPP
from utils.utils import build_optimizer

warnings.filterwarnings("ignore")

CFG = Config()
seed_it(CFG.seed)

writer = tb.SummaryWriter()


def train_model(model, criterion, optimizer, lr_scheduler=None, fold=0, folds=0):
    total_iters = len(train_loader)
    best_miou = 0
    best_epoch = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, CFG.epochs + 1):
        losses = []
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if CFG.print_freq > 0 and (i % CFG.print_freq == 0):
                print(' Fold:{} Epoch:{}({}/{}) lr:{} loss:{}:'.format(
                    fold + 1, epoch, i, total_iters, optimizer.param_groups[-1]['lr'], loss.item()))

            index = fold * CFG.epochs * len(train_loader) + epoch * len(train_loader) + i + 1
            if index % 20 == 0:
                # print('add', index, loss)
                writer.add_scalar('loss', loss, index)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], index)

        lr_scheduler.step()

        # 计算验证集IoU
        val_iou = val_model(model, val_loader)
        writer.add_scalar('val_iou', np.stack(val_iou).mean(0).mean(), fold*folds*CFG.epochs + epoch)

        train_loss_epochs.append(np.array(losses).mean())  # 保存当前epoch的train_loss.val_mIoU.lr_epochs
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 保存最优模型
        best_model_path = CFG.model_save_dir + "/" + 'fold_' + str(fold + 1) + '_best' + '.pth'
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print("Best epoch/fold/miou: {}/{}/{}".format(best_epoch, fold + 1, best_miou))
    return train_loss_epochs, val_mIoU_epochs, lr_epochs


# 计算验证集Iou
def val_model(model, loader):
    val_iou = []
    model.eval()  # 冻结模型中的Bn和Dropout
    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            out = out.argmax(1)
            iou = cal_iou(out, labels)
            val_iou.append(iou)
    return val_iou


# 计算IoU
def cal_iou(pred, mask):
    iou_result = []
    for idx in range(CFG.num_classes):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p * t).sum()
        iou = 2 * overlap / (uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_image_paths = sorted(glob.glob(CFG.train_image_paths))
    train_label_paths = sorted(glob.glob(CFG.train_label_paths))  # sort是为了保证图片和标签顺序对应
    test_image_paths = sorted(glob.glob(CFG.test_image_paths))

    print(len(train_image_paths))
    train_image_paths = np.array(train_image_paths)
    train_label_paths = np.array(train_label_paths)
    test_image_paths = np.array(test_image_paths)
    # eda_visual(train_image_paths, train_label_paths, CFG)  # 可视化图片和标签

    folds = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed).split(range(len(train_image_paths)),
                                                                                  range(len(train_label_paths)))  # 多折
    # 初始化模型
    model = UnetPP(num_classes=CFG.num_classes).to(device)
    # 优化器,学习率策略，损失函数
    # optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=1e-3)
    optimizer = build_optimizer(model, optim='adam', lr=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2,
                                                                     eta_min=1e-5)  # 余弦退火学习率,T_0是周期，T_mult就是之后每个周期T_0 = T_0 * T_mult，eta_min最低学习率
    DiceLoss_fn = DiceLoss(mode='multiclass')  # diceloss可以缓解类别不平衡,但不容易训练
    # SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)  # 使用标签平滑的交叉熵
    focal_loss = FocalLoss('multiclass')
    criterion = L.JointLoss(first=DiceLoss_fn, second=focal_loss, first_weight=0.5, second_weight=0.5).to(
        device)  # SoftCrossEntropyLoss+DiceLoss结合的损失函数

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # if fold > 1:  # 示例代码仅呈现前两个fold的训练结果
        #     break
        print(f"===============training fold_nth:{fold + 1}======================")
        train_dataset = MyDataset(train_image_paths[trn_idx], train_label_paths[trn_idx], get_train_transforms(CFG),
                                  mode='train')
        val_dataset = MyDataset(train_image_paths[val_idx], train_label_paths[val_idx], get_val_transforms(CFG),
                                mode='train')

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

        # 训练模型
        train_loss_epochs, val_mIoU_epochs, lr_epochs = train_model(model, criterion, optimizer, scheduler, fold, CFG.n_fold)

    writer.close()
    '''
    # =============================== 测试 ========================
    print('开始测试........')
    print('测试集数量：', test_image_paths.shape)
    test_dataset = MyDataset(test_image_paths, test_image_paths, get_val_transforms(CFG), mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_folder_n = 5
    # test_folder_n = CFG.n_fold
    model_lists = []
    for fold in range(test_folder_n):
        model = MyModel(num_classes=CFG.num_classes).to(device)
        model.load_state_dict(torch.load(CFG.model_save_dir + 'fold_' + str(fold + 1) + '_best.pth'))
        model.eval()
        model_lists.append(model)

    for i, inputs in enumerate(tqdm(test_loader)):
        out_all = []
        for fold in range(len(model_lists) - 2, len(model_lists)):
            model = model_lists[fold]
            model.eval()
            inputs = inputs.to(device)
            out1 = model(inputs)
            out2 = model(torch.flip(inputs, dims=[2]))
            out2 = torch.flip(out2, dims=[2])
            out3 = model(torch.flip(inputs, dims=[3]))
            out3 = torch.flip(out3, dims=[3])
            out = (out1 + out2 + out3) / 3.0
            out_all.append(out.detach().cpu().numpy())
        out_all = np.mean(out_all, 0)
        out_all = out_all.argmax(1) + 1  # 模型输出类别是从0开始，提交的是从1才代表耕地类别，所以+1
        out_all = out_all.squeeze()
        test_name = test_image_paths[i].split('/')[-1].replace('GF', 'LT')
        io.imsave(CFG.result_save_dir + '/' + test_name, out_all)
    '''
