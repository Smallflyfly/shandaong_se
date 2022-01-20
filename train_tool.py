#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/20 14:46 
"""
import glob
import logging
import numpy as np

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader

from config.config import Config, seed_it
from dataset.dataset import MyDataset, get_train_transforms, get_val_transforms
from model.model import UnetModel
from train import val_model
from utils.loss import SoftDiceLoss
from utils.utils import load_pretrained_weights, build_optimizer, build_scheduler
import tensorboardX as tb

logging.basicConfig(filename='log_unet_train.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)
CFG = Config()
seed_it(CFG.seed)

bce_fn = nn.BCEWithLogitsLoss()  # nn.NLLLoss()
dice_fn = SoftDiceLoss()

writer = tb.SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer, lr_scheduler=None, fold=0, folds=0):
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
                logging.info(' Fold:{} Epoch:{}({}/{}) lr:{:6f} loss:{:6f}:'.format(
                    fold + 1, epoch, i, total_iters, optimizer.param_groups[-1]['lr'], loss.item()))

            index = fold * CFG.epochs * len(train_loader) + epoch * len(train_loader) + i + 1
            if index % 20 == 0:
                # print('add', index, loss)
                writer.add_scalar('loss', loss, index)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], index)

        lr_scheduler.step()

        # 计算验证集IoU
        val_iou = val_model(model, val_loader)
        writer.add_scalar('val_iou', np.stack(val_iou).mean(0).mean(), fold*CFG.epochs + epoch)

        train_loss_epochs.append(np.array(losses).mean())  # 保存当前epoch的train_loss.val_mIoU.lr_epochs
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 保存最优模型
        best_model_path = CFG.model_save_dir + "/" + CFG.net_name + '_fold_' + str(fold + 1) + '_best' + '.pth'
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print("Best epoch/fold/miou: {}/{}/{}".format(best_epoch, fold + 1, best_miou))

    return train_loss_epochs, val_mIoU_epochs, lr_epochs


def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio*bce + (1-ratio)*dice


def train(model):
    train_image_paths = sorted(glob.glob(CFG.train_image_paths))
    train_label_paths = sorted(glob.glob(CFG.train_label_paths))
    test_image_paths = sorted(glob.glob(CFG.test_image_paths))

    train_image_paths = np.array(train_image_paths)
    train_label_paths = np.array(train_label_paths)
    test_image_paths = np.array(test_image_paths)

    folds = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed).split(range(len(train_image_paths)),
                                                                                  range(len(train_label_paths)))
    load_pretrained_weights(model, 'weights/')
    model.to(device)

    optimizer = build_optimizer(model, optim='AdamW', lr=CFG.learning_rate, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine_anneal_warm_restarts')

    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_dataset = MyDataset(train_image_paths[trn_idx], train_label_paths[trn_idx], get_train_transforms(CFG),
                                  mode='train')
        val_dataset = MyDataset(train_image_paths[val_idx], train_label_paths[val_idx], get_val_transforms(CFG),
                                mode='train')

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

        train_loss_epochs, val_mIoU_epochs, lr_epochs = train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, fold,
                                                                    CFG.n_fold)

    writer.close()


if __name__ == '__main__':
    model = UnetModel(num_classes=CFG.num_classes)
    train(model)