#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/06
"""
import os
import os.path as osp
import pickle
import warnings
from collections import OrderedDict
from functools import partial

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")


def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
        checkpoint = load_checkpoint(weight_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        print(
            'Successfully loaded pretrained weights from "{}"'.
                format(weight_path)
        )


def build_optimizer(model, optim='adam', lr=0.005, weight_decay=5e-04, momentum=0.9, sgd_dampening=0,
                    sgd_nesterov=False, rmsprop_alpha=0.99, adam_beta1=0.9, adam_beta2=0.99, staged_lr=False,
                    new_layers='', base_lr_mult=0.1):
    param_groups = model.parameters()
    optimizer = None
    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups,  lr=lr, weight_decay=weight_decay)

    return optimizer


def build_scheduler(optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1):
    global scheduler
    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch))

    elif lr_scheduler == 'cosine_anneal_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6,
                                                                         last_epoch=-1)

    return scheduler


def check_dir_file(dir_files):
    if isinstance(dir_files, str):
        dir_files = [dir_files]
    for fpath in dir_files:
        if not os.path.exists(fpath):
            raise FileNotFoundError("{} IS NOT FOUND".format(fpath))


def eda_visual(img_paths, label_paths, CFG):
    # 统计最大像素
    #     min_pix = 100000000
    #     max_pix = -100000000
    #     for i in tqdm(range(len(img_paths))):
    #         image = cv2.imread(img_paths[i],cv2.IMREAD_UNCHANGED)
    #         mask = cv2.imread(label_paths[i], 0) - 1
    #         augments = get_eda_transforms()(image=image, mask=mask)

    #         image, mask = augments['image'], augments['mask']
    #         image = np.ascontiguousarray(image)
    #         temp_min = image.min()
    #         temp_max = image.max()

    #         if temp_min < min_pix:
    #             min_pix = temp_min
    #         if temp_max > max_pix:
    #             max_pix= temp_max
    #     print(min_pix)
    #     print(max_pix)
    # 显示一幅图和标签
    index = 3
    image = cv2.imread(img_paths[index], cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(label_paths[index], cv2.IMREAD_UNCHANGED)
    print('mask:', mask.shape)
    mask = cv2.imread(label_paths[index], 0) - 1

    def get_eda_transforms():
        return A.Compose([
            A.RandomResizedCrop(CFG.img_size, CFG.img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(p=0.25),
        ], p=1.)

    augments = get_eda_transforms()(image=image, mask=mask)
    image, mask = augments['image'], augments['mask']
    image = np.ascontiguousarray(image) / 1023.0
    plt.figure(figsize=(18, 16))
    plt.subplot(121)
    plt.imshow(mask)
    plt.subplot(122)
    plt.imshow(image)
    print(image.shape)
    plt.show()
    # 统计标签面积比列
    gendi_num, lindi_num, caodi_num, shuiyu_num, chengxiang_num, others_num = 0, 0, 0, 0, 0, 0
    for label_path in label_paths:
        label = cv2.imread(label_path)
        gendi_num += np.sum(label == 1)
        lindi_num += np.sum(label == 2)
        caodi_num += np.sum(label == 3)
        shuiyu_num += np.sum(label == 4)
        chengxiang_num += np.sum(label == 5)
        others_num += np.sum(label == 6)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示的问题
    plt.rcParams['axes.unicode_minus'] = False
    classes = ('耕地', '林地', '草地', '水域', '城乡、工矿、居民用地', '未利用土地')
    numbers = [gendi_num, lindi_num, caodi_num, shuiyu_num, chengxiang_num, others_num]
    plt.barh(classes, numbers)
    for i, v in enumerate(numbers):
        plt.text(v, i, str(round(v / label.size / len(label_paths) * 100, 1)) + "%", verticalalignment="center")
    plt.title('类别数目')
    plt.xlabel('像素数量')
    plt.ylabel('类别')
    plt.show()
