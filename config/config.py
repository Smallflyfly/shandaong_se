#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/13 10:21 
"""

import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


class Config:
    train_image_paths = './data/train_GF/*.tif' #注意这里的要加*.tif
    train_label_paths = './data/train_LT/*.tif'
    test_image_paths = './data/test_GF/*.tif'
    seed = 20220 #随机种子，使代码可复现
    epochs = 20
    batch_size = 8 #训练的批处理大小
    n_fold = 10 #折数
    learning_rate = 0.5e-4 #学习率，具体的学习率、策略、损失函数等要在正文的代码里设置
    img_size = 256 #调整的图片尺寸
    num_classes = 6 #类别数目
    print_freq = 100 #训练输出的频率
    model_save_dir = './' #模型权重保存文件夹
    result_save_dir = './results/' #输出结果保存文件夹,这个如果修改注意要改最后的!zip -rj results.zip ./results命令
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)
    net_name = 'Unet'


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)