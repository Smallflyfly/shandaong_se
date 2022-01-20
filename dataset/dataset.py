#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/12 19:30 
"""
import warnings

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def get_train_transforms(CFG):
    return A.Compose([
            # A.RandomResizedCrop(CFG.img_size, CFG.img_size),
            A.Resize(CFG.img_size, CFG.img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.25),
            A.RandomRotate90(p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0), #注意这里的最大像素是1023
            ToTensorV2(p=1.0),
        ], p=1.)

def get_val_transforms(CFG):
    return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms=None, mode='train'):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.mode = mode
        self.len = len(image_paths)
    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        if self.mode == "train":
            label = cv2.imread(self.label_paths[index], 0) - 1 #提交的是从1才代表耕地类别，模型类别是从0开始，所以-1
            augments = self.transforms(image=img, mask=label)
            return augments['image'],  augments['mask'].to(torch.int64)
        elif self.mode == "test":
            augments = self.transforms(image=img)
            return augments['image']
    def __len__(self):
        return self.len