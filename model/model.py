#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/13 10:42 
"""

import warnings

import segmentation_models_pytorch as smp
import torch.nn as nn

warnings.filterwarnings("ignore")


class MyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                decoder_attention_type="scse",
                classes=num_classes,
        )

    def forward(self, x):
        out = self.model(x)
        return out