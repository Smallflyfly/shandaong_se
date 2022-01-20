#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/20 16:44 
"""
from config.config import Config, seed_it
from model.model import UnetPP
from train_tool import train

CFG = Config()
seed_it(CFG.seed)

if __name__ == '__main__':
    model = UnetPP(num_classes=CFG.num_classes)
    train(model)