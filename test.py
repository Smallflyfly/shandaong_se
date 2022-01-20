#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2022/01/14
"""
import glob

import torch
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import Config, seed_it
from dataset.dataset import MyDataset, get_val_transforms
from model.model import MyModel

CFG = Config()
seed_it(CFG.seed)


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_image_paths = sorted(glob.glob(CFG.test_image_paths))
    test_image_paths = np.array(test_image_paths)

    print('开始测试........')
    print('测试集数量：', test_image_paths.shape)
    test_dataset = MyDataset(test_image_paths, test_image_paths, get_val_transforms(CFG), mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_folder_n = 10
    # test_folder_n = CFG.n_fold
    model_lists = []
    for fold in range(7, 10):
        model = MyModel(num_classes=CFG.num_classes).to(device)
        model.load_state_dict(torch.load(CFG.model_save_dir + 'fold_' + str(fold + 1) + '_best.pth'))
        model.eval()
        model_lists.append(model)

    for i, inputs in enumerate(tqdm(test_loader)):
        out_all = []
        for fold in range(len(model_lists)):
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


if __name__ == '__main__':
    test()