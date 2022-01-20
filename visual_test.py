#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2022/01/17
"""

import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

from config.config import Config, seed_it

warnings.filterwarnings("ignore")

CFG = Config()
seed_it(CFG.seed)


def img_visual(img_path, label_path, CFG):
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
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    print('mask:', mask.shape)
    # mask = cv2.imread(label_path, 0) - 1
    mask = cv2.imread(label_path, 0)
    print(mask)

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
    # gendi_num, lindi_num, caodi_num, shuiyu_num, chengxiang_num, others_num = 0, 0, 0, 0, 0, 0
    # label = cv2.imread(label_path)
    # gendi_num += np.sum(label == 1)
    # lindi_num += np.sum(label == 2)
    # caodi_num += np.sum(label == 3)
    # shuiyu_num += np.sum(label == 4)
    # chengxiang_num += np.sum(label == 5)
    # others_num += np.sum(label == 6)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示的问题
    # plt.rcParams['axes.unicode_minus'] = False
    # classes = ('耕地', '林地', '草地', '水域', '城乡、工矿、居民用地', '未利用土地')
    # numbers = [gendi_num, lindi_num, caodi_num, shuiyu_num, chengxiang_num, others_num]
    # plt.barh(classes, numbers)
    # for i, v in enumerate(numbers):
    #     plt.text(v, i, str(round(v / label.size / len(label_paths) * 100, 1)) + "%", verticalalignment="center")
    # plt.title('类别数目')
    # plt.xlabel('像素数量')
    # plt.ylabel('类别')
    # plt.show()


if __name__ == '__main__':
    img_visual('data/test_GF/000025_GF.tif', 'results/000025_LT.tif', CFG)
    # img_visual('data/train_GF/000005_GF.tif', 'data/train_LT/000005_LT.tif', CFG)