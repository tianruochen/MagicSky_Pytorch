#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :loader_test.py.py
# @Time     :2020/11/12 下午4:04
# @Author   :Chang Qing
 

import cv2
import torch
import numpy as np


import torchvision.transforms.functional as F
from PIL import Image

if __name__ == "__main__":
    gray_image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    print(gray_image.shape)   # (512, 683)
    print(gray_image.dtype)
    cv2.imwrite("gray_test.png", gray_image)
    # 图片保存 都会保存成3通道 uint8类型
    # gray_image = cv2.imread("gray_test.png")
    # print(gray_image.shape)   # (512, 683, 3)
    # print(gray_image.dtype)
    # print(gray_image)

    img1 = F.to_pil_image(gray_image)  # (683, 512)  转换成了Image类型
    print(img1.size)
    img1 = F.resize(img1, [680, 500])
    print(img1.size)
    img1 = F.hflip(img1)
    img1.save("fliped.png")
    img1 = Image.open("fliped.png")
    print(len(img1.split()))
    # img1 = cv2.imread("fliped.png")
    # img1 = F.to_pil_image(img1)
    print(img1.size)
    # F.rotate 处理的是三通道的Image图像  不能是单通道的灰度图
    img1 = F.rotate(img1, 270)
    print(img1.size)



    # norm_image = gray_image.astype(np.float64)
    # print(norm_image.dtype)
    # print(norm_image)
    #
    # cv2.imwrite("norm_gray_test.png", norm_image)
    # norm_image = cv2.imread("norm_gray_test.png")
    # print(norm_image.dtype)
    # print(norm_image)


