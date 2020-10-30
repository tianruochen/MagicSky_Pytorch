#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :dataloader.py
# @Time     :2020/10/28 下午5:27
# @Author   :Chang Qing

import os
import glob
import random

import cv2
import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class PairedDataAugmentation:

    def __init__(
            self, img_resize, with_random_flip=False, with_random_rotate=False,
            with_random_crop=False, with_random_brightness=False,with_random_gamma=False,
            with_random_saturation=False
    ):
        self.img_resize = img_resize
        self.with_random_flip = with_random_flip
        self.with_random_rotate = with_random_rotate
        self.with_random_crop = with_random_crop
        self.with_random_brightness = with_random_brightness
        self.with_random_gamma = with_random_gamma
        self.with_random_saturation = with_random_saturation

    def transform(self, img1, img2):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_resize, self.img_resize], interpolation=3)
        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_resize, self.img_resize], interpolation=3)

        if self.with_random_flip and random.random() > 0.5:
            if random.randint(0, 2):
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
            else:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)

        if self.with_random_rotate and random.random() > 0.5:
            angle = random.randint(1, 4) * 90
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_resize). \
                get_params(img=img1, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_resize, self.img_resize))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_resize, self.img_resize))

        if self.with_random_brightness and random.random() > 0.5:
            # multiply a random number within a - b
            img1 = TF.adjust_brightness(img1, brightness_factor=random.uniform(0.5, 1.5))

        if self.with_random_gamma and random.random() > 0.5:
            # img**gamma
            img1 = TF.adjust_gamma(img1, gamma=random.uniform(0.5, 1.5))

        if self.with_random_saturation and random.random() > 0.5:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            img1 = TF.adjust_saturation(img1, saturation_factor=random.uniform(0.5, 1.5))

        # to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2


#------------------------------------------------------------------------------
#	DataSet for Sky Segmentation
#------------------------------------------------------------------------------

class MagicSkyDataSet(Dataset):

    def __init__(self, root_dir, img_resize=224, random_flip=True, random_crop=True,
                 random_rotate=True, random_brightness=True, random_gamma=True,
                 random_saturation=True, is_train=True):

        self.root_dir = root_dir
        self.img_resize = img_resize
        self.is_train = is_train
        if self.is_train:
            self.img_paths = glob.glob(os.path.join(os.path.abspath(self.root_dir), "images/train/", "*.jpg"))
            print(len(self.img_paths))
            self.augm = PairedDataAugmentation(
                img_resize=self.img_resize,
                with_random_flip=random_flip,
                with_random_crop=random_crop,
                with_random_rotate=random_rotate,
                with_random_gamma=random_gamma,
                with_random_brightness=random_brightness,
                with_random_saturation=random_saturation
            )
        else:
            self.img_paths = glob.glob(os.path.join(self.root_dir, "images/val", "*.jpg"))
            self.augm = PairedDataAugmentation(
                img_resize=self.img_resize
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input image
        img_A = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        # label image
        img_B_path = self.img_paths[idx].replace('images', 'density_estimation+guided_filter').replace('.jpg', '.png')
        img_B = cv2.imread(img_B_path, cv2.IMREAD_COLOR)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        img_A, img_B = self.augm.transform(img_A, img_B)

        return img_A, img_B
#------------------------------------------------------------------------------
#	DataLoader for Sky Segmentation
#------------------------------------------------------------------------------

class MagicSkyDataLoader(object):

    def __init__(self, root_dir, img_resize=224, random_flip=False, random_crop=False,
                 random_rotate=False,random_brightness=False, random_gamma=False,
                 random_saturation=False, is_train=True,
                 shuffle=True, batch_size=1, n_workers=1, pin_memory=True):
        super(MagicSkyDataLoader, self).__init__()

        # parameters of dataset
        self.root_dir = root_dir               # 训练数据根目录
        self.img_resize = img_resize
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.random_rotate = random_rotate
        self.random_brightness = random_brightness
        self.random_gamma = random_gamma
        self.random_saturation = random_saturation
        self.is_train = is_train

        # parameters of dataloader
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory


        self.dataset = MagicSkyDataSet(
            root_dir=self.root_dir,
            img_resize=self.img_resize,
            random_flip=self.random_flip,
            random_crop=self.random_crop,
            random_rotate=self.random_rotate,
            random_brightness=self.random_brightness,
            random_gamma=self.random_gamma,
            random_saturation=self.random_saturation,
            is_train=self.is_train
        )

    @property
    def loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

