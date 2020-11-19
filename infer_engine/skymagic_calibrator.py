#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :skymagic_calibrator.py.py
# @Time     :2020/11/19 上午10:23
# @Author   :Chang Qing

import os
import glob
from PIL import Image

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torchvision.transforms as transforms


class Skymagic_Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, args,
                 files_path="/home/changqing/workspaces/MagicSky_Pytorch/datasets/cvprw2020_sky_seg/images/val"):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = "skymagic_cache"
        self.channel = args.channel
        self.height = args.height
        self.width = args.weight
        self.imgs_A = glob.glob(os.path.join(files_path, "*.jpg"))
        np.random.shuffle(self.imgs_A)
        # self.imgs_B = [img_A.replace("images", "density_estimation+guided_filter").replace("jpg", "png") for img_A in self.imgs_A]
        self.batch_idx = 0
        self.batch_size = args.batch_size
        self.max_batch_idx = len(self.imgs_A) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.channel, self.height, self.width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

        self.transform = transforms.Compose([
            transforms.Resize([self.height, self.width]),
            transforms.ToTensor(),
        ])

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs_A[self.batch_idx * self.batch_size: \
                                      (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.channel, self.height, self.width), dtype=np.floag32)
            for idx, file in enumerate(batch_files):
                img = Image.open(file)
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size / self.batch_size), "not valid image! " + file
                batch_imgs[idx] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        if os.path.exits(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


