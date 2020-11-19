#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :common_utils.py
# @Time     :2020/10/30 上午11:51
# @Author   :Chang Qing

import os
import time
import cv2
import numpy as np
# ------------------------------------------------------------------------------
#   Get instance
# ------------------------------------------------------------------------------
# 从module指定的py文件中 根据用途name获取对应的tpye类，并根据参数 *args 和config['args']创建对象

def get_instance(module, name, config, *args):
    # module_class = getattr(module, config[name]['type'])
    # module_obj = module_class(*args, config['args'])
    # return module_obj
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def cvtcolor_and_resize(img_HD, out_size_w, out_size_h):

    img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
    img_HD = np.array(img_HD / 255., dtype=np.float32)
    # cv2.resize 参数 (W,H)
    img_HD = cv2.resize(img_HD, (out_size_w, out_size_h))

    return img_HD


def get_time_str():
    timestamp = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(timestamp))
    return time_str

def is_image(data_path):
    data_suffix = os.path.splitext(data_path)
    if data_suffix in ["jpg", "jpeg", "png"]:
        return True
    else:
        return False
