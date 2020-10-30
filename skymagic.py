#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :skymagic.py
# @Time     :2020/10/30 上午10:57
# @Author   :Chang Qing

import os
import sys
import argparse

import cv2
import torch

from infer_engine import MagicSky
from infer_engine import infer_utils

# device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
parser = argparse.ArgumentParser(description="Sky Magic")
parser.add_argument('-p', '--path', type=str, default="./config/bgsky_configs/default_sky.json",
                    help="sky configuration file")

if __name__ == "__main__":
    sky_config_path = parser.parse_args().path
    sky_config = infer_utils.parse_sky_config(sky_config_path)
    magic_sky = MagicSky(sky_config)
    magic_sky.magic_prepare()
    magic_sky.magic()




