#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :skymagic.py
# @Time     :2020/10/30 上午10:57
# @Author   :Chang Qing

import os
import argparse

import cv2
from infer_engine import MagicSky
from infer_engine import infer_utils

# device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
parser = argparse.ArgumentParser(description="Sky Magic")
parser.add_argument('-p', '--path', type=str, default="./config/bgsky_configs/default_video_sky.json",
                    help="sky configuration file")
parser.add_argument('--input', type=str, default=None, help="input file path")
parser.add_argument('--bgsky', type=str, default=None, help="background sky path")


if __name__ == "__main__":
    args = parser.parse_args()
    sky_config_path = args.path
    # sky_config_path = "./config/bgsky_configs/default_video_sky.json"
    sky_config = infer_utils.parse_sky_config(sky_config_path)
    if args.input is not None:
        sky_config["datadir"] = args.input
    if args.bgsky is not None:
        sky_config["sky_box"] = args.bgsky
    input_suffix = os.path.splitext(sky_config["datadir"])[-1]
    if input_suffix.lower in ["png", 'jpg', 'jpeg']:
        sky_config["input_mode"] = "image"
    else:
        sky_config["input_mode"] = "video"
    # sky_config["arch"] = {
    #     "type": "UNet",
    #     "args": {
    #         "backbone": "mobilenetv2",
    #         "num_classes": 1,
    #         "pretrained_backbone": None
    #     }
    # }
    # sky_config["ckptdir"] = "/home/changqing/workspaces/MagicSky_Pytorch/checkpoints/UNet/1116_201425/model_best.pth"
    # sky_config["datadir"] = "./test_videos/canyon.mp4"
    # sky_config["sky_box"] = "jupiter.jpg"
    sky_config["datadir"] = "./test_videos/annarbor.mp4"
    sky_config["sky_box"] = "floatingcastle.jpg"
    start_tick = cv2.getTickCount()
    magic_sky = MagicSky(sky_config)
    magic_sky.magic_prepare()
    magic_sky.magic()
    end_tick = cv2.getTickCount()
    time_cost = (end_tick - start_tick) / cv2.getTickFrequency()
    print("time_cost: ", time_cost)

'''
7s 的视频  206 帧 实际处理172帧  耗时77.8s  输入384，384 输出485，840  model 17ms refine 145ms 
7s                      172帧  耗时79.5s                           22ms              145
'''


