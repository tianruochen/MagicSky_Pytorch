#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :skymagic.py
# @Time     :2020/10/30 上午10:57
# @Author   :Chang Qing

import argparse

from infer_engine import MagicSky
from infer_engine import infer_utils

# device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
parser = argparse.ArgumentParser(description="Sky Magic")
parser.add_argument('-p', '--path', type=str, default="./config/bgsky_configs/annarbor-castle.json",
                    help="sky configuration file")


if __name__ == "__main__":

    sky_config_path = parser.parse_args().path
    # sky_config_path = "./config/bgsky_configs/canyon-jupiter.json"
    sky_config = infer_utils.parse_sky_config(sky_config_path)
    sky_config["arch"] = {
        "type": "UNet",
        "args": {
            "backbone": "mobilenetv2",
            "num_classes": 1,
            "pretrained_backbone": None
        }
    }
    sky_config["ckptdir"] = "/home/changqing/workspaces/MagicSky_Pytorch/checkpoints/UNet/1116_201425/model_best.pth"
    sky_config["datadir"] = "./test_videos/canyon.mp4"
    sky_config["sky_box"] = "jupiter.jpg"
    magic_sky = MagicSky(sky_config)
    magic_sky.magic_prepare()
    magic_sky.magic()

