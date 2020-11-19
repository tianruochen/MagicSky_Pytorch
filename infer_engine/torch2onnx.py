#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :torch2onnx.py
# @Time     :2020/11/5 下午3:57
# @Author   :Chang Qing

import os
import argparse

import cv2
import onnx
import torch
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt

import models as arch_module
from utils import get_instance
from infer_engine import MagicSky
from infer_engine import infer_utils
from utils.common_utils import cvtcolor_and_resize
from infer_engine.synthesize import SynthesizeEngine

os.environ["CUDA_VISITED_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
parser = argparse.ArgumentParser(description="Sky Magic")
parser.add_argument('-p', '--path', type=str, default="../config/bgsky_configs/annarbor-castle.json",
                    help="sky configuration file")


def transform_to_onnx(batch_size, sky_config):
    model = get_instance(arch_module, "arch", sky_config)
    model.load_pretrained_model(sky_config["ckptdir"])
    in_size_h = sky_config["in_size_h"]
    in_size_w = sky_config["in_size_w"]
    input_names = ["input"]
    output_names = ["p_mask"]

    dynamic = False
    if batch_size < 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1,3,in_size_h,in_size_w))
        onnx_file_name = "{}-1_3_{}_{}-dynamic.onnx".format(model.__class__.__name__, in_size_h, in_size_w)
        dynamic_axes = {"input": {0: "batch_size"}, "p_mask": {0: "batch_size"}}
    else:
        x = torch.randn((batch_size, 3, in_size_h, in_size_w))
        onnx_file_name = "{}-1_3_{}_{}-static.onnx".format(model.__class__.__name__, in_size_h, in_size_w)
        dynamic_axes = None

    # export the onnx model
    torch.onnx.export(model, x, onnx_file_name,
                      verbose=False,
                      training=False,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)
    print("Onnx model export done")
    # return onnx model file name
    return onnx_file_name


def model_forward(session, image_src):
    print(session.get_inputs()[0].shape)
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # init input data
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, 0)
    img_in = img_in / 255.0
    print("Shape of the model input: ", img_in.shape)

    # forward to the model and get G_pred
    input_name = session.get_inputs()[0].name
    G_pred = session.run(None, {input_name: img_in})
    return G_pred[0]

def post_processing(image_src, G_pred, sky_config):
    imgHD_w = sky_config["out_size_w"]
    imgHD_h = sky_config["out_size_h"]
    imgHD = cvtcolor_and_resize(image_src, imgHD_w, imgHD_h)
    synthesize_engine = SynthesizeEngine(sky_config)
    synthesize_engine.load_bgsky(imgHD_w, imgHD_h)
    skymask = synthesize_engine.skymask_refinement(G_pred, imgHD)
    syneth = synthesize_engine.skyblend(imgHD, None, skymask)
    syneth = syneth.clip(min=0, max=1)

    return syneth, skymask

def main(sky_config):
    data_path = sky_config["datadir"]
    batch_size = 1

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(batch_size, sky_config)
    else:
        # Transform to onnx as specific batch size
        transform_to_onnx(batch_size, sky_config)
        onnx_path_demo = transform_to_onnx(1, sky_config)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    data_path = "/home/changqing/workspaces/MagicSky_Pytorch/test_images/img1.jpg"
    image_src = cv2.imread(data_path)
    print(image_src.shape)
    G_pred = model_forward(session, image_src)
    # G_pred = G_pred * 255.0
    print(G_pred.shape)   # 1,1,384,384
    G_pred = torch.from_numpy(G_pred)
    G_pred = torch.nn.functional.interpolate(G_pred, (480,845), mode='bicubic', align_corners=False)
    G_pred = G_pred[0, :].permute([1, 2, 0])  # 480,845,1
    G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)   # 480, 845, 3
    G_pred = np.array(G_pred.detach().cpu())
    G_pred = np.clip(G_pred, a_min=0, a_max=1)


    # post processing
    syneth, refined_skymask = post_processing(image_src, G_pred, sky_config)

    plt.imsave("onnx_coarse_skymask.jpg", G_pred)
    plt.imsave("onnx_refine_skymask.jpg", refined_skymask)
    plt.imsave("onnx_syneth_image.jpg", syneth)






if __name__ == "__main__":
    print("Converting ot onnx and run demo...")
    sky_config_path = parser.parse_args().path
    # sky_config_path = "./config/bgsky_configs/canyon-jupiter.json"
    sky_config = infer_utils.parse_sky_config(sky_config_path)
    # sky_config["arch"] = {
    #     "type": "UNet",
    #     "args": {
    #         "backbone": "mobilenetv2",
    #         "num_classes": 1,
    #         "pretrained_backbone": None
    #     }
    # }
    # sky_config[
    #     "ckptdir"] = "/home/changqing/workspaces/MagicSky_Pytorch/checkpoints/UNet/1102_190654/model_best.pth"
    # sky_config["datadir"] = "./test_videos/canyon.mp4"
    # sky_config["sky_box"] = "jupiter.jpg"
    main(sky_config)
    # magic_sky = MagicSky(sky_config)
    # magic_sky.magic_prepare()
    # magic_sky.magic()


