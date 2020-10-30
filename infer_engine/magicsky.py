#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :magicsky.py
# @Time     :2020/10/30 上午11:14
# @Author   :Chang Qing

import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import models as arch_module
from utils import get_instance, cvtcolor_and_resize
from infer_engine.synthesize import SynthesizeEngine

os.environ["CUDA_VISITED_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MagicSky(object):

    def __init__(self, sky_config):

        self.ckptdir = sky_config["ckptdir"]         # model checkpoint
        self.input_mode = sky_config["input_mode"]   # video or image
        self.datadir = sky_config["datadir"]         # input path
        self.arch = sky_config["arch"]
        self.model = get_instance(arch_module, 'arch', sky_config)

        # model input size
        self.in_size_w, self.in_size_h = sky_config["in_size_w"], sky_config["in_size_h"]
        # final output size
        self.out_size_w, self.out_size_h = sky_config["out_size_w"], sky_config["out_size_h"]

        self.sky_box = sky_config["sky_box"]

        # some params for frame optimize
        self.skybox_center_crop = sky_config["skybox_center_crop"]
        self.auto_light_matching = sky_config["auto_light_matching"]
        self.relighting_factor = sky_config["relighting_factor"]
        self.recoloring_factor = sky_config["recoloring_factor"]
        self.halo_effect = sky_config["halo_effect"]

        self.output_dir = sky_config["output_dir"]
        self.save_jpgs = sky_config["save_jpgs"]

        self.synthesize_engine = SynthesizeEngine(sky_config)




    def magic_prepare(self):

        # load best checkpoint to inference
        self.model.load_pretrained_model(self.ckptdir)
        self.model.to(device)
        self.model.eval()

        # oupput
        if self.input_mode == "video":
            self.video_writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (self.out_size_w, self.out_size_h))
            self.video_writer_cat = cv2.VideoWriter("demo-cat.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
                                                20.0, (2* self.out_size_w, self.out_size_h))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write_video(self, img_HD, syneth):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        self.video_writer.write(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        self.video_writer_cat.write(frame_cat)

        # cv2.imshow('frame_cat', frame_cat)
        cv2.waitKey(1)

    def magic_image(self):
        pass

    def magic_imgseq(self):
        pass

    def magic_video(self):
        print("balabala....")
        print(self.datadir)
        cap = cv2.VideoCapture(self.datadir)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_pre = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if ret:
                img_HD = cvtcolor_and_resize(frame, self.out_size_w, self.out_size_h)
                if img_HD_pre is None:
                    img_HD_pre = img_HD

                # core process
                syneth, G_pred, skymask = self.synthesize_engine.synthesize(self.model,
                                        img_HD, img_HD_pre, self.in_size_w, self.in_size_h, device)

                if self.save_jpgs:
                    fpath = os.path.join(self.output_dir, str(idx) + '.jpg')
                    plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                    plt.imsave(fpath[:-4] + '_coarse_skymask.jpg', G_pred)
                    plt.imsave(fpath[:-4] + '_refined_skymask.jpg', skymask)
                    plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

                self.write_video(img_HD, syneth)
                print('processing: %d / %d ...' % (idx, num_frame))

                img_HD_prev = img_HD
                idx += 1

            else:  # if reach the last frame
                break


    def magic(self):   # change to balabala
        if self.input_mode == "image":
            self.magic_image()
        elif self.input_mode == "imgseq":
            self.magic_imgseq()
        elif self.input_mode == "video":
            self.magic_video()
        else:
            print("wrong input mode, select one in [image, imgseq, video, ]")
            exit()


        '''
        "net_G": "coord_resnet50",
        self.net_G = sky_config.net_G
        
        "ckptdir": "./checkpoints_G_coord_resnet50",
        "input_mode": "video",
        "datadir": "./test_videos/canyon.mp4",
        "skybox": "jupiter.jpg",

        "in_size_w": 384,
        "in_size_h": 384,
        "out_size_w": 845,
        "out_size_h": 480,

        "skybox_cernter_crop": 0.5,
        "auto_light_matching": false,
        "relighting_factor": 0.8,
        "recoloring_factor": 0.5,
        "halo_effect": true,
        
        "output_dir": "./eval_output",
        "save_jpgs": false
        '''