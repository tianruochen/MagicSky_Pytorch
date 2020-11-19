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
from utils.common_utils import is_image

os.environ["CUDA_VISITED_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MagicSky(object):

    def __init__(self, sky_config):

        self.ckptdir = sky_config["ckptdir"]  # model checkpoint
        self.input_mode = sky_config["input_mode"]  # video or image
        self.datadir = sky_config["datadir"]  # input path
        self.arch = sky_config["arch"]
        self.model = get_instance(arch_module, 'arch', sky_config)

        # model input size
        self.fix_size = sky_config["fix_size"]
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
        self.output_path = ""
        self.save_jpgs = sky_config["save_jpgs"]

        self.synthesize_engine = SynthesizeEngine(sky_config)


    def adaptive_data_size(self):
        if self.input_mode == "video":
            cap = cv2.VideoCapture(self.datadir)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 32 * 32)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 32 * 32)
            if frame_w < frame_h:
                self.in_size_w = self.out_size_w = frame_w
                self.in_size_h = self.out_size_h = frame_h
            print("adaptive data size...")
            print(self.in_size_w, self.out_size_w)
        else:
            # 图片的处理 。。。
            image = cv2.imread(self.datadir)
            self.in_size_w = self.out_size_w = int(image.shape[1] // 32 * 32)
            self.in_size_h = self.out_size_h = int(image.shape[0] // 32 * 32)


    def magic_prepare(self):

        # load best checkpoint to inference
        self.model.load_pretrained_model(self.ckptdir)
        self.model.to(device)
        self.model.eval()

        # load data info
        if not self.fix_size:
            self.adaptive_data_size()

        # oupput
        if self.input_mode == "video":
            print(self.datadir)
            video_base_name = os.path.splitext(os.path.basename(self.datadir))[0]
            bgsky_name = self.sky_box.split('.')[0]
            print(video_base_name)
            print(bgsky_name)
            video_prefix = os.path.join(os.getcwd(), "demo") + '/' + self.model.__class__.__name__ \
                           + '_' + video_base_name + '_' + bgsky_name
            print(video_prefix)
            # self.video_magic = cv2.VideoWriter(video_prefix + "_magic.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
            #                                    20.0, (self.out_size_w, self.out_size_h))
            # self.video_mask = cv2.VideoWriter(video_prefix + "_mask.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
            #                                   20.0, (self.out_size_w, self.out_size_h))
            # self.video_magic_cat = cv2.VideoWriter(video_prefix + "_magic-cat.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
            #                                        20.0, (2 * self.out_size_w, self.out_size_h))
            # self.video_mask_cat = cv2.VideoWriter(video_prefix + "_mask-cat.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
            #                                       20.0, (2 * self.out_size_w, self.out_size_h))
            self.video_all = cv2.VideoWriter(video_prefix + "_all.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
                                             20.0, (2 * self.out_size_w, 2 * self.out_size_h))
            self.output_path = video_prefix + "_all.mp4"
            print("==========")
            print(self.out_size_h, self.out_size_w)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write_video(self, img_HD, syneth, mask=None):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        # self.video_magic.write(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        # self.video_magic_cat.write(frame_cat)

        if mask is not None:
            frame_mask = np.array(255.0 * mask[:, :, ::-1], dtype=np.uint8)
            # self.video_mask.write(frame_mask)
            masked_frame = img_HD * 255.0 * mask
            masked_frame = masked_frame.astype(np.uint8)
            frame_mask_cat = np.concatenate([frame_mask, masked_frame], axis=1)
            # self.video_mask_cat.write(frame_mask_cat)
            frame_all = np.concatenate([frame_cat, frame_mask_cat], axis=0)
            print("frame_all shape : ")
            print(frame_all.shape)
            self.video_all.write(frame_all)
        # cv2.imshow('frame_cat', frame_cat)
        # cv2.waitKey(1)

    def save_images(self, src_path, syneth, G_pred, skymask):
        src_base_name, src_suffix = os.path.splitext(src_path)
        if syneth is not None:
            plt.imsave(src_base_name + "_syneth" + src_suffix, syneth)
        if G_pred is not None:
            plt.imsave(src_base_name + "_pred" + src_suffix, G_pred)
        if skymask is not None:
            plt.imsave(src_base_name + "_skymask" + src_suffix, skymask)

    def magic_image(self):
        print("magic image...")
        assert self.input_mode == "image"
        if isinstance(self.datadir, str):
            img_path = self.datadir
            print(img_path)
            img_basename = os.path.basename(img_path).split(".")[0]
            try:
                img = cv2.imread(img_path)
                print(img.shape)
            except Exception as e:
                print("Error image!")
                print(e)
                return
        elif isinstance(self.datadir, np.ndarray):
            img = self.datadir
            img_basename = "test"
        img_HD = cvtcolor_and_resize(img, self.out_size_w, self.out_size_h)
        self.synthesize_engine.load_bgsky(self.out_size_w, self.out_size_h)
        syneth, G_pred, skymask = self.synthesize_engine.synthesize(self.model,
                                                                    img_HD, None, self.in_size_w,
                                                                    self.in_size_h, device)
        # save image
        fpath = os.path.join(self.output_dir,"images", img_basename)
        plt.imsave(fpath + "_input.jpg", img_HD)
        plt.imsave(fpath + "_coarse_skymask.jpg", G_pred)
        plt.imsave(fpath + "_refined_skymask.jpg", skymask)
        plt.imsave(fpath + "_syneth.jpg", syneth.clip(min=0, max=1))
        self.output_path = fpath + "_syneth.jpg"

    def magic_imgseq(self):
        print("imgseq magic...")
        imgseq_dir = os.path.abspath(self.datadir)
        imgseq_list = [img_path for img_path in os.listdir(imgseq_dir) if is_image(img_path)]
        imgseq_list = sorted(imgseq_list)
        img_HD_pre = None
        idx = 0
        for img_path in imgseq_list:
            image = cv2.imread(img_path)
            img_HD = cvtcolor_and_resize(image, self.out_size_w, self.out_size_h)
            if img_HD_pre is None:
                img_HD_pre = img_HD

            plt.imsave(img_path[:-4]+"_input.jpg", img_HD)
            img_HD_pre = img_HD

            self.synthesize_engine.load_bgsky(self.out_size_w, self.out_size_h)
            syneth, G_pred, skymask = self.synthesize_engine.synthesize(self.model,
                                                                        img_HD, img_HD_pre,
                                                                        self.in_size_w, self.in_size_h, device)

            if self.save_jpgs:
                plt.imsave(img_path[:-4] + "_syneth.jpg", syneth)
                plt.imsave(img_path[:-4] + "_pred.jpg", G_pred)
                plt.imsave(img_path[:-4] + "_skymas.jpg", skymask)

            self.write_video(img_HD, syneth, skymask)

    def magic_video(self):
        print("video magic....")
        # print(os.path.abspath(self.datadir))
        cap = cv2.VideoCapture(self.datadir)
        frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_pre = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if ret:
                # img_HD = frame
                # self.in_size_h = int(frame_h) // 32 * 32
                # self.in_size_w = int(frame_w) // 32 * 32
                # self.out_size_h = int(frame_h) // 32 * 32
                # self.out_size_w = int(frame_w) // 32 * 32

                img_HD = cvtcolor_and_resize(frame, self.out_size_w, self.out_size_h)
                if img_HD_pre is None:
                    img_HD_pre = img_HD

                fpath = os.path.join(self.output_dir, str(idx) + '.jpg')
                plt.imsave(fpath[:-4] + '_input.jpg', img_HD)

                # core process
                self.synthesize_engine.load_bgsky(self.out_size_w, self.out_size_h)

                syneth, G_pred, skymask = self.synthesize_engine.synthesize(self.model,
                                                                            img_HD, img_HD_pre, self.in_size_w,
                                                                            self.in_size_h, device)

                if self.save_jpgs:

                    plt.imsave(fpath[:-4] + '_coarse_skymask.jpg', G_pred)
                    plt.imsave(fpath[:-4] + '_refined_skymask.jpg', skymask)
                    plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

                print(img_HD.shape, syneth.shape, skymask.shape)
                self.write_video(img_HD, syneth, skymask)
                print('processing: %d / %d ...' % (idx, num_frame))

                img_HD_pre = img_HD
                idx += 1

            else:  # if reach the last frame
                break


    def magic(self):  # change to balabala
        if self.input_mode == "image":
            self.magic_image()
        elif self.input_mode == "imgseq":
            self.magic_imgseq()
        elif self.input_mode == "video":
            self.magic_video()
        else:
            print("wrong input mode, select one in [image, imgseq, video, ]")
            exit()
        return self.output_path


if __name__ == "__main__":
    img = cv2.imread("../test_images/timg.jpeg")
    print(img.shape)
