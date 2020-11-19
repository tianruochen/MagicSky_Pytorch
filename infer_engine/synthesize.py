#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :synthesize.py
# @Time     :2020/10/30 下午2:53
# @Author   :Chang Qing

import os

import cv2
import torch
import numpy as np

from cv2.ximgproc import guidedFilter
from infer_engine.infer_utils import *
# from infer_engine.infer_utils import removeOutliers
# from infer_engine.infer_utils import update_transformation_matrix
# from infer_engine.infer_utils import estimate_partial_transform
# from infer_engine.infer_utils import build_transformation_matrix

class SynthesizeEngine():

    def __init__(self, sky_config):
        self.sky_config = sky_config
        self.frame_id = 0
        self.M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # self.load_bgsky()

    def load_bgsky(self, w, h):

        self.sky_config["out_size_w"] = int(w)
        self.sky_config["out_size_h"] = int(h)

        print('initialize new sky...')

        if '.jpg' in self.sky_config["sky_box"]:
            # static backgroud
            # skybox_img = cv2.imread(os.path.join(r'./skyimages', self.sky_config["sky_box"]), cv2.IMREAD_COLOR)
            skybox_img = cv2.imread("/home/changqing/workspaces/MagicSky_Pytorch/skyimages/jupiter.jpg", cv2.IMREAD_COLOR)
            skybox_img = cv2.cvtColor(skybox_img, cv2.COLOR_BGR2RGB)

            self.skybox_img = cv2.resize(
                skybox_img, (self.sky_config["out_size_w"], self.sky_config["out_size_h"]))
            cc = 1. / self.sky_config["skybox_center_crop"]
            imgtile = cv2.resize(
                skybox_img, (int(cc * self.sky_config["out_size_w"]),
                             int(cc * self.sky_config["out_size_h"])))
            self.skybox_imgx2 = self.tile_skybox_img(imgtile)
            self.skybox_imgx2 = np.expand_dims(self.skybox_imgx2, axis=0)

        else:
            # video backgroud
            cap = cv2.VideoCapture(os.path.join(r'./skybox', self.sky_config["sky_box"]))
            m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cc = 1. / self.sky_config["skybox_cernter_crop"]
            self.skybox_imgx2 = np.zeros(
                [m_frames, int(cc * self.sky_config["out_size_h"]),
                 int(cc * self.sky_config["out_size_w"]), 3], np.uint8)
            for i in range(m_frames):
                _, skybox_img = cap.read()
                skybox_img = cv2.cvtColor(skybox_img, cv2.COLOR_BGR2RGB)
                imgtile = cv2.resize(
                    skybox_img, (int(cc * self.sky_config["out_size_w"]),
                                 int(cc * self.sky_config["out_size_h"])))
                skybox_imgx2 = self.tile_skybox_img(imgtile)
                self.skybox_imgx2[i, :] = skybox_imgx2

    def skymask_refinement(self, G_pred, img):

        # G_pred 480 845 3   img 480，845，3
        r, eps = 20, 0.01
        # 导向滤波：导向滤波比起双边滤波来说在边界附近效果较好
        refined_skymask = guidedFilter(img[:,:,2], G_pred[:,:,0], r, eps)
        refined_skymask = guidedFilter(img[:, :, 2], refined_skymask, r, eps)
        # print(refined_skymask.shape)    # 480，845

        refined_skymask = np.stack(
            [refined_skymask, refined_skymask, refined_skymask], axis=-1)   # 480，845，3

        return np.clip(refined_skymask, a_min=0, a_max=1)

    def get_skybg_from_box(self, m):

        # 更新变换矩阵 获得相对于最原始的背景图像的变化矩阵
        self.M = update_transformation_matrix(self.M, m)

        nbgs, bgh, bgw, c = self.skybox_imgx2.shape   # 1，960，1690，3
        fetch_id = self.frame_id % nbgs
        skybg_warp = cv2.warpAffine(
            self.skybox_imgx2[fetch_id, :,:,:], self.M,
            (bgw, bgh), borderMode=cv2.BORDER_WRAP)

        skybg = skybg_warp[0:self.sky_config["out_size_h"], 0:self.sky_config["out_size_w"], :]

        self.frame_id += 1

        return np.array(skybg, np.float32)/255.


    def skybox_tracking(self, frame, frame_prev, skymask):

        if np.mean(skymask) < 0.05:
            print('sky area is too small')
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
        prev_gray = np.array(255*prev_gray, dtype=np.uint8)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        curr_gray = np.array(255*curr_gray, dtype=np.uint8)    # 480，845
        # mask 480，845
        mask = np.array(skymask[:,:,0] > 0.6, dtype=np.uint8)

        template_size = int(0.05*mask.shape[0]) + 10    # 24
        mask = cv2.erode(mask, np.ones([template_size, template_size]))
        mask = cv2.erode(mask, np.ones([20, 20]))

        # mask = cv2.dilate(mask, np.ones([template_size, template_size]))

        # ShiTomasi corner detection  跟踪检测图像的角点  返回检测到的角点  （105，1，2）
        # 用于获得光流估计所需要的角点  old_gray表示输入图像，mask表示掩膜
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, mask=mask, maxCorners=200,
            qualityLevel=0.01, minDistance=20, blockSize=4)

        # prev_pts = cv2.goodFeaturesToTrack(
        #     prev_gray, maxCorners=200,
        #     qualityLevel=0.01, minDistance=30, blockSize=3)

        if prev_pts is None:
            print('no feature point detected')
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Calculate optical flow (i.e. track feature points)
        # 获得光流检测后的角点位置  curr_pts表示检测后的角点坐标(105,1,2)，st表示是否为运动的角点(105,1)，err表示是否出错(105,1)
        # 参数 old_gray(curr_pts):表示输入前一帧图像，frame_gray(curr_gray)表示后一帧图像，p0(prev_pts)表示需要检测的角点，
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None)
        # Filter only valid points
        idx = np.where(status == 1)[0]        # (105,)
        if idx.size == 0:
            print('no good point matched')
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # (105,1,2)  (105,1,2)
        print(curr_pts.shape[0])
        prev_pts, curr_pts = removeOutliers(prev_pts, curr_pts)
        print(curr_pts.shape[0])
        if curr_pts.shape[0] < 7:   #10
            print('no good point matched...')
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # limit the motion to translation + rotation
        # [0.0,0.0,0.0]
        dxdyda = estimate_partial_transform((
            np.array(prev_pts), np.array(curr_pts)))

        # 根据平移和角度 构建(2,3)的变换矩阵
        m = build_transformation_matrix(dxdyda)

        return m


    def relighting(self, img, skybg, skymask):
        # img (180,845,3)
        # skybg (480,845,3)
        # skymask (480,845,3)

        # color matching, reference: skybox_img
        step = int(img.shape[0]/20)   # 24
        skybg_thumb = skybg[::step, ::step, :]   # (20,36,3)
        img_thumb = img[::step, ::step, :]     # (20,36,3)
        skymask_thumb = skymask[::step, ::step, :]    # (20,36,3)
        # skybg_mean   求各个通道上的均值  （1，1，3）[[[0.57362145 0.5457734  0.5581264 ]]]
        skybg_mean = np.mean(skybg_thumb, axis=(0, 1), keepdims=True)
        # [[[0.31462008 0.3152687  0.29735819]]]
        # skymask_thumb: 是天空的像素值越接近1   这里的img_mean 主要求得的是非天空的区域的通道均值
        img_mean = np.sum(img_thumb * (1-skymask_thumb), axis=(0, 1), keepdims=True) \
                   / ((1-skymask_thumb).sum(axis=(0,1), keepdims=True) + 1e-9)
        # [[[0.25900137 0.23050469 0.2607682 ]]] （1，1，3）
        diff = skybg_mean - img_mean
        # 目的是调整图像的颜色，避免换天后，天空颜色和原图颜色差异过大
        img_colortune = img + self.sky_config["recoloring_factor"]*diff    # self.recoloring : 0.5

        if self.sky_config["auto_light_matching"]:
            img = img_colortune
        else:
            #keep foreground ambient_light and maunally adjust lighting
            # 保持前景环境光，并对光线进行微调
            img = self.sky_config["relighting_factor"]*(img_colortune + (img.mean() - img_colortune.mean()))

        return img


    def halo(self, syneth, skybg, skymask):

        # reflection
        halo = 0.5*cv2.blur(
            skybg*skymask, (int(self.sky_config["out_size_w"]/5),
                            int(self.sky_config["out_size_w"]/5)))
        # screen blend 1 - (1-a)(1-b)
        syneth_with_halo = 1 - (1-syneth) * (1-halo)

        return syneth_with_halo


    def skyblend(self, img, img_prev, skymask):
        print(img.shape)
        print(skymask.shape)
        if img_prev is not None:
            print(img_prev.shape)

        # 如果是处理单张图像的话 img_prev == None
        if img_prev is None:
            m = self.M
        # （2，3）的变换矩阵
        else:
            m = self.skybox_tracking(img, img_prev, skymask)

        # 根据变换矩阵获得最终可用的背景 （480，845）
        skybg = self.get_skybg_from_box(m)

        # 对图像重新打光  使得图像前景光与天空背景光差异不过于太大
        img = self.relighting(img, skybg, skymask)
        # 根据skymask合成新的图片
        # skymask[skymask >= 0.45] = 0.95
        skymask[skymask < 0.3] = 0.1
        syneth = img * (1 - skymask) + skybg * skymask

        if self.sky_config["halo_effect"]:
            # halo effect brings better visual realism but will slow down the speed
            # 增加光圈效应
            syneth = self.halo(syneth, skybg, skymask)

        # 对雨进行单独处理
        if 'rainy' in self.sky_config["sky_box"]:
            syneth = self.rainmodel.forward(syneth)

        return np.clip(syneth, a_min=0, a_max=1)


    def tile_skybox_img(self, imgtile):

        screen_y1 = int(imgtile.shape[0] / 2 - self.sky_config["out_size_h"] / 2)
        screen_x1 = int(imgtile.shape[1] / 2 - self.sky_config["out_size_w"] / 2)
        imgtile = np.concatenate(
            [imgtile[screen_y1:, :, :], imgtile[0:screen_y1, :, :]], axis=0)
        imgtile = np.concatenate(
            [imgtile[:, screen_x1:, :], imgtile[:, 0:screen_x1, :]], axis=1)

        return imgtile

    def synthesize(self, mode, img_HD, img_HD_prev, in_size_w, in_size_h, device):

        h, w, c = img_HD.shape  # 480，845，3

        img = cv2.resize(img_HD, (in_size_w, in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            print(img.shape)
            G_pred = mode(img.to(device))  # bs,1,384,384  (0-1之间)
            # bs,1,384,384 -- bs 1 480,845
            print(h,w)
            G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])  # 480,845,1
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)  # 480,845，3
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)  # 480,845,3

        # G_pred 480，845 3   img_HD 480，845，3   采用导向滤波，平滑图像并保持边缘
        skymask = self.skymask_refinement(G_pred, img_HD)

        syneth = self.skyblend(img_HD, img_HD_prev, skymask)

        # 返回合成的新图，模型的输出，天空的掩膜（不是二值图像，而是模型输出经过导向滤波后的输出）
        # shape均为：(480，845 3)
        return syneth, G_pred, skymask




