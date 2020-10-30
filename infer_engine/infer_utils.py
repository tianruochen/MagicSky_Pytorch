#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :infer_utils.py
# @Time     :2020/10/30 上午11:07
# @Author   :Chang Qing

import json

import cv2
import numpy as np
from sklearn.neighbors import KernelDensity


def parse_sky_config(sky_config_path):
    with open(sky_config_path) as f:
        data = json.load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    # sky_config = Struct(**data)
    sky_config = data
    return sky_config


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def estimate_partial_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process

    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    prev_matched_kp, cur_matched_kp = matched_keypoints

    # transform = cv2.estimateRigidTransform(np.array(prev_matched_kp),
    #                                        np.array(cur_matched_kp),
    #                                        False)
    # transform （2，3）变换矩阵。 知道了特征点在前一帧图像中的位置，并且通过跟踪得到了特征点在当前帧的位置，
    # 那么就可以得到前一帧到当前帧的欧几里得变换
    transform = cv2.estimateAffinePartial2D(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp))[0]

    if transform is not None:
        # translation x    # dx，dy决定了图像平移
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation     da决定了图像旋转
        # arctan2的值域是 [−π,π], 因为可以根据 x 1 x1 x1和 x 2 x2 x2来确定点落在哪个象限
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]

def removeOutliers(prev_pts, curr_pts):
    # prev_pts (105,1,2)
    # curr_pts (105,1,2)

    # d (105,1)
    d = np.sum((prev_pts - curr_pts)**2, axis=-1)**0.5
    # d_(105,1)
    d_ = np.array(d).reshape(-1, 1)
    # 高斯核密度估计
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(d_)
    density = np.exp(kde.score_samples(d_))   # （105）[0.79,0.79,...]

    prev_pts = prev_pts[np.where((density >= 0.1))]
    curr_pts = curr_pts[np.where((density >= 0.1))]

    return prev_pts, curr_pts

if __name__ == "__main__":
    pass