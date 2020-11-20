#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :server_utils.py
# @Time     :2020/11/18 下午3:38
# @Author   :Chang Qing

import os
import uuid
import datetime
from flask import jsonify
from pymongo import MongoClient
from utils.data_trans import *


def error_resp(error_code, error_message):
    resp = jsonify(error_code=error_code, error_message=error_message)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def log_info(text):
    with open("skymagic_service_log.txt", "a") as f:
        f.write('%s' % datetime.datetime.now())
        f.write('    ')
        f.write(text)
        f.write('\n')
    return


def get_connection(db_params):

    host, port = db_params["host"], db_params["port"]
    db_name, tb_name = db_params["database"], db_params["table"]
    client = MongoClient(host, int(port))
    database = client[db_name]
    table = client[tb_name]
    return table


def write2db(db_params, info):
    collection = get_connection(db_params)
    if type(info) is dict:
        _id = collection.update_one({'_id': info['_id']}, {'$set': info}, upsert=True)
    elif type(info) is list:
        for _ in info:
            _id = collection.update_one({'_id': _['_id']}, {'$set': _}, upsert=True)
    return _id


def parse_and_save_data(data, temp_dir):
    """
    parse_and_save_data
    :param data: post data (json)
    :param temp_dir: (./eval_ouput)
    :return: data_path and data_type(image:0, video:1)
    """
    if "name" in data:
        data_basename = data.get("name")
    else:
        data_basename = "test"

    if 'url' in data:
        url = data.get('url')
        data_path, data_type = url2nparr(data.get('url'), temp_dir, data_basename)
        print(data_path, data_type)
        log_info('Get %s image' % url)
    elif 'image' in data:
        # log_info('Got image buffer')

        data_path, data_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'numpy' in data:
        # log_info('Got numpy string')
        data_path, data_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return data_path, data_type


def parse_and_save_bgsky(data, temp_dir):
    """
        parse_and_save_data
        :param data: post data (json)
        :param temp_dir: (./eval_ouput)
        :return: data_path and data_type(image:0, video:1)
        """
    if "bgsky_name" in data:
        data_basename = data.get("bgsky_name")
    else:
        data_basename = "bgsky_test"

    if 'bgsky_url' in data:
        url = data.get('bgsky_url')
        bgsky_path, bgsky_type = url2nparr(data.get('bgsky_url'), temp_dir, data_basename)
        print(bgsky_path, bgsky_type)
        log_info('Get %s sky background image' % url)
    elif 'bgsky_image' in data:
        # log_info('Got image buffer')
        bgsky_path, bgsky_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'bgsky_numpy' in data:
        # log_info('Got numpy string')
        bgsky_path, bgsky_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return bgsky_path, bgsky_type