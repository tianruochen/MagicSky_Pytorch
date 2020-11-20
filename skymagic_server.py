#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :skymagic_server.py
# @Time     :2020/11/6 上午11:16
# @Author   :Chang Qing

import json
import time
import argparse
import datetime

from waitress import serve
from pymongo import MongoClient
from flask import Flask, request, jsonify

from infer_engine import infer_utils
from infer_engine import MagicSky
from utils.security import check_security
from utils import get_time_str
from utils.server_utils import parse_and_save_data, log_info
from utils.server_utils import parse_and_save_bgsky
from utils.server_utils import error_resp


app = Flask(__name__)
# db_param = json.load(open("./config/database_configs/mongo.json"))
# if db_param["durl"]:
#     client = MongoClient(db_param["durl"])
# else:
#     client = MongoClient(host=db_param["host"], port=int(db_param["port"]))
# app.config["collection"] = client[db_param["db"]][db_param["table"]]
app.config["secrets"] = json.load(open("./config/database_configs/secrets.json"))
app.config["port"] = 0


@app.route('/healthcheck')
def healthcheck():
    return error_resp(0, "working")

@app.route("/api/magic_sky", methods=["POST"])
def sky_edit():
    if request.method == "POST":
        data = json.loads(request.data)
        if "timestamp" not in data and "sign" not in data:
            return error_resp(1, "Param miss")

        # check secrets
        secure, secret = check_security(data.get("timestamp"), data.get("sign"), app.config["secrets"])

        if not secure:
            return error_resp(1, "you need a right signature before post a request")

        # get the image or video code and path
        temp_dir = sky_config["output_dir"]
        data_path, data_type = parse_and_save_data(data, temp_dir)
        print("data_path:", data_path)
        print("data_type:", data_type)
        bgsky_path, bgsky_type = parse_and_save_bgsky(data, temp_dir)

        if data_type == 0:
            # processing image: modify sky config file
            sky_config["input_mode"] = "image"
        else:
            # processing video: modify sky config file
            sky_config["input_mode"] = "video"
        sky_config["datadir"] = data_path

        if bgsky_type != 0:
            print("Only pictures are supported for the moment")
        else:
            if bgsky_path is not None:
                sky_config["sky_box"] = bgsky_path

        # sky editing...
        log_info("%s : begin..." % get_time_str())
        magic_sky = MagicSky(sky_config)
        magic_sky.magic_prepare()
        res_path = magic_sky.magic()
        log_info("%s : end..." % get_time_str())
        # ndarray to str  .__str__()
        data_path_hash = hash(str(time.time()) + data_path)
        # write to db
        db_info = {'_id': data_path_hash,
                   'service': secret,
                   'req_time': str(get_time_str()),
                   'res_path': res_path,
                   "url": None
                   }
        print(db_info)
        # write2db(db_info)
        # log_info('Write to db %s' % data.get('url'))

        resp = jsonify(error_code=0, data=db_info)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        log_info('Edit %s %s done' % (data_path_hash,
                                      '' if db_info['url'] is None else db_info['url']))
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sky Magic Service")
    parser.add_argument("--sky_config", type=str, default="./config/bgsky_configs/default_video_sky.json",
                       help="path of sky config(json file)")
    parser.add_argument("--port", type=int, default=6606, help="service port (default is 6606)")
    # parser.add_argument("--temp_dir", type=str, default="./eval_output/", help="tamp directory for post data")

    args = parser.parse_args()
    sky_config_path = args.sky_config
    sky_config = infer_utils.parse_sky_config(sky_config_path)

    if args.port:
        app.config["port"] = args.port
    serve(app, host="0.0.0.0", port=int(args.port), threads=3)

