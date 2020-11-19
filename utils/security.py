#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :security.py.py
# @Time     :2020/11/6 上午11:37
# @Author   :Chang Qing
 
import time
import hmac
import hashlib
import base64
import urllib


def gen_signature(secret):
    timestamp = round(time.time() * 1000)
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign

def check_security(timestamp, sign, secrets):
    cur_timestamp = time.time() - timestamp
    # time out
    if cur_timestamp - timestamp > 60:
        return False, None
    for secret in secrets:
        # generate candidate sign
        secret_encode = secret.encode("utf-8")
        str_sign = "{}\n{}".format(timestamp, secret)
        str_sign_encode = str_sign.encode("utf-8")
        hmac_code = hmac.new(secret_encode, str_sign_encode, digestmod=hashlib.sha256).digest()
        candidate_sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        # match
        if candidate_sign == sign:
            return True, secret
    return False, None


