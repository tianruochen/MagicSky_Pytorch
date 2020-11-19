#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :logger_utils.py
# @Time     :2020/11/10 下午5:37
# @Author   :Chang Qing
 
import json
import logging


class Logger:

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

