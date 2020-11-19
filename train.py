#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :train.py.py
# @Time     :2020/10/28 下午3:24
# @Author   :Chang Qing

import os
import json
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import models as arch_module

import dataloaders.dataloader as data_module
import evaluation.losses as loss_module

import evaluation.metrics as metric_module

from utils import Logger
from trainer.trainer import Trainer



#------------------------------------------------------------------------------
#   Get instance
#------------------------------------------------------------------------------
# 从module指定的py文件中 根据用途name获取对应的tpye类，并根据参数 *args 和config['args']创建对象
def get_instance(module, name, config, *args):
    # module_class = getattr(module, config[name]['type'])
    # module_obj = module_class(*args, config['args'])
    # return module_obj
	return getattr(module, config[name]['type'])(*args, **config[name]['args'])


#------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------
def main(config, resume, device):

    train_logger = Logger()
    # Setup data_loader instance
    train_loader = get_instance(data_module, "train_loader", config).loader
    valid_loader = get_instance(data_module, "valid_loader", config).loader


    # Build model architecture
    model = get_instance(arch_module, "arch", config)
    # Summary need model input size
    # img_resize = config["train_loader"]["args"]["img_resize"]
    # model.summary(input_shape=(3, img_resize, img_resize))

    # Get loss function and metrics
    loss = getattr(loss_module, config["loss"])
    # loss = nn.MSELoss()
    metrics = [getattr(metric_module, met) for met in config['metrics']]

    # Build optimizer and learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, "optimizer", config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, "lr_scheduler", config, optimizer)

    # Create trianer and start training
    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      device=device,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)
    trainer.train()



#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Magic sky train model")
    parser.add_argument('-c', '--config', default="./config/train_configs/config_UNetPlus.json", type=str, help='config file path (default:None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default:None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of gpus to enable (default: all)' )
    args = parser.parse_args()

    # load configuration file
    if args.resume:
        # load config file form checkpoint
        config = torch.load(args.resume)['config']
    elif args.config:
        # load config file from cmd
        config = json.load(open(args.config))
    else:
        # Assert Error
        raise AssertionError("Configuration file need to be specific. Add -c config.json, for example.")

    print("current gpu: ", torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # run the main function
    main(config, args.resume, device)

