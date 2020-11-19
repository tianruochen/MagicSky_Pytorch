#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :onnx2trt.py
# @Time     :2020/11/5 下午3:58
# @Author   :Chang Qing

import sys
import onnx

import argparse
import tensorrt as trt
from infer_engine import Skymagic_Calibrator

def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(network_creation_flag) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size
        builder.max_workspace_size = 1 << 30
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            print(builder)
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx_file_path))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
            print('Beginning ONNX file parsing')
            # print(model.read())
            parser.parse(model.read())
            print(parser.parse(model.read()))
        print('Completed parsing of ONNX file')

        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))
        #
        # #Check if last layer recognizes it's output
        # if not last_layer.get_output(0):
        #     # If not, then mark the output using TensorRT API
        #     network.mark_output(last_layer.get_output(0))

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print(engine)
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


def loadEngine2TensorRT(filepath):
    '''
    通过加载计划文件，构建TensorRT运行引擎
    '''
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="convert onnx to trt")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size (default 1)")
    parser.add_argument("--channel", type=int, default=3, help="the number of input channel")
    parser.add_argument("--height", type=int, default=384, help="input height")
    parser.add_argument("--weight", type=int, default=384, help="input weight")
    parser.add_argument("--cache_file", type=str, default="", help="cache file")
    parser.add_argument("--mode", type=str, default="int8", help="trt mode (fp32, fp16 or int8)")
    parser.add_argument("--onnx_file_path", type=str, default="", help="onnx file path")
    parser.add_argument("--engine_file_path", type=str, default="", help="engine file path")
    args = parser.parse_args()

    args.onnx_file_path = "/home/changqing/workspaces/MagicSky_Pytorch/infer_engine/UNet-1_3_384_384-static.onnx"
    args.engine_file_path = "/home/changqing/workspaces/MagicSky_Pytorch/infer_engine/UNet-1_3_384_384-static.engine"
    if args.mode.lower() == "int8":
        calib = Skymagic_Calibrator(args)
    else:
        calib = None

    ONNX2TRT(args, calib)




"""
1.convert from onnx of static batch size
trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file>
        --workspace=<size_in_megabytes> --fp16
trtexec --onnx=/home/changqing/workspaces/MagicSky_Pytorch/infer_engine/UNet-1_3_384_384-static.onnx
        --explicitBatch=1
        --saveEngine=/home/changqing/workspaces/MagicSky_Pytorch/infer_engine/UNet-1_3_384_384-static.engine
        --fp16

        
2.convert form onnx of dynamic batch size
trtexec --onnx=<onnx_file> \
        --minShapes=input:<shape_of_min_batch> --optShapes=input:<shape_of_opt_batch> --maxShapes=input:<shape_of_max_batch> \
        --workspace=<size_in_megabytes> --saveEngine=<engine_file> --fp16
for example:
trtexec --onnx=yolov4_-1_3_320_512_dynamic.onnx \
        --minShapes=input:1x3x320x512 --optShapes=input:4x3x320x512 --maxShapes=input:8x3x320x512 \
        --workspace=2048 --saveEngine=yolov4_-1_3_320_512_dynamic.engine --fp16

trtexec --onnx=/home/changqing/workspaces/MagicSky_Pytorch/infer_engine/UNet-1_3_384_384-static.onnx \
        --minShapes=input:1x3x384x384 --optShapes=input:1x3x384x384   --maxShapes=input:1x1x480x845 \
        --workspace=2048 --saveEngine=UNet-1_3_384_384-static.engine --fp16
"""
