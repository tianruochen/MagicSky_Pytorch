# MagicSky-Pytorch


![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)

A pytorch implementation of SkyAR
- paper 
- Source code

```
├── README.md
├── base                            base class for model and trainer
├── checkpoints                     model checkpoints
├── config                     
│     ├── bgsky_configs             some config files for background sky image
│     ├── database_configs          some config files for database
│     ├── train_configs             some config files for training different models
├── dataloaders                
│     ├── dataloader.py             dataset and dataloader
├── datasets    
│     ├── cvprw2020_sky_seg         the images for trianing and validation          
├── demo                       
├── eval_output                     restore some model outputs
├── evaluation                      the implement of loss function and evaluate metrics 
│     ├── losses.py    
│     ├── metrics.py      
├── infer_engine
│     ├── infer_utils.py            some utils function for model inference
│     ├── magicsky.py               core class for sky edit        
│     ├── skymagic_calibrator.py    custom calibrator, use to calibrate int8 TensorRT model
│     ├── synrain.py
│     ├── synthesize.py
│     ├── torch2onnx.py             convert pytorch model file to onnx file 
│     ├── onnx2trt.py               convert onnx file to trt engine
├── models                          networks for Unet, UnetPlus, Resnet50FCN, ResnetFPN
│     ├── backbonds                 some backbone inplementation   
│     ├── backbones                 
│     ├── UNet.py, UNetPlus.py...   networks                   
├── skyimages                       background images
├── test_images
├── test_videos 
├── trainer                         base class for model training
├── utils                    
├── train.py                        model training
├── skymagic.py                     model infernece
├── skymagic_server.py              a simple flask api for sky magic
├── skymagic_trt.py        

```
## data
- [CVPRW20-SkyOpt dataset](https://github.com/google/sky-optimization)
- baidu(https://pan.baidu.com/s/1PlSBuG5WW2_16AAAJDFStg    Extraction code:mr09)



## trian
1.use default paramters -- config/train_configs/config_UNet.json
```angular2
python train.py
```
2.use specific config file
```angular2
python train.py --config config_path (file path in config/trian_configs/)
```
3.resume checkpoint 
```angular2
python train.py --config config_path --resume checkpoint_path
```
example:
```angular2
python train.py --config ./config/train_configs/config_UNetPlus.json \
                --resume ./checkpoints/UNetPlus/model_best.pth
```

## inference
#### comparsion on cvprw2020_sky_seg dataset

| Model type     |  backbone     | params    |  miou         
| -------------- | ----------:   | --------: | --------: 
| UNet           |  mobilenet v2 |   17MB    |   0.639 |     
| ResNetFPN      |  resnet18     |  63.37MB  |   0.671 | 
| ResNetFPN      |  resnet50     |  276.1MB  |         |           
| UNetPlus       |  resnet50     |  117.8MB  |         |     
| ResNet50FCN    |  resnet50     |  184.9MB  |   0.693 |     

#### trained model path  (10.10.101.15)
| Model type     |  Model path     
| -------------- | ----------  | 
|UNet-mobilenetv2   |  /home/changqing/workspaces/MagicSky_Pytorch/checkpoints/UNet/1116_201425/model_best.pth |
|ResNetFPN_resnet18 |  /home/changqing/workspaces/MagicSky_Pytorch/checkpoints/ResNetFPN/1117_095050/model_best.pth |
|ResNetFPN_resnet50 | -- |
|UNetPlus_resnet50  | -- |
|ResNet50FCN_resnet50| /home/changqing/workspaces/MagicSky_Pytorch/checkpoints/ResNet50FCN/1116_201917/model_best.pth|

## inference
> use default params
```angular2 
python skymagic.py
# for image input
python skymagic.py --path ./config/bgsky_configs/default_image_sky.json
# for video input
python skymagic.py --path ./config/bgsky_configs/default_video_sky.json
```     
> use specific bgsky params 
```angular2
python skymagic.py --path bgsky_config.json    #(config file in ./config/bgsky_configs/)
```
> custom input and background 
```angular2
python skymagic.py --input the_path_of_input_file --bgsky the_path_of_bgsky
```
> flask service test
```angular2
python skymagic_server.py     # start flask service
python server_test.py         # test  

'''
 payload = {
        "url": "http://tbvideo.ixiaochuan.cn/zyvdorigine/b4/15/f057-f5f9-4acc-9b9b-26b77abf1532",
        #"url": "https://ns-strategy.cdn.bcebos.com/ns-strategy/upload/fc_big_pic/part-00147-2263.jpg",
        #"image": b64_image_str,
        "bgsky_url": url,
        #"bgsky_image": b64_sky_image_str,
        "timestamp": t,
        "sign": s
    }
'''
for image suport b64_image_str, ndarray_str and url
for video, only support url
```