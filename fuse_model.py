import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import tqdm
import copy

import model.yolov3
import model.yolov3_quant
import utils.datasets
import utils.utils
from test import evaluate
from torchvision import models
from torchinfo import summary

# 모델 준비하기
modules_to_fuse = ["0", "1", "2"]
inputs = torch.randn((1, 3,416, 416))
model = model.yolov3_quant.YOLOv3(416, 10)
#summary(model, (1, 3, 416, 416))

#print(model)

quant_model = copy.deepcopy(model)
quant_model.eval()
for n0, m0 in quant_model.named_children():
    for n1, m1 in m0.named_children():
        if isinstance(m1, nn.Sequential):
            for n2, m2 in m1.named_children():
                if isinstance(m2, nn.Sequential):
                    torch.ao.quantization.fuse_modules(m2, modules_to_fuse, inplace = True)
        if isinstance(m1, nn.Sequential) and len(m1) == 3:
           torch.ao.quantization.fuse_modules(m1, modules_to_fuse, inplace = True)

        
print(quant_model)
#-----------------------------------------------------darknet53-----------------------------------------------------#
'''
m=torch.quantization.fuse_modules(m, [["darknet53.conv_1.0","darknet53.conv_1.1","darknet53.conv_1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.conv_2.0","darknet53.conv_2.1","darknet53.conv_2.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_1_1.0.0","darknet53.residual_1_1.0.1","darknet53.residual_1_1.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_1_1.1.0","darknet53.residual_1_1.1.1","darknet53.residual_1_1.1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.conv_3.0","darknet53.conv_3.1","darknet53.conv_3.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_2_1.0.0","darknet53.residual_2_1.0.1","darknet53.residual_2_1.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_2_1.1.0","darknet53.residual_2_1.1.1","darknet53.residual_2_1.1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_2_2.0.0","darknet53.residual_2_2.0.1","darknet53.residual_2_2.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_2_2.1.0","darknet53.residual_2_2.1.1","darknet53.residual_2_2.1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.conv_4.0","darknet53.conv_4.1","darknet53.conv_4.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_1.0.0","darknet53.residual_3_1.0.1","darknet53.residual_3_1.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_2.1.0","darknet53.residual_3_2.1.1","darknet53.residual_3_2.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_3.0.0","darknet53.residual_3_3.0.1","darknet53.residual_3_3.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_4.1.0","darknet53.residual_3_4.1.1","darknet53.residual_3_4.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_5.0.0","darknet53.residual_3_5.0.1","darknet53.residual_3_5.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_6.1.0","darknet53.residual_3_6.1.1","darknet53.residual_3_6.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_7.0.0","darknet53.residual_3_7.0.1","darknet53.residual_3_7.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_3_8.1.0","darknet53.residual_3_8.1.1","darknet53.residual_3_8.1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.conv_5.0","darknet53.conv_5.1","darknet53.conv_5.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_1.0.0","darknet53.residual_4_1.0.1","darknet53.residual_4_1.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_2.1.0","darknet53.residual_4_2.1.1","darknet53.residual_4_2.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_3.0.0","darknet53.residual_4_3.0.1","darknet53.residual_4_3.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_4.1.0","darknet53.residual_4_4.1.1","darknet53.residual_4_4.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_5.0.0","darknet53.residual_4_5.0.1","darknet53.residual_4_5.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_6.1.0","darknet53.residual_4_6.1.1","darknet53.residual_4_6.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_7.0.0","darknet53.residual_4_7.0.1","darknet53.residual_4_7.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_4_8.1.0","darknet53.residual_4_8.1.1","darknet53.residual_4_8.1.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.conv_6.0","darknet53.conv_6.1","darknet53.conv_6.2"]])

m=torch.quantization.fuse_modules(m, [["darknet53.residual_5_1.0.0","darknet53.residual_5_1.0.1","darknet53.residual_5_1.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_5_2.1.0","darknet53.residual_5_2.1.1","darknet53.residual_5_2.1.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_5_3.0.0","darknet53.residual_5_3.0.1","darknet53.residual_5_3.0.2"]])
m=torch.quantization.fuse_modules(m, [["darknet53.residual_5_4.1.0","darknet53.residual_5_4.1.1","darknet53.residual_5_4.1.2"]])
#----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------conv_block3-----------------------------------------------------#
m=torch.quantization.fuse_modules(m, [["conv_block3.0.0","conv_block3.0.1","conv_block3.0.2"]])
m=torch.quantization.fuse_modules(m, [["conv_block3.0.0","conv_block3.0.1","conv_block3.0.2"]])
m=torch.quantization.fuse_modules(m, [["conv_block3.0.0","conv_block3.0.1","conv_block3.0.2"]])
m=torch.quantization.fuse_modules(m, [["conv_block3.0.0","conv_block3.0.1","conv_block3.0.2"]])
m=torch.quantization.fuse_modules(m, [["conv_block3.0.0","conv_block3.0.1","conv_block3.0.2"]])
'''

inputs = torch.randn(1, 3, 416, 416)
out = quant_model(inputs)
print("fused", out)


out = model(inputs)
print("normal", out)

#summary(quant_model, (1, 3, 416, 416),depth=5)
