# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import sys
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import lean.quantize as quantize
import lean.funcs as funcs
from lean.train import qat_train

from mmcv import Config
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

#Additions
from mmcv.runner import  load_checkpoint,save_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.cnn import resnet
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d

def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def load_model(cfg, checkpoint_path = None):
    model = build_model(cfg.model)
    if checkpoint_path != None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    return model

def quantize_net(model):
    quantize.quantize_encoders_lidar_branch(model.encoders.lidar.backbone)    
    quantize.quantize_encoders_camera_branch(model.encoders.camera)
    quantize.replace_to_quantization_module(model.fuser)
    quantize.quantize_decoder(model.decoder)
    model.encoders.lidar.backbone = funcs.layer_fusion_bn(model.encoders.lidar.backbone)
    return model
    
def main():
    quantize.initialize()  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", default="bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml", help="config file")
    parser.add_argument("--ckpt", default="model/resnet50/bevfusion-det.pth", help="the checkpoint file to resume from")
    parser.add_argument("--save_path", default="qat/ckpt/bevfusion_fp16.pth", help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=300, help="calibrate batch")
    args = parser.parse_args()

    args.ptq_only = True
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # save_path = 'qat/ckpt/bevfusion_fp16.pth'
    # os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_root = os.path.join(os.path.dirname(args.ckpt), f"onnx_fp16")
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(os.path.dirname(args.ckpt), os.path.basename(args.ckpt).split(".pth")[0] + "_fp16.pth")

    #Create Model
    model = load_model(cfg, checkpoint_path = args.ckpt)
    # model = quantize_net(model)
    model = fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    torch.save(model, save_path)
    print(f"save checkpoint to target path: {save_path}")
    return

if __name__ == "__main__":
    main()