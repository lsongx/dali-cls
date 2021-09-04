import argparse
import time
import os

from mmcls.models import build_backbone
from mmcls.core.evaluation import accuracy
from mmcls.models.initializers import dw_conv

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


root = "./data/val"
network_cfg = dict(type='MobilenetV1', implement='local')
ckpt = './data/out/checkpoint_75.49.pth'
# network_cfg = dict(type='tf_mobilenetv3_small_075', implement='timm')
# ckpt = './data/out/checkpoint_67.52.pth'
network_cfg = dict(type='resnet18', implement='torchvision')
ckpt = './data/out/checkpoint_74.07.pth'

# use_dw_conv = True
use_dw_conv = False

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
dataset = ImageFolder(root=root, transform=val_transform)
batch_size=64,
loader = DataLoader(dataset,
                    num_workers=32, 
                    batch_size=64,
                    shuffle=False,
                    pin_memory=False)

network = build_backbone(network_cfg)
if use_dw_conv:
    dw_conv(network)
network.load_state_dict(torch.load(ckpt, 'cpu'), strict=True)
network.cuda()
network.eval()

with torch.no_grad():
    prec1, prec5, total = 0., 0., 0.
    for x, y in loader:
        pred = network(x.cuda())
        p1, p5 = accuracy(pred, y.cuda(), (1, 5), False)
        prec1 += p1
        prec5 += p5
        total += x.shape[0]

print(f'Top1 {prec1.item()/total:.4%}')
print(f'Top5 {prec5.item()/total:.4%}')
