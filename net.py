import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, pdb, time, copy
from PIL import Image
import train

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, stride=(2, 2), padding=(1, 1), bias=False)
                nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            )
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x
        out += res
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 1),
            BasicBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1)
        )
        self.layer5 = nn.Sequential(
            BasicBlock(512, 512, 1),
            BasicBlock(512, 512, 1)
        )    
        self.fc = nn.Linear(512, 10, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)

model = models.resnet18()
pdb.set_trace()
# class Net()