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
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(inplanes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(stride, stride), bias=False),
            )
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x
        out += res
        return out

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if inplanes != planes * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(planes * self.expansion, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_ = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            # BasicBlock(64, 64, 1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1),
            # BasicBlock(128, 128, 1),
            # BasicBlock(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
            # BasicBlock(256, 256, 1),
            # BasicBlock(256, 256, 1),
            # BasicBlock(256, 256, 1),
            # BasicBlock(256, 256, 1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1),
            # BasicBlock(512, 512, 1),
        )
        self.fc = nn.Linear(512, 10, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # pdb.set_trace()
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1_(out)
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = Net()
    pdb.set_trace()
    # class Net()