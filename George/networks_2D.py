import torch
import torch.nn as nn
import torch.nn.functional as F  


class Res2DBlock(nn.Module):
    expansion = 1 #we don't use the block.expansion here

    def __init__(self, inplanes, planes, stride=1,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1,
                     padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out