# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from nets.util.misc import NestedTensor
from nets.position_encoding import build_position_encoding

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, planes):
        super(ResNet3D, self).__init__()
        self.inplanes = 1
        self.planes = planes

        self.layer1 = self._make_layer(block, planes=planes[0], blocks=layers[0])
        self.layer2 = self._make_layer(block, planes=planes[1], blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=planes[2], blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=planes[3], blocks=layers[3], stride=2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        feature = self.layer4(x4)
        return feature

class Backbone(nn.Module):

    def __init__(self, resnet: str, train_backbone: bool, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        if resnet == "resnet18":
            self.model = ResNet3D(block=BasicBlock, layers=[2, 2, 2, 2], planes=[24//2, 48//2, 96//2, 192//2])
        if resnet == "resnet34":
            self.model = ResNet3D(block=BasicBlock, layers=[3, 4, 6, 3], planes=[24//2, 48//2, 96//2, 192//2])        
        if resnet == "resnet50":
            self.model = ResNet3D(block=Bottleneck, layers=[3, 4, 6, 3], planes=[24//4, 48//4, 96//4, 192//4])
        for name, parameter in self.model.named_parameters():
            if train_backbone == False:
                parameter.requires_grad_(False)

    def forward(self, tensor: NestedTensor):
        x = self.model(tensor.tensor)
        out: NestedTensor = {}
        mask = tensor.mask
        # if mask is not None:
        #     mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        x = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        out.append(x)
        # position encoding
        pos.append(self[1](x).to(x.tensor.dtype))
        return out, pos


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    position_embedding = build_position_encoding(args)
    backbone = Backbone(resnet=args.backbone, train_backbone=train_backbone, num_channels=args.hidden_dim)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

if __name__=="__main__":
    model = ResNet3D(resnet="resnet-34", block=Bottleneck, layers=[3, 4, 6, 3], planes=[6, 12, 24, 48])
    image = torch.zeros([2, 1, 32, 32, 32])
    feature = model(image)
    print(feature.shape())