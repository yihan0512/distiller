#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .simplenet_cifar import simplenet_cifar
from .resnet_cifar import resnet20_cifar

from collections import OrderedDict

import ipdb

__all__ = ['doublenet_cifar']


class Doublenet(nn.Module):
    def __init__(self):
        super(Doublenet, self).__init__()
        self.auxnet = simplenet_cifar()
        self.deepnet = resnet20_cifar()
        self.fc = nn.Linear(20, 10)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.relu,
        )
    
    def forward(self, x):
        # ipdb.set_trace()
        x1 = self.auxnet(x)
        x2 = self.deepnet(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        # print(x)
        return x


def update_ckpt(ckpt):
    new_state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        new_state_dict[key.replace('module.', '')] = value
    ckpt['state_dict'] = new_state_dict
    return ckpt


def doublenet_cifar():
    pth = '/home/ru4n6/Documents/distiller/examples/classifier_compression/logs'
    model = Doublenet()
    
    # loading pretrained weights
    # print('loading weights for deepnet...')
    # ckpt = torch.load(os.path.join(pth, 'resnet20-best-1/best.pth.tar'))
    # ckpt = update_ckpt(ckpt)
    # model.deepnet.load_state_dict(ckpt['state_dict'])
    # print('loading weights for auxnet...')
    # ckpt = torch.load(os.path.join(pth, 'simplenet-best-2/best.pth.tar'))
    # ckpt = update_ckpt(ckpt)
    # model.auxnet.load_state_dict(ckpt['state_dict'])
    return model
