"""
Convolution Layers with pytorch
"""

import torch
import torch.nn as nn

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from functools import partial


__all__ = ['_make_divisible', 'SiLU', 'ConvNormAct', 'ResidualAdd', 'SELayer', 'MBConv', 'FusedMBConv']


def _make_divisible(v, divisor, min_value=None):
    """
    이 함수는 tf 를 참고.
    모든 레이어를 8로 나눌 수 있는 채널 수를 가지도록 하며 10% 이상 떨어지지 않도록 함.
    
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    
    
    """
    if min_value is None:
        min_value = divisor
        
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    
    if new_v < 0.9 * v:
        new_v += divisor
        
    return new_v


# SiLU 활성화 함수
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # pytorch 구버전 호환
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


# SELayer
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    
class ConvNormAct(nn.Module):
    def __init__(
        self,  
        inp, 
        oup, 
        kernel_size,
        stride,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):
        super(ConvNormAct, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, **kwards),
            norm(oup),
            act()
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super(ResidualAdd, self).__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x):
        res = x
        x = self.block(x)
        
        if self.shortcut:
            res = self.shortcut(res)
            
        x += res
        
        return x
    

class MBConv(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion,
        use_se
    ):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        
        expanded_channels = round(inp * expansion)
        
        self.identity = stride == 1 and inp == oup
        
        layers = []
        
        layers.append(
            # pw
            ConvNormAct(
                inp=inp, 
                oup=expanded_channels, 
                kernel_size=1, 
                stride=1, 
                act=SiLU, 
                bias=False),
        )
        
        layers.append(
            # dw
            ConvNormAct(
                inp=expanded_channels, 
                oup=expanded_channels, 
                kernel_size=3, 
                stride=stride,
                groups=expanded_channels,
                act=SiLU,
                bias=False),
        )
        
        if use_se:
            layers.append(
                # SE
                SELayer(inp, expanded_channels)
            )
        
        # pw-linear
        layers.append(
            nn.Conv2d(expanded_channels, oup, 1, 1, 0, bias=False))
        layers.append(
            nn.BatchNorm2d(oup))
        
        
        self.conv = nn.Sequential(*layers)
        self.res = Residual(self.conv)
        
        
    def forward(self, x):
        if self.identity:
            return self.res(x)
        else:
            return self.conv(x)
        
        
class FusedMBConv(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion,
        use_se
    ):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        
        expanded_channels = round(inp * expansion)
        
        self.identity = stride == 1 and inp == oup
        
        layers = []
        
        layers.append(
            # pw
            ConvNormAct(
                inp=inp, 
                oup=expanded_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                act=SiLU, 
                bias=False),
        )
        
        if use_se:
            layers.append(
                # SE
                SELayer(inp, expanded_channels)
            )
        
        # pw-linear
        layers.append(
            nn.Conv2d(expanded_channels, oup, 1, 1, 0, bias=False))
        layers.append(
            nn.BatchNorm2d(oup))
        
        
        self.conv = nn.Sequential(*layers)
        self.res = Residual(self.conv)
        
        
    def forward(self, x):
        if self.identity:
            return self.res(x)
        else:
            return self.conv(x)