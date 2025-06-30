import os
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir))
sys.path.append(os.path.join(current_dir, 'RADIO'))
sys.path.append(os.path.join(current_dir, 'mamba'))
from Lora_Radio import Lora_Radio_model
from vmamba import VSSBlock, Permute
from GrootV.classification.models.grootv import GrootV, GrootV_3D, GrootVLayer, GrootV3DLayer

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class STM_block(nn.Module):
    def __init__(self, inchannel=768, out_channels=128):
        super(STM_block, self).__init__()
        self.inchannel = inchannel
        self.conv2 = nn.Conv2d(inchannel, out_channels, kernel_size=1)
        self.GrootV_S1 = GrootV3DLayer(channels=out_channels)
        self.smooth_layer_x = ResBlock(out_channels, out_channels)
        self.smooth_layer_y = ResBlock(out_channels, out_channels)

    def forward(self, x, y):
        B, C, H, W = x.size()

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]

        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm

class RadioMambaCD(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.radio = Lora_Radio_model(r=16)
        self.adapter = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stm = STM(
            encoder_dims=64, 
            hidden_dim=64,
            norm_layer=nn.LayerNorm,
            ssm_act_layer=nn.SiLU,
            mlp_act_layer=nn.GELU
        )
        self.change_head = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        initialize_weights(self.adapter, self.stm, self.change_head)

    def extract_features(self, x):
        """提取图像特征"""
        B, C, H, W = x.shape
        embeddings = self.radio(x)[1]  # 使用Radio的视觉特征输出
        return self.adapter(
            embeddings.transpose(1, 2).reshape(B, -1, H//16, W//16)
        )

    def forward(self, x1, x2):
        # 提取双时相特征
        f1 = self.extract_features(x1)  # [B, 64, H/16, W/16]
        f2 = self.extract_features(x2)
        
        # Mamba时空建模
        f1_proc, f2_proc = self.stm(f1, f2)
        
        fuse = torch.cat([f1_proc, f2_proc], dim=1)
        change_map = self.change_head(fuse)
        
        # 上采样到原始尺寸
        return F.interpolate(change_map, size=x1.shape[-2:], mode='bilinear', align_corners=True)

class RadioMamba_v2(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.radio = Lora_Radio_model(r=16)
        self.transit = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.MambaLayer = STM_block(256, 128)

        self.change_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, 1, 1, bias=True),
            # relu后面不用
        )
        initialize_weights(self.transit, self.MambaLayer, self.change_head)

    def extract_features(self, x):
        """提取图像特征"""
        B, C, H, W = x.shape
        embeddings = self.radio(x)[1]  # 使用Radio的视觉特征输出
        embeddings = embeddings.transpose(1, 2).reshape(B, -1, H//16, W//16)
        return self.transit(embeddings)

    def forward(self, x1, x2):
        input_size = size=x1.shape[-2:]
        #x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        #x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')

        # 提取双时相特征
        f1 = self.extract_features(x1)  # [B, 64, H/16, W/16]
        f2 = self.extract_features(x2)
        
        # Mamba时空建模
        f1_proc, f2_proc = self.MambaLayer(f1, f2)
        
        fuse = torch.cat([f1_proc, f2_proc], dim=1)
        change_map = self.change_head(fuse)
        
        # 上采样到原始尺寸
        return F.interpolate(change_map, size=input_size, mode='bilinear')

if __name__ == "__main__":
    model = RadioMamba_v2().cuda()
    x1 = torch.randn(2, 3, 512, 512).cuda()
    x2 = torch.randn(2, 3, 512, 512).cuda()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数量: 共 {total_params/1e6:.2f}M | 可训练 {train_params/1e6:.2f}M")
    
    # 测试前向传播
    with torch.no_grad():
        out = model(x1, x2)
        print(f"输入尺寸: {x1.shape} -> 输出尺寸: {out.shape}")
        print(f"预测变化图范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
