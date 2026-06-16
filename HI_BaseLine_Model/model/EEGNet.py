"""
作者：王帆
时间：2022. 9.8
"""
import torch
import math
from torch import nn
from torch import Tensor
from typing import Union, Tuple  # 新增导入


class EEGNetModel(nn.Module):
    """
    标准EEGNet模型（适配32通道、1000采样点手写想象EEG数据）
    支持返回中间特征，用于特征分析/可视化
    """
    def __init__(self, chans=32, classes=5, time_points=1000, temp_kernel=25,
                 f1=8, f2=16, d=2, pk1=16, pk2=8, dropout_rate=0.5,
                 max_norm1=1, max_norm2=1):
        super().__init__()
        # 计算全连接层输入维度
        linear_size = (time_points // (pk1 * pk2)) * f2

        # 时域卷积模块
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )

        # 深度卷积 + 池化模块
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )

        # 可分离卷积 + 池化模块
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # 权重最大范数约束
        self._apply_max_norm(self.block2[0], max_norm1)
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer: nn.Module, max_norm: float):
        """对卷积/全连接层权重施加L2最大范数约束"""
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x: Tensor, return_feature: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        前向传播
        :param x: 输入张量 shape [batch, 1, chans, time_points]
        :param return_feature: 是否返回卷积提取的扁平化特征
        :return: 分类输出 / (分类输出, 中间特征)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feature = self.flatten(x)
        output = self.fc(feature)

        if return_feature:
            return output, feature
        return output