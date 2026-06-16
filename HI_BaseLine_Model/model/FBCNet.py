"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 带权重约束的卷积层
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


# 带权重约束的线性层
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


# 时间聚合层：方差
class VarLayer(nn.Module):
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


# 时间聚合层：对数方差
class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


# 时间聚合层：均值
class MeanLayer(nn.Module):
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


# 时间聚合层映射字典
temporal_layer = {
    'VarLayer': VarLayer,
    'LogVarLayer': LogVarLayer,
    'MeanLayer': MeanLayer
}


# Swish 激活函数
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNetModel(nn.Module):
    def __init__(self,
                 nChan=32,
                 nTime=1000,
                 nClass=10,
                 nBands=10,
                 m=16,
                 temporalLayer='LogVarLayer',
                 strideFactor=4,
                 doWeightNorm=True):
        super(FBCNetModel, self).__init__()
        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # 空间卷积块 SCB
        self.scb = self._build_SCB(m, nChan, nBands, doWeightNorm)
        # 时间聚合层
        self.temporalLayer = temporal_layer[temporalLayer](dim=3)
        # 最终分类层
        self.lastLayer = self._build_LastBlock(
            inF=m * nBands * strideFactor,
            outF=nClass,
            doWeightNorm=doWeightNorm
        )

    def _build_SCB(self, m, nChan, nBands, doWeightNorm):
        return nn.Sequential(
            Conv2dWithConstraint(
                in_channels=nBands,
                out_channels=m * nBands,
                kernel_size=(nChan, 1),
                groups=nBands,
                max_norm=2,
                doWeightNorm=doWeightNorm,
                padding=0
            ),
            nn.BatchNorm2d(m * nBands),
            nn.SiLU()
        )

    def _build_LastBlock(self, inF, outF, doWeightNorm):
        return nn.Sequential(
            LinearWithConstraint(
                in_features=inF,
                out_features=outF,
                max_norm=0.5,
                doWeightNorm=doWeightNorm
            ),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 输入形状: [batch, 1, nChan, nTime, nBands]
        x = x.permute((0, 4, 2, 3, 1))
        x = x.squeeze(-1)

        # 空间滤波
        x = self.scb(x)

        # 时间分块补齐
        pad_length = x.shape[3] % self.strideFactor
        if pad_length != 0:
            x = F.pad(x, (0, pad_length))
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])

        # 时间聚合
        x = self.temporalLayer(x)

        # 展平 + 分类
        x = torch.flatten(x, start_dim=1)
        x = self.lastLayer(x)
        return x