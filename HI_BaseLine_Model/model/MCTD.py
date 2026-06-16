"""
作者：王帆
时间：2022. 9.8
"""
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn


# 固定随机种子（保证可复现）
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


set_seed(0)


# -------------------------- 局部特征提取模块（适配32通道+1000采样点） --------------------------
class Localfeature(nn.Module):
    def __init__(self, emb_size, kernel, in_chans=32):
        super().__init__()
        # ShallowNet：适配32通道EEG，250Hz采样率1000采样点
        self.shallownet = nn.Sequential(
            # 时间维度卷积：1输入通道，40卷积核，核大小(1, kernel)
            nn.Conv2d(1, 40, (1, kernel), (1, 1), padding='same'),
            # 空间维度卷积：40输入通道，40卷积核，核大小(32, 1)（匹配32通道）
            nn.Conv2d(40, 40, (in_chans, 1), (1, 1), padding='valid'),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # 池化：适配1000采样点（250Hz×4秒），池化核(1, 20)，步长(1, 10)
            nn.AvgPool2d((1, 20), (1, 10)),
            nn.Dropout(0.5),
        )

        # 特征投影：将40通道特征映射到emb_size维度，并调整维度格式
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # [batch, emb, 1, time] → [batch, time, emb]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x输入维度：[batch, 1, 32, 1000]
        x = self.shallownet(x)  # [batch, 40, 1, time]
        x = self.projection(x)  # [batch, time, emb_size]
        return x


# -------------------------- 多头注意力模块 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 确保emb_size能被num_heads整除
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: [batch, seq_len, emb_size]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # 注意力分数计算
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # 注意力归一化
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # 注意力加权求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# -------------------------- 残差连接模块 --------------------------
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# -------------------------- 前馈网络模块 --------------------------
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


# -------------------------- Transformer编码器模块 --------------------------
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,  # 适配emb_size=40 → 40/8=5，整除
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


# -------------------------- 全局特征提取模块 --------------------------
class GlobalFeature(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


# -------------------------- 分类头模块（动态适配特征维度） --------------------------
class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # 移除预设的feature_size，改为动态初始化
        self.emb_size = emb_size
        self.n_classes = n_classes
        # 降维层先不初始化，前向传播时动态创建
        self.reducedim = None
        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: [batch, feature_size]
        # 动态初始化降维层（仅第一次前向传播时执行）
        if self.reducedim is None:
            feature_size = x.shape[1]
            self.reducedim = nn.Linear(feature_size, 256).to(x.device)
            print(f" 动态初始化分类头降维层：{feature_size} → 256")

        feat = self.reducedim(x)
        out = self.fc(feat)
        return feat, out


# -------------------------- 局部-全局特征融合模块 --------------------------
class LocalGlobalFeature(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, kernel=4, in_chans=32):
        super().__init__(
            Localfeature(emb_size, kernel, in_chans),
            GlobalFeature(depth, emb_size),
        )


# -------------------------- 核心MCTD模型（多尺度卷积Transformer） --------------------------
class MCTD(nn.Module):
    def __init__(self,
                 in_chans=32,  # 32通道EEG
                 emb_size=40,  # 特征嵌入维度
                 n_classes=10,  # 手写想象10分类（0-9）
                 depth=6):  # Transformer编码器层数
        super().__init__()

        # 多尺度卷积核（适配250Hz采样率，覆盖不同时间尺度）
        self.kernel_sizes = [4, 8, 16, 32, 64]

        # 多尺度特征提取器（每个卷积核对应一个分支）
        self.conformers = nn.ModuleList([
            LocalGlobalFeature(emb_size=emb_size, kernel=kernel_size, in_chans=in_chans)
            for kernel_size in self.kernel_sizes
        ])

        # 分类头：移除预设的feature_size，仅传emb_size和n_classes
        self.cls_heads = nn.ModuleList([
            ClassificationHead(emb_size, n_classes) for _ in self.kernel_sizes
        ])

    def forward(self, x):
        # x输入维度：[batch, 1, 32, 1000]
        # 1. 多尺度特征提取
        x_outs = [conformer(x) for conformer in self.conformers]  # 每个分支：[batch, time, emb_size]

        # 2. 展平特征（动态适配time维度）
        x_outs = [x.contiguous().view(x.size(0), -1) for x in x_outs]  # [batch, time×emb_size]

        # 3. 多尺度分类（分类头动态初始化）
        feats = []  # 保存每个尺度的特征（用于融合）
        outputs = []  # 保存每个尺度的分类输出

        for x_out, head in zip(x_outs, self.cls_heads):
            feat, output = head(x_out)
            feats.append(feat)
            outputs.append(output)

        # 返回：多尺度特征列表 + 多尺度输出列表
        return feats, outputs


# -------------------------- 模型测试代码（验证输入输出维度） --------------------------
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型（适配32通道、6分类）
    model = MCTD(in_chans=32, n_classes=6).to(device)

    # 模拟输入：batch_size=16, 1通道, 32电极, 1000采样点
    dummy_input = torch.randn(16, 1, 32, 1000).to(device)

    # 前向传播（触发分类头动态初始化）
    feats, outputs = model(dummy_input)

    # 打印输出维度
    print("=" * 50)
    print("模型输入维度：", dummy_input.shape)
    print("=" * 50)
    for i, (feat, output) in enumerate(zip(feats, outputs)):
        print(f"尺度{i + 1}（核大小{model.kernel_sizes[i]}）：")
        print(f"  特征维度：{feat.shape}")
        print(f"  输出维度：{output.shape}")
    print("=" * 50)
    print("模型参数总数：{:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

