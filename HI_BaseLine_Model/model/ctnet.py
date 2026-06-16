"""
作者：王帆
时间：2022. 9.8
"""
import torch
import math
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange


class PatchEmbeddingCNN(nn.Module):
    """将EEG数据通过CNN映射为Transformer可处理的patch嵌入特征
    适配：32通道、1000采样点手写想象EEG数据
    """

    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=32, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # 时间维度卷积（捕捉时序特征）
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            # 深度可分离卷积（通道维度，适配32通道）
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # 池化1（时间维度降维）
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            # 空间卷积
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # 池化2（调整特征长度适配Transformer）
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        # 维度映射到emb_size
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
            nn.Linear(f2, emb_size)  # 新增：确保输出维度等于emb_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制（捕捉EEG时序长程依赖）"""

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 确保emb_size能被num_heads整除
        assert emb_size % num_heads == 0, f"emb_size {emb_size} 不能被num_heads {num_heads} 整除"

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 维度重排：(b, n, emb) -> (b, h, n, d)
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # 注意力分数计算
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # 缩放+Softmax（防止梯度消失）
        scaling = self.emb_size ** (1 / 2)
        att = nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # 注意力加权+维度还原
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    """Transformer前馈网络块（增强特征表达）"""

    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    """残差连接+层归一化（缓解梯度消失）"""

    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    """单个Transformer编码器块"""

    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ), emb_size, drop_p)
        )


class TransformerEncoder(nn.Sequential):
    """堆叠多个Transformer编码器块"""

    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class BranchEEGNetTransformer(nn.Sequential):
    """CNN+Transformer分支（核心特征提取）
    默认参数适配32通道手写想象数据
    """

    def __init__(self, heads=4, depth=6, emb_size=40, number_channel=32,
                 f1=20, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, kernel_size=kernel_size, D=D,
                              pooling_size1=pooling_size1, pooling_size2=pooling_size2,
                              dropout_rate=dropout_rate, number_channel=number_channel, emb_size=emb_size),
            TransformerEncoder(heads, depth, emb_size)  # 补充：原代码漏加Transformer编码器
        )


class PositioinalEncoding(nn.Module):
    """可学习位置编码（捕捉EEG时序位置信息）"""

    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        # 适配不同设备（CPU/GPU）
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class ClassificationHead(nn.Sequential):
    """分类头（适配多任务类别数）"""

    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class CTNet(nn.Module):
    """CTNet主模型（适配32通道手写想象EEG数据）
    参数说明：
        n_classes: 分类类别数（笔画5/拼音6/字母26/混合37）
        number_channel: 电极通道数（固定32）
        heads: Transformer注意力头数
        emb_size: 嵌入维度
        depth: Transformer编码器层数
        flatten_eeg1: 分类头输入维度（seq_len × emb_size）
    """

    def __init__(self, n_classes, number_channel=32, heads=4, emb_size=40, depth=6,
                 eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8, eeg1_dropout_rate=0.3,
                 flatten_eeg1=600):
        super().__init__()
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1

        # 验证维度匹配
        assert flatten_eeg1 % emb_size == 0, f"flatten_eeg1 {flatten_eeg1} 必须是emb_size {emb_size} 的整数倍"

        # CNN+Transformer特征提取分支（适配32通道）
        self.cnn = BranchEEGNetTransformer(
            heads=heads, depth=depth, emb_size=emb_size, number_channel=number_channel,
            f1=eeg1_f1, kernel_size=eeg1_kernel_size, D=eeg1_D,
            pooling_size1=eeg1_pooling_size1, pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate
        )

        # 位置编码（动态传入emb_size）
        self.position = PositioinalEncoding(embedding=emb_size, dropout=0.1)

        # 扁平化+分类头
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1, n_classes)

    def forward(self, x):
        # x shape: (batch, 1, 32, 1000) → (批次, 卷积通道, 电极通道, 时间采样点)
        cnn_feat = self.cnn(x)  # (batch, seq_len, emb_size)

        # 位置编码（增强时序信息）
        cnn_feat = cnn_feat * math.sqrt(self.emb_size)
        cnn_feat = self.position(cnn_feat)

        # 分类输出
        out = self.classification(self.flatten(cnn_feat))
        return cnn_feat, out