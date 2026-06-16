"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- 原始论文版 Causal Conv --------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# -------------------------- 原始论文版 TCN Block --------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = self.drop1(F.elu(self.bn1(self.conv1(x))))
        x = self.drop2(F.elu(self.bn2(self.conv2(x))))
        return x + res


# -------------------------- 原始论文版 Attention --------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        res = x
        x, _ = self.attn(x, x, x)
        x = self.drop(x) + res
        x = self.norm(x)
        return x.permute(0, 2, 1)


# -------------------------- 原始论文版 ATCNet —— 速度超快 --------------------------
class ATCNet(nn.Module):
    def __init__(
            self,
            in_chans=32,
            n_classes=10,
            F1=16, D=2, kernLength=64,
            poolSize1=8, poolSize2=7,
            num_heads=2, depth=2,
            n_windows=5,
            dropout=0.2
    ):
        super().__init__()
        F2 = F1 * D
        self.n_windows = n_windows

        # 原始EEGNet编码器（轻量）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, (in_chans, 1), groups=F1, bias=False),
            nn.Conv2d(F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AvgPool2d((1, poolSize1)),
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, poolSize2)),
            nn.Dropout(dropout)
        )

        # 原始ATCNet：每个窗口独立 TCN + Attention
        self.window_blocks = nn.ModuleList()
        for _ in range(n_windows):
            self.window_blocks.append(nn.ModuleList([
                Attention(F2, num_heads, dropout),
                TCNBlock(F2, F2, 4, dilation=1, dropout=dropout),
                nn.Linear(F2, n_classes)
            ]))

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze(2)
        T = x.shape[-1]
        win_len = T // self.n_windows

        outs = []
        for i, (att, tcn, fc) in enumerate(self.window_blocks):
            st = i * win_len
            ed = st + win_len
            seg = x[:, :, st:ed]
            seg = att(seg)
            seg = tcn(seg)
            out = fc(seg[:, :, -1])
            outs.append(out)

        return torch.stack(outs).mean(0)