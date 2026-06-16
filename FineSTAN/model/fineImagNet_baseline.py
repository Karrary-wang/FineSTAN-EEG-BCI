import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np


def attention(query, key, value, dropout=None):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhqk,bhkd->bhqd', attn, value), attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, N, D = x.shape
        position = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / D))

        pe = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_k)
        self.w_o = nn.Linear(n_head * self.d_k, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = rearrange(self.w_q(x), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(x), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(x), "b n (h d) -> b h n d", h=self.n_head)

        out, _ = attention(q, k, v, dropout=self.dropout_attn)
        out = rearrange(out, 'b h q d -> b q (h d)')
        return self.dropout(self.w_o(out))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.3, fc_drop=0.3):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim, num_heads, attn_drop)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * fc_ratio),
            nn.GELU(),
            nn.Dropout(fc_drop),
            nn.Linear(embed_dim * fc_ratio, embed_dim),
            nn.Dropout(fc_drop)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pos_enc(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ====================== 基线模型：移除所有4个创新模块 ======================
class FineImagNet_Baseline(nn.Module):
    # 👇 仅此处添加 selected_channel_idx=None，修复传参报错
    def __init__(self, num_classes=10, num_samples=1000, num_channels=32, embed_dim=32, pool_size=180,
                 pool_stride=30, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.3, fc_drop=0.3,
                 classify_drop=0.5, pool_dropout=0.3, selected_channel_idx=None):
        super().__init__()
        assert embed_dim % 4 == 0

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # ========== 1. 替换MSTCM：单尺度1×1卷积（无多尺度创新）==========
        self.input_conv = nn.Conv2d(1, embed_dim, (1, 1), padding=(0, 0))
        self.input_bn = nn.BatchNorm2d(embed_dim)
        self.input_act = nn.ELU()

        # ========== 2. 移除ACSM：不做通道选择，使用全部32通道 ==========
        # 空间卷积适配全部通道
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.num_channels, 1))
        self.spatial_bn = nn.BatchNorm2d(embed_dim)
        self.spatial_act = nn.ELU()

        # ========== 3. 移除SegSTA：无任何时序注意力操作 ==========
        # 仅保留偏置项（与完整模型对齐，保证维度一致）
        self.pooled_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))

        # ========== 4. 替换DSGM：单路普通平均池化（无双统计/门控创新）==========
        self.pool = nn.AvgPool1d(pool_size, pool_stride, padding=0, count_include_pad=False)
        self.eps = 1e-8

        # ========== 主干Transformer完全保留（与完整模型一致）==========
        self.transformers = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop)
            for _ in range(depth)
        ])

        self.pool_drop = nn.Dropout(pool_dropout) if pool_dropout > 0 else nn.Identity()
        self.cls_drop = nn.Dropout(classify_drop)

        self._calc_cls_dim()

    def _pre_attention_features(self, x):
        # 单尺度1×1卷积，无多尺度
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_act(x)
        return x

    def _extract_features(self, x):
        x = self._pre_attention_features(x)
        # 移除ACSM，直接用全通道
        x = self.spatial_act(self.spatial_bn(self.spatial_conv(x)))
        x = x.squeeze(dim=2)
        # 移除SegSTA，直接返回原始特征
        return x

    def _calc_cls_dim(self):
        with torch.no_grad():
            current_device = next(self.parameters()).device
            current_dtype = next(self.parameters()).dtype
            dummy = torch.randn(1, self.num_channels, self.num_samples, device=current_device, dtype=current_dtype)

            x = self._extract_features(dummy)
            B, D, T = x.shape
            pad_len = (self.pool_size - (T % self.pool_size)) % self.pool_size
            x_padded = F.pad(x, (0, pad_len), mode='constant', value=0) if pad_len > 0 else x
            x_pool = self.pool(x_padded)
            x_pool = x_pool + self.pooled_bias

            x_pool = rearrange(self.pool_drop(x_pool), 'b d n -> b n d')
            x_trans = self.transformers(x_pool)
            cls_in_dim = x_trans.flatten(1).shape[1]

        self.classify = nn.Linear(cls_in_dim, self.num_classes).to(current_device)

    def forward(self, x):
        x = x.to(next(self.parameters()).device, next(self.parameters()).dtype)
        x = self._extract_features(x)

        B, D, T = x.shape
        pad_len = (self.pool_size - (T % self.pool_size)) % self.pool_size
        x_padded = F.pad(x, (0, pad_len), mode='constant', value=0) if pad_len > 0 else x
        x_pool = self.pool(x_padded)
        x_pool = x_pool + self.pooled_bias

        x_pool = rearrange(self.pool_drop(x_pool), 'b d n -> b n d')
        x_trans = self.transformers(x_pool)
        x_flat = x_trans.flatten(1)
        return self.classify(self.cls_drop(x_flat))

# ====================== 测试代码（可直接运行，与完整模型对齐）======================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = FineImagNet_Baseline(
        num_classes=5,
        num_samples=1000,
    ).to(device)

    test_input = torch.randn(8, 32, 1000).to(device)
    test_output = model(test_input)

    print("输入 shape:", test_input.shape)
    print("输出 shape:", test_output.shape)
    print("基线模型 FineImagNet_Baseline 运行成功！")