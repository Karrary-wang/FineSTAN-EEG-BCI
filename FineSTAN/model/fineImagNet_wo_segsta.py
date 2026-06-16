import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def attention(query, key, value, dropout=None):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhqk,bhkd->bhqd', attn, value), attn


def compute_channel_importance(x, labels):
    B, C, T = x.shape
    var_importance = torch.zeros(C, device=x.device)

    unique_labels = torch.unique(labels)
    for c in range(C):
        class_vars = []
        for k in unique_labels:
            class_data = x[labels == k, c, :]
            var_k = class_data.var(dim=-1).mean()
            class_vars.append(var_k)
        var_importance[c] = torch.stack(class_vars).std()

    x_np = x.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    x_flat = x_np.transpose(0, 2, 1).reshape(B * T, C)
    labels_repeat = np.repeat(labels_np, T)
    mi_all = mutual_info_classif(x_flat, labels_repeat, random_state=42)
    mi_importance = torch.from_numpy(mi_all).float().to(x.device)

    var_norm = (var_importance - var_importance.min()) / (var_importance.max() - var_importance.min() + 1e-8)
    mi_norm = (mi_importance - mi_importance.min()) / (mi_importance.max() - mi_importance.min() + 1e-8)
    return 0.5 * var_norm + 0.5 * mi_norm


def select_motor_channels(importance, top_k=16):
    top_k_idx = torch.argsort(importance, descending=True)[:top_k]
    return top_k_idx


class DualPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-8
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=0, count_include_pad=False)

    def forward(self, x):
        B, D, T = x.shape
        pad_len = (self.kernel_size - (T % self.kernel_size)) % self.kernel_size
        x_padded = F.pad(x, (0, pad_len), mode='constant', value=0) if pad_len > 0 else x

        x_avg = self.pool(x_padded)
        x_var = torch.clamp(self.pool(x_padded ** 2) - x_avg ** 2, self.eps, 1e6)
        return x_avg, torch.log(x_var)


class MeanVarGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_linear = nn.Linear(embed_dim, embed_dim)
        self.temp_raw = nn.Parameter(torch.tensor(0.0))
        nn.init.constant_(self.gate_linear.weight, 0.1)
        nn.init.constant_(self.gate_linear.bias, 0.0)

    @property
    def temp(self):
        return F.softplus(self.temp_raw) + 0.1

    def forward(self, x_avg, x_var):
        gate_logits = self.gate_linear(x_avg.transpose(1, 2))
        gate = torch.sigmoid(self.temp * gate_logits).transpose(1, 2)
        return x_avg, x_var * gate


class MotorAttention(nn.Module):
    def __init__(self, selected_channel_idx):
        super().__init__()
        self.register_buffer('selected_idx', torch.tensor(selected_channel_idx, dtype=torch.long))

    def forward(self, x):
        return x[:, :, self.selected_idx, :]


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


# ====================== 消融实验3：移除 SegSTA 时序注意力 ======================
class FineImagNet_wo_SegSTA(nn.Module):
    def __init__(self, num_classes=10, num_samples=1000, num_channels=32, embed_dim=32, pool_size=180,
                 pool_stride=30, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.3, fc_drop=0.3,
                 classify_drop=0.5, pool_dropout=0.3, attn_norm_type="softmax", temp_coeff_init=0.5,
                 global_weight_init=0.3, selected_channel_idx=None, window_size=100, stride=50, seg_fusion_init=0.5):
        super().__init__()
        assert embed_dim % 4 == 0
        assert selected_channel_idx is not None

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.selected_channel_num = len(selected_channel_idx)

        # ===================== 保留 MSTCM 多尺度卷积 =====================
        self.kernel_sizes = [7, 15, 31, 65]
        self.temp_conv = nn.ModuleList([
            nn.Conv2d(1, embed_dim // 4, (1, k), padding=(0, k // 2))
            for k in self.kernel_sizes
        ])
        self.scale_logits = nn.Parameter(torch.tensor([2, 1.5, 1.0, 1]))

        self.temp_bn = nn.BatchNorm2d(embed_dim)
        self.instance_norm = nn.InstanceNorm2d(embed_dim)

        # ===================== 保留 ACSM 通道筛选 =====================
        self.motor_attention = MotorAttention(selected_channel_idx)

        # 空间卷积
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.selected_channel_num, 1))
        self.spatial_bn = nn.BatchNorm2d(embed_dim)
        self.spatial_act = nn.ELU()

        # ===================== 移除 SegSTA，仅保留偏置项 =====================
        self.pooled_avg_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pooled_var_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))

        # ===================== 保留 DualStat 模块 =====================
        self.dual_pool = DualPool(pool_size, pool_stride)
        self.mean_var_gate = MeanVarGate(embed_dim)

        # ===================== 完全保留 Transformer 主干 =====================
        self.transformers = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop)
            for _ in range(depth)
        ])

        self.pool_drop = nn.Dropout(pool_dropout) if pool_dropout > 0 else nn.Identity()
        self.cls_drop = nn.Dropout(classify_drop)

        self._calc_cls_dim()

    def _pre_attention_features(self, x):
        x = x.unsqueeze(1)
        scale_weights = F.softmax(self.scale_logits, dim=0)
        scaled_features = [conv(x) * w for conv, w in zip(self.temp_conv, scale_weights)]
        x = torch.cat(scaled_features, dim=1)
        x = self.temp_bn(x)
        x = self.instance_norm(x)
        return x

    def _extract_features(self, x):
        x = self._pre_attention_features(x)
        x = self.motor_attention(x)
        x = self.spatial_act(self.spatial_bn(self.spatial_conv(x)))
        x = x.squeeze(dim=2)
        # ===================== 核心改动：移除 SegSTA 融合 =====================
        return x

    def _calc_cls_dim(self):
        with torch.no_grad():
            current_device = next(self.parameters()).device
            current_dtype = next(self.parameters()).dtype
            dummy = torch.randn(1, self.num_channels, self.num_samples, device=current_device, dtype=current_dtype)

            x = self._extract_features(dummy)
            x_avg, x_var = self.dual_pool(x)
            x_avg, x_var = self.mean_var_gate(x_avg, x_var)

            x_avg = x_avg + self.pooled_avg_bias
            x_var = x_var + self.pooled_var_bias

            x_avg = rearrange(self.pool_drop(x_avg), 'b d n -> b n d')
            x_var = rearrange(self.pool_drop(x_var), 'b d n -> b n d')
            x_avg = self.transformers(x_avg)
            x_var = self.transformers(x_var)

            x_cat = torch.cat([x_avg, x_var], dim=-1)
            cls_in_dim = x_cat.flatten(1).shape[1]

        self.classify = nn.Linear(cls_in_dim, self.num_classes).to(current_device)

    def forward(self, x):
        x = x.to(next(self.parameters()).device, next(self.parameters()).dtype)
        x = self._extract_features(x)

        x_avg, x_var = self.dual_pool(x)
        x_avg, x_var = self.mean_var_gate(x_avg, x_var)

        x_avg = x_avg + self.pooled_avg_bias
        x_var = x_var + self.pooled_var_bias

        x_avg = rearrange(self.pool_drop(x_avg), 'b d n -> b n d')
        x_var = rearrange(self.pool_drop(x_var), 'b d n -> b n d')
        x_avg = self.transformers(x_avg)
        x_var = self.transformers(x_var)

        x_cat = torch.cat([x_avg, x_var], dim=-1)
        x_flat = x_cat.flatten(1)
        return self.classify(self.cls_drop(x_flat))


# ===================== 测试代码 =====================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dummy_data = torch.randn(100, 32, 1000).to(device)
    dummy_labels = torch.randint(0, 5, (100,)).to(device)

    importance = compute_channel_importance(dummy_data, dummy_labels)
    selected_idx = select_motor_channels(importance, top_k=16)

    model = FineImagNet_wo_SegSTA(
        num_classes=5,
        num_samples=1000,
        selected_channel_idx=selected_idx,
    ).to(device)

    test_input = torch.randn(8, 32, 1000).to(device)
    test_output = model(test_input)
    print("输入 shape:", test_input.shape)
    print("输出 shape:", test_output.shape)
    print("✅ 消融模型3 w/o_SegSTA 运行成功！")
