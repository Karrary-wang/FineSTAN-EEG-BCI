"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from einops import rearrange
from ptflops import get_model_complexity_info


# -------------------------- 基础算子 & 模块定义 --------------------------
def attention(query, key, value, dropout=None):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhqk,bhkd->bhqd', attn, value), attn


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


class SegmentedTemporalAttention(nn.Module):
    def __init__(self, embed_dim, window_size=100, stride=50, num_samples=1000, norm_type="softmax", dropout=0.2,
                 temp_coeff_init=0.5, global_weight_init=0.3, seg_fusion_init=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_type = norm_type
        self.window_size = window_size
        self.stride = stride if stride is not None else self.window_size // 2

        self.local_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        self.seg_fusion_raw = nn.Parameter(torch.tensor(math.log(seg_fusion_init / (1 - seg_fusion_init))))
        self.pooled_avg_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pooled_var_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.temp_coeff = nn.Parameter(torch.tensor(temp_coeff_init))
        self.global_weight_raw = nn.Parameter(torch.tensor(math.log(global_weight_init / (1 - global_weight_init))))

    def _local_window_attention(self, x_trans):
        B, T, D = x_trans.shape
        pad_len = (self.stride - (T % self.stride)) % self.stride
        x_padded = F.pad(x_trans, (0, 0, 0, pad_len), mode='constant', value=0) if pad_len > 0 else x_trans
        T_pad = x_padded.shape[1]

        x_unfolded = F.unfold(
            x_padded.transpose(1, 2).unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            stride=(self.stride, 1)
        )
        x_windowed = x_unfolded.view(B, D, self.window_size, -1).permute(0, 3, 2, 1)
        B, N, W, D = x_windowed.shape

        local_attn_flat = self.local_mlp(x_windowed.flatten(0, 2))
        local_attn = local_attn_flat.view(B, N, W, 1)

        if pad_len > 0:
            last_window_valid = W - pad_len
            local_attn[:, -1, last_window_valid:, :] = -float('inf')

        safe_temp = F.softplus(self.temp_coeff) + 0.1
        local_attn = F.softmax(local_attn / safe_temp, dim=2)

        attn_unfolded = local_attn.permute(0, 3, 2, 1).flatten(1, 2)
        count_unfolded = torch.ones_like(attn_unfolded)

        attn_sum = F.fold(
            attn_unfolded,
            output_size=(T_pad, 1),
            kernel_size=(self.window_size, 1),
            stride=(self.stride, 1)
        ).squeeze(-1)

        count_folded = F.fold(
            count_unfolded,
            output_size=(T_pad, 1),
            kernel_size=(self.window_size, 1),
            stride=(self.stride, 1)
        ).squeeze(-1)

        attn_recon = attn_sum[:, :, :T] / count_folded[:, :, :T].clamp(min=1e-8)
        return attn_recon.transpose(1, 2)

    def _global_attention(self, x_trans):
        B, T, D = x_trans.shape
        global_attn = self.global_mlp(x_trans)

        if self.norm_type == "softmax":
            safe_temp = F.softplus(self.temp_coeff) + 0.1
            global_attn = F.softmax(global_attn / safe_temp, dim=1)
            global_weight = torch.sigmoid(self.global_weight_raw)
            global_avg_attn = torch.ones_like(global_attn) / T
            global_attn = (1 - global_weight) * global_attn + global_weight * global_avg_attn
        else:
            global_attn = torch.sigmoid(global_attn)
        return global_attn

    def forward(self, x):
        B, D, T = x.shape
        x_trans = x.permute(0, 2, 1)

        local_attn = self._local_window_attention(x_trans)
        local_out = x_trans * local_attn

        global_attn = self._global_attention(x_trans)
        global_out = x_trans * global_attn

        seg_fusion = torch.sigmoid(self.seg_fusion_raw)
        fused_out = seg_fusion * local_out + (1 - seg_fusion) * global_out

        return fused_out.permute(0, 2, 1)

    def get_segmented_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)
            B, D, T = x.shape
            x_trans = x.permute(0, 2, 1)
            local_attn = self._local_window_attention(x_trans)
            global_attn = self._global_attention(x_trans)
            return local_attn.squeeze(-1).detach().cpu().numpy(), global_attn.squeeze(-1).detach().cpu().numpy()


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


class FineImagNet_V3_New(nn.Module):
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

        self.kernel_sizes = [7, 15, 31, 65]
        self.temp_conv = nn.ModuleList([
            nn.Conv2d(1, embed_dim // 4, (1, k), padding=(0, k // 2))
            for k in self.kernel_sizes
        ])
        self.scale_logits = nn.Parameter(torch.tensor([2, 1.5, 1.0, 1]))

        self.temp_bn = nn.BatchNorm2d(embed_dim)
        self.instance_norm = nn.InstanceNorm2d(embed_dim)

        self.motor_attention = MotorAttention(selected_channel_idx)
        self.temporal_attn = SegmentedTemporalAttention(
            embed_dim, window_size=window_size, stride=stride, num_samples=num_samples, norm_type=attn_norm_type,
            temp_coeff_init=temp_coeff_init, global_weight_init=global_weight_init, seg_fusion_init=seg_fusion_init
        )

        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.selected_channel_num, 1))
        self.spatial_bn = nn.BatchNorm2d(embed_dim)
        self.spatial_act = nn.ELU()

        self.dual_pool = DualPool(pool_size, pool_stride)
        self.mean_var_gate = MeanVarGate(embed_dim)

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
        x = x + self.temporal_attn(x)
        return x

    def _calc_cls_dim(self):
        with torch.no_grad():
            current_device = next(self.parameters()).device
            current_dtype = next(self.parameters()).dtype
            dummy = torch.randn(1, self.num_channels, self.num_samples, device=current_device, dtype=current_dtype)

            for param in self.parameters():
                param.data = param.data.to(current_device)

            x = self._extract_features(dummy)
            x_avg, x_var = self.dual_pool(x)
            x_avg, x_var = self.mean_var_gate(x_avg, x_var)

            x_avg = x_avg + self.temporal_attn.pooled_avg_bias
            x_var = x_var + self.temporal_attn.pooled_var_bias

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

        x_avg = x_avg + self.temporal_attn.pooled_avg_bias
        x_var = x_var + self.temporal_attn.pooled_var_bias

        x_avg = rearrange(self.pool_drop(x_avg), 'b d n -> b n d')
        x_var = rearrange(self.pool_drop(x_var), 'b d n -> b n d')
        x_avg = self.transformers(x_avg)
        x_var = self.transformers(x_var)

        x_cat = torch.cat([x_avg, x_var], dim=-1)
        x_flat = x_cat.flatten(1)
        return self.classify(self.cls_drop(x_flat))


# -------------------------- 统计工具函数 --------------------------
def count_trainable_params(model):

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数量: {total_params:,}")
    print(f"参数量 (K): {total_params / 1000:.2f} K\n")
    return total_params


def get_model_flops(model, input_shape):

    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    flops = macs * 2  # MACs -> FLOPs 换算
    print(f"FLOPs (M): {flops / 1e6:.2f} M\n")
    return flops


def measure_inference_time(model, input_tensor, warmup=50, repeat=1000):

    for _ in range(warmup):
        _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeat):
        _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / repeat * 1000
    print(f"单样本平均推理时间: {avg_ms:.4f} ms")
    return avg_ms


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}\n")

    # 固定16个选中通道，结果稳定
    selected_idx = list(range(16))

    # 模型初始化：ELL-HI 10分类 统一基准
    model = FineImagNet_V3_New(
        num_classes=10,
        num_samples=1000,
        num_channels=32,
        embed_dim=32,
        selected_channel_idx=selected_idx,
        window_size=100,
        stride=50
    ).to(device)
    model.eval()

    # 输入 shape: (C, T) = (32, 1000)  对应 batch=1
    input_shape = (32, 1000)
    test_input = torch.randn(1, 32, 1000).to(device)

    print("===== 模型复杂度统计结果 =====")
    count_trainable_params(model)
    get_model_flops(model, input_shape)
    measure_inference_time(model, test_input)
    print("==============================")
"""
"""
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 数据（ELL-HI任务） --------------------------
models = [
    'ShallowConvNet', 'DeepConvNet', 'EEGNet', 'FBCNet',
    'EA-EEG', 'ATCNet', 'EEGConformer', 'MCTD',
    'SATrans-Net', 'CTNet', 'TransNet', 'FineSTAN (Ours)'
]
acc_ell = [48.77, 45.16, 49.46, 44.75, 50.29, 49.21, 51.55, 43.09, 51.99, 52.73, 49.84, 60.70]
params_k = [76.73, 391.51, 2.43, 12.01, 76.65, 83.60, 776.79, 5944.49, 13.71, 158.33, 113.90, 90.40]
flops_m = [165.19, 135.59, 15.12, 0.96, 74.06, 74.66, 176.99, 980.10, 70.41, 97.64, 166.32, 111.95]

# -------------------------- 2. 全局字体放大配置 --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13        # 基础全局字体放大
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# 气泡大小：整体放大一倍
bubble_sizes = (np.array(flops_m) / max(flops_m)) * 3000 + 200

sc = ax.scatter(
    params_k, acc_ell,
    s=bubble_sizes,
    c=flops_m,
    cmap='viridis',
    alpha=0.85,
    edgecolors='black',
    linewidths=0.8
)

# -------------------------- 3. 标签位置（完全保留你之前调好的位置） --------------------------
offsets = [
    (0.25, -0.8),   # 0 ShallowConvNet
    (0.15, 0.3),    # 1 DeepConvNet
    (0.15, 0.2),    # 2 EEGNet
    (0.15, 0.2),    # 3 FBCNet
    (0.05, 0.7),    # 4 EA-EEG
    (0.30, -0.3),   # 5 ATCNet
    (0.20, 0.2),    # 6 EEGConformer
    (-0.35, 0.15),  # 7 MCTD
    (0.20, 0.2),    # 8 SATrans-Net
    (0.20, 0.2),    # 9 CTNet
    (0.30, 0.4),    # 10 TransNet
    (0.15, 0.3)     # 11 FineSTAN (Ours)
]

for i, model in enumerate(models):
    dx_ratio, dy = offsets[i]
    x_pos = params_k[i] * (1 + dx_ratio)
    y_pos = acc_ell[i] + dy

    if model == 'FineSTAN (Ours)':
        ax.text(
            x_pos, y_pos, model,
            fontsize=12, fontweight='bold', color='#c0392b'  # 本文方法字体放大
        )
    else:
        ax.text(
            x_pos, y_pos, model,
            fontsize=11  # 普通模型名称字体放大
        )

# -------------------------- 4. 坐标轴、标题（字体统一放大） --------------------------
ax.set_xscale('log')
ax.set_xlabel('Parameters (K)', fontsize=15, fontweight='medium')
ax.set_ylabel('ELL-HI Accuracy (%)', fontsize=15, fontweight='medium')
ax.set_title('Accuracy vs. Model Size (Bubble Size/Color: FLOPs)', fontsize=16, pad=12)

ax.grid(True, which='both', axis='both')
ax.set_ylim([42, 63.5])
ax.set_xlim([1, 10000])

# 坐标轴刻度字体放大
ax.tick_params(axis='both', labelsize=12)

# -------------------------- 5. 色条（字体放大） --------------------------
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('FLOPs (Million)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# -------------------------- 6. 保存 --------------------------
plt.tight_layout()
plt.savefig('accuracy_complexity_tradeoff.png', dpi=900, bbox_inches='tight')
plt.show()
"""

