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


class FineSTAN(nn.Module):
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

    def get_scale_weights(self):
        return F.softmax(self.scale_logits, dim=0).detach().cpu().numpy()

    def get_mean_var_gate_weights(self):
        return self.mean_var_gate.gate_linear.weight.detach().cpu().numpy()

    def get_mean_var_gate_bias(self):
        return self.mean_var_gate.gate_linear.bias.detach().cpu().numpy()

    def get_mean_var_gate_temp(self):
        return float(self.mean_var_gate.temp.detach().cpu())

    def get_seg_fusion_weight(self):
        return float(torch.sigmoid(self.temporal_attn.seg_fusion_raw).detach().cpu())

    def get_attention_visual_weights(self, x):
        x = x.to(next(self.parameters()).device)
        x_feat = self._pre_attention_features(x)
        x_feat = self.motor_attention(x_feat)
        x_feat = self.spatial_act(self.spatial_bn(self.spatial_conv(x_feat)))
        x_feat = x_feat.squeeze(dim=2)
        return self.temporal_attn.get_segmented_attention_weights(x_feat)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dummy_data = torch.randn(100, 32, 1000).to(device)
    dummy_labels = torch.randint(0, 5, (100,)).to(device)

    importance = compute_channel_importance(dummy_data, dummy_labels)
    selected_idx = select_motor_channels(importance, top_k=16)

    model = FineSTAN(
        num_classes=5,
        num_samples=1000,
        selected_channel_idx=selected_idx,
        window_size=100,
        stride=50,
        seg_fusion_init=0.5
    ).to(device)

    test_input = torch.randn(8, 32, 1000).to(device)
    test_output = model(test_input)
    local_attn, global_attn = model.get_attention_visual_weights(test_input)

    assert local_attn.shape == (8, 1000)
    assert global_attn.shape == (8, 1000)
    assert test_output.shape == (8, 5)
    assert (local_attn >= 0).all() and (local_attn <= 1).all()