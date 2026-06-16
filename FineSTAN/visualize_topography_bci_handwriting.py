"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mne
import yaml
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import math
from matplotlib.cm import ScalarMappable

# 字体配置（统一Arial）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14

# ===================== 配置 =====================
SUBJECT_ID = 12
CLASS_NAMES = {1: 'a', 2: 'd', 3: 'e', 4: 'f', 5: 'j', 6: 'n', 7: 'o', 8: 's', 9: 't', 10: 'v'}
ALL_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_CLASSES = 10

DATA_ROOT = r"D:/EEG_HandWriting_Project_Code/FineSTAN/dataset/bci_handwriting_data_English"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\FineSTAN\output\t-SNE_Plot_choice_model\sub12\epoch_400\cv_2026-05-03--15-55"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\FineSTAN\output\cat_topography\ELL-HI"

os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== GradCAM =====================
class GradCAM:
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        if self.use_cuda:
            self.model = self.model.cuda()

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for layer in self.target_layers:
            layer.register_forward_hook(save_activation)
            layer.register_full_backward_hook(lambda m, gi, go: save_gradient(m, gi, go))

    def __call__(self, input_tensor, target_category=None):
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        target = output[0, target_category]
        target.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        return cam.cpu().numpy()


# ===================== 模型结构 =====================
def attention(query, key, value, dropout=None):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhqk,bhkd->bhqd', attn, value), attn


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
        x_padded = F.pad(x, (0, pad_len)) if pad_len else x
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
        self.stride = stride
        self.local_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ELU(), nn.Dropout(dropout),
                                       nn.Linear(embed_dim // 2, 1))
        self.global_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ELU(), nn.Dropout(dropout),
                                        nn.Linear(embed_dim // 2, 1))
        self.seg_fusion_raw = nn.Parameter(torch.tensor(math.log(seg_fusion_init / (1 - seg_fusion_init))))
        self.pooled_avg_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pooled_var_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.temp_coeff = nn.Parameter(torch.tensor(temp_coeff_init))
        self.global_weight_raw = nn.Parameter(torch.tensor(math.log(global_weight_init / (1 - global_weight_init))))

    def _local_window_attention(self, x_trans):
        B, T, D = x_trans.shape
        pad_len = (self.stride - (T % self.stride)) % self.stride
        x_padded = F.pad(x_trans, (0, 0, 0, pad_len)) if pad_len else x_trans
        T_pad = x_padded.shape[1]
        x_unfolded = F.unfold(x_padded.transpose(1, 2).unsqueeze(-1), kernel_size=(self.window_size, 1),
                              stride=(self.stride, 1))
        x_windowed = x_unfolded.view(B, D, self.window_size, -1).permute(0, 3, 2, 1)
        B, N, W, D = x_windowed.shape
        local_attn_flat = self.local_mlp(x_windowed.flatten(0, 2))
        local_attn = local_attn_flat.view(B, N, W, 1)
        if pad_len > 0:
            local_attn[:, -1, W - pad_len:, :] = -float('inf')
        safe_temp = F.softplus(self.temp_coeff) + 0.1
        local_attn = F.softmax(local_attn / safe_temp, dim=2)
        attn_unfolded = local_attn.permute(0, 3, 2, 1).flatten(1, 2)
        count_unfolded = torch.ones_like(attn_unfolded)
        attn_sum = F.fold(attn_unfolded, output_size=(T_pad, 1), kernel_size=(self.window_size, 1),
                          stride=(self.stride, 1)).squeeze(-1)
        count_folded = F.fold(count_unfolded, output_size=(T_pad, 1), kernel_size=(self.window_size, 1),
                              stride=(self.stride, 1)).squeeze(-1)
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
        q = rearrange(self.w_q(x), 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(self.w_k(x), 'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(self.w_v(x), 'b n (h d) -> b h n d', h=self.n_head)
        out, _ = attention(q, k, v, dropout=self.dropout_attn)
        out = rearrange(out, 'b h q d -> b q (h d)')
        return self.dropout(self.w_o(out))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.3, fc_drop=0.3):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * fc_ratio), nn.GELU(), nn.Dropout(fc_drop),
                                 nn.Linear(embed_dim * fc_ratio, embed_dim), nn.Dropout(fc_drop))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pos_enc(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# 【已修正】模型类名改为 FineSTAN
class FineSTAN(nn.Module):
    def __init__(self, num_classes=10, num_samples=1000, num_channels=32, embed_dim=32, pool_size=180, pool_stride=30,
                 num_heads=8, fc_ratio=4, depth=4, attn_drop=0.3, fc_drop=0.3, classify_drop=0.5, pool_dropout=0.3,
                 attn_norm_type="softmax", temp_coeff_init=0.5, global_weight_init=0.3, selected_channel_idx=None,
                 window_size=100, stride=50, seg_fusion_init=0.5):
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
        self.temp_conv = nn.ModuleList(
            [nn.Conv2d(1, embed_dim // 4, (1, k), padding=(0, k // 2)) for k in self.kernel_sizes])
        self.scale_logits = nn.Parameter(torch.tensor([2, 1.5, 1.0, 1]))
        self.temp_bn = nn.BatchNorm2d(embed_dim)
        self.instance_norm = nn.InstanceNorm2d(embed_dim)
        self.motor_attention = MotorAttention(selected_channel_idx)
        self.temporal_attn = SegmentedTemporalAttention(embed_dim, window_size, stride, num_samples, attn_norm_type,
                                                        0.2, temp_coeff_init, global_weight_init, seg_fusion_init)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.selected_channel_num, 1))
        self.spatial_bn = nn.BatchNorm2d(embed_dim)
        self.spatial_act = nn.ELU()
        self.dual_pool = DualPool(pool_size, pool_stride)
        self.mean_var_gate = MeanVarGate(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
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
            dev = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            dummy = torch.randn(1, self.num_channels, self.num_samples, device=dev, dtype=dtype)
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
        self.classify = nn.Linear(cls_in_dim, self.num_classes).to(dev)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
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


def reshape_transform(tensor):
    return rearrange(tensor, 'b e 1 t -> b e t 1')


def compute_cat_for_category(data, labels, model, target_category, cam):
    target_index = np.where(labels == target_category)[0]
    if len(target_index) == 0:
        return None, None
    target_data = data[target_index]
    all_cam = []
    for i in range(len(target_data)):
        t = torch.as_tensor(target_data[i:i + 1], dtype=torch.float32)
        t.requires_grad = True
        cm = cam(t, target_category - 1)
        all_cam.append(cm)
    mean_raw = np.mean(target_data, axis=0)
    vis_raw = np.mean(mean_raw, axis=1)
    mean_cam = np.mean(all_cam, axis=0)
    hybrid = mean_raw * mean_cam
    vis_hybrid = np.mean(hybrid, axis=1)
    return vis_raw, vis_hybrid


def main():
    data_file = f"A{SUBJECT_ID:02d}E"
    data_path = os.path.join(DATA_ROOT, f"{data_file}_data.npy")
    label_path = os.path.join(DATA_ROOT, f"{data_file}_label.npy")
    data = np.load(data_path)
    labels = np.load(label_path).squeeze()

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    args = cv_config['network_args']

    # 【已修正】导入并初始化 FineSTAN
    model = FineSTAN(**args)
    sd = torch.load(weights_path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    cam = GradCAM(model, [model.spatial_conv], use_cuda=False, reshape_transform=reshape_transform)

    ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Cz', 'C3', 'C4',
                'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2']
    info = mne.create_info(ch_names, 250, 'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))

    results = {}
    for c in ALL_CATEGORIES:
        vr, vh = compute_cat_for_category(data, labels, model, c, cam)
        if vr is not None:
            results[c] = (vr, vh)

    raws = [results[c][0] for c in ALL_CATEGORIES]
    cats = [results[c][1] for c in ALL_CATEGORIES]
    raw_vmax = np.max(np.abs(raws))
    raw_lim = (-raw_vmax, raw_vmax)
    cat_vmin = np.percentile(cats, 1.5)
    cat_vmax = np.percentile(cats, 98.5)
    cat_abs = max(abs(cat_vmin), abs(cat_vmax))
    cat_lim = (-cat_abs, cat_abs)

    # 4行5列布局
    fig, axes = plt.subplots(4, 5, figsize=(26, 18))
    axes = axes.flatten()

    # 正确设置标题
    for i in range(5):
        axes[i].set_title(CLASS_NAMES[ALL_CATEGORIES[i]], fontsize=36, pad=8, weight='medium')
        axes[i + 5].set_title(CLASS_NAMES[ALL_CATEGORIES[i + 5]], fontsize=36, pad=8, weight='medium')
        axes[i + 10].set_title(CLASS_NAMES[ALL_CATEGORIES[i]], fontsize=36, pad=8, weight='medium')
        axes[i + 15].set_title(CLASS_NAMES[ALL_CATEGORIES[i + 5]], fontsize=36, pad=8, weight='medium')

    # 左侧标签
    fig.text(0.017, 0.76, 'Raw', va='center', ha='center', fontsize=36, weight='bold')
    fig.text(0.017, 0.24, 'Activated', va='center', ha='center', fontsize=36, weight='bold')

    # 绘制 Raw 第1行
    for i in range(5):
        vr, _ = results[ALL_CATEGORIES[i]]
        mne.viz.plot_topomap(vr, info, axes=axes[i], show=False, res=1200, cmap='RdBu_r', vlim=raw_lim, contours=4)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    # 绘制 Raw 第2行
    for i in range(5):
        vr, _ = results[ALL_CATEGORIES[i + 5]]
        mne.viz.plot_topomap(vr, info, axes=axes[i + 5], show=False, res=1200, cmap='RdBu_r', vlim=raw_lim, contours=4)
        axes[i + 5].set_xticks([])
        axes[i + 5].set_yticks([])
    # 绘制 Activated 第3行
    for i in range(5):
        _, vh = results[ALL_CATEGORIES[i]]
        mne.viz.plot_topomap(vh, info, axes=axes[i + 10], show=False, res=1200, cmap='RdBu_r', vlim=cat_lim, contours=4)
        axes[i + 10].set_xticks([])
        axes[i + 10].set_yticks([])
    # 绘制 Activated 第4行
    for i in range(5):
        _, vh = results[ALL_CATEGORIES[i + 5]]
        mne.viz.plot_topomap(vh, info, axes=axes[i + 15], show=False, res=1200, cmap='RdBu_r', vlim=cat_lim, contours=4)
        axes[i + 15].set_xticks([])
        axes[i + 15].set_yticks([])

    # 间距+色柱
    plt.subplots_adjust(left=0.07, right=0.93, top=0.96, bottom=0.04, wspace=0.48, hspace=0.25)

    sm = ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=cat_lim[0], vmax=cat_lim[1]))
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation='vertical',
        fraction=0.03,
        pad=0.028,
        shrink=0.3
    )
    cbar.set_ticks([])
    cbar.ax.set_ylabel('')
    cbar.ax.text(1.5, 1.02, 'max', transform=cbar.ax.transAxes, fontsize=28, weight='bold', va='bottom', ha='center')
    cbar.ax.text(1.5, -0.02, 'min', transform=cbar.ax.transAxes, fontsize=28, weight='bold', va='top', ha='center')

    # 保存图片
    save_path = os.path.join(SAVE_DIR, "ELL-HI_sub12_4x5_final.png")
    fig.savefig(save_path, dpi=650, bbox_inches='tight')
    plt.close()
    print("英文字母地形图已保存完成！")


if __name__ == '__main__':
    main()
    """

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mne
import yaml
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import math
from matplotlib.cm import ScalarMappable

# 字体配置（与英文原版完全一致）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14

# ===================== 核心配置 =====================
SUBJECT_ID = 10
# 拼音单韵母6分类
CLASS_NAMES = {1: 'a', 2: 'o', 3: 'e', 4: 'i', 5: 'u', 6: 'ü'}
ALL_CATEGORIES = [1, 2, 3, 4, 5, 6]
NUM_CLASSES = 6

# 数据、模型、保存路径
DATA_ROOT = r"D:/EEG_HandWriting_Project_Code/EEG-TransNet-main/dataset/bci_handwriting_data_Pinyin"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub10\epoch_400\cv_2026-05-28--06-55"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\cat_topography\PINYIN_SV"

os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== GradCAM 模块 =====================
class GradCAM:
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        if self.use_cuda:
            self.model = self.model.cuda()

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for layer in self.target_layers:
            layer.register_forward_hook(save_activation)
            layer.register_full_backward_hook(lambda m, gi, go: save_gradient(m, gi, go))

    def __call__(self, input_tensor, target_category=None):
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        target = output[0, target_category]
        target.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        return cam.cpu().numpy()


# ===================== 模型结构 =====================
def attention(query, key, value, dropout=None):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.einsum('bhqk,bhkd->bhqd', attn, value), attn


class DualPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-8
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=0)

    def forward(self, x):
        B, D, T = x.shape
        pad_len = (self.kernel_size - (T % self.kernel_size)) % self.kernel_size
        x_padded = F.pad(x, (0, pad_len)) if pad_len else x
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
        self.stride = stride
        self.local_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ELU(), nn.Dropout(dropout),
                                       nn.Linear(embed_dim // 2, 1))
        self.global_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ELU(), nn.Dropout(dropout),
                                        nn.Linear(embed_dim // 2, 1))
        self.seg_fusion_raw = nn.Parameter(torch.tensor(math.log(seg_fusion_init / (1 - seg_fusion_init))))
        self.pooled_avg_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pooled_var_bias = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.temp_coeff = nn.Parameter(torch.tensor(temp_coeff_init))
        self.global_weight_raw = nn.Parameter(torch.tensor(math.log(global_weight_init / (1 - global_weight_init))))

    def _local_window_attention(self, x_trans):
        B, T, D = x_trans.shape
        pad_len = (self.stride - (T % self.stride)) % self.stride
        x_padded = F.pad(x_trans, (0, 0, 0, pad_len)) if pad_len else x_trans
        T_pad = x_padded.shape[1]
        x_unfolded = F.unfold(x_padded.transpose(1, 2).unsqueeze(-1), kernel_size=(self.window_size, 1),
                              stride=(self.stride, 1))
        x_windowed = x_unfolded.view(B, D, self.window_size, -1).permute(0, 3, 2, 1)
        B, N, W, D = x_windowed.shape
        local_attn_flat = self.local_mlp(x_windowed.flatten(0, 2))
        local_attn = local_attn_flat.view(B, N, W, 1)
        if pad_len > 0:
            local_attn[:, -1, W - pad_len:, :] = -float('inf')
        safe_temp = F.softplus(self.temp_coeff) + 0.1
        local_attn = F.softmax(local_attn / safe_temp, dim=2)
        attn_unfolded = local_attn.permute(0, 3, 2, 1).flatten(1, 2)
        count_unfolded = torch.ones_like(attn_unfolded)
        attn_sum = F.fold(attn_unfolded, output_size=(T_pad, 1), kernel_size=(self.window_size, 1),
                          stride=(self.stride, 1)).squeeze(-1)
        count_folded = F.fold(count_unfolded, output_size=(T_pad, 1), kernel_size=(self.window_size, 1),
                              stride=(self.stride, 1)).squeeze(-1)
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
        q = rearrange(self.w_q(x), 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(self.w_k(x), 'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(self.w_v(x), 'b n (h d) -> b h n d', h=self.n_head)
        out, _ = attention(q, k, v, dropout=self.dropout_attn)
        out = rearrange(out, 'b h q d -> b q (h d)')
        return self.dropout(self.w_o(out))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.3, fc_drop=0.3):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * fc_ratio), nn.GELU(), nn.Dropout(fc_drop),
                                 nn.Linear(embed_dim * fc_ratio, embed_dim), nn.Dropout(fc_drop))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pos_enc(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

# 【已修正】模型类名改为 FineSTAN
class FineSTAN(nn.Module):
    def __init__(self, num_classes=10, num_samples=1000, num_channels=32, embed_dim=32, pool_size=180, pool_stride=30,
                 num_heads=8, fc_ratio=4, depth=4, attn_drop=0.3, fc_drop=0.3, classify_drop=0.5, pool_dropout=0.3,
                 attn_norm_type="softmax", temp_coeff_init=0.5, global_weight_init=0.3, selected_channel_idx=None,
                 window_size=100, stride=50, seg_fusion_init=0.5):
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
        self.temp_conv = nn.ModuleList(
            [nn.Conv2d(1, embed_dim // 4, (1, k), padding=(0, k // 2)) for k in self.kernel_sizes])
        self.scale_logits = nn.Parameter(torch.tensor([2, 1.5, 1.0, 1]))
        self.temp_bn = nn.BatchNorm2d(embed_dim)
        self.instance_norm = nn.InstanceNorm2d(embed_dim)
        self.motor_attention = MotorAttention(selected_channel_idx)
        self.temporal_attn = SegmentedTemporalAttention(embed_dim, window_size, stride, num_samples, attn_norm_type,
                                                        0.2, temp_coeff_init, global_weight_init, seg_fusion_init)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.selected_channel_num, 1))
        self.spatial_bn = nn.BatchNorm2d(embed_dim)
        self.spatial_act = nn.ELU()
        self.dual_pool = DualPool(pool_size, pool_stride)
        self.mean_var_gate = MeanVarGate(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
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
            dev = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            dummy = torch.randn(1, self.num_channels, self.num_samples, device=dev, dtype=dtype)
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
        self.classify = nn.Linear(cls_in_dim, self.num_classes).to(dev)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
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


def reshape_transform(tensor):
    return rearrange(tensor, 'b e 1 t -> b e t 1')


def compute_cat_for_category(data, labels, model, target_category, cam):
    target_index = np.where(labels == target_category)[0]
    if len(target_index) == 0:
        return None, None
    target_data = data[target_index]
    all_cam = []
    for i in range(len(target_data)):
        t = torch.as_tensor(target_data[i:i + 1], dtype=torch.float32)
        t.requires_grad = True
        cm = cam(t, target_category - 1)
        all_cam.append(cm)
    mean_raw = np.mean(target_data, axis=0)
    vis_raw = np.mean(mean_raw, axis=1)
    mean_cam = np