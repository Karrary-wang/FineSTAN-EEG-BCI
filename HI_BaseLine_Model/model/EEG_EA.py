"""
作者：王帆
时间：2022. 9.8
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


# ===================== 基础模块定义 =====================
class Pooling1D(nn.Module):
    """PoolFormer池化（1D版，适配EEG时序特征）"""

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = self.pool(x)
        return y - x  # 残差池化，保留细节


class Conv1(nn.Module):
    """1D卷积（带权重归一化+BN+ELU）"""

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                              padding=self.padding, groups=groups)
        self.conv = nn.utils.weight_norm(self.conv)  # 权重归一化，稳定训练
        self.norm = nn.BatchNorm1d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight)
        self.act = nn.ELU()  # 适配EEG信号的非线性

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class WindowPool(nn.Module):
    """滑动窗口池化（压缩时间维度，保留时序特征）"""

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=kernel_size,
                              stride=stride, groups=1)

    def forward(self, x):
        B, C, T = x.size()
        x = rearrange(x, 'b c t -> (b c) 1 t')  # 适配Conv1d输入
        out = self.pool(x)
        x = rearrange(out, "(b c) h t -> b (h c) t", b=B)
        x = rearrange(x, "b (h c) t -> b c (h t)", c=C)
        return x


class Conv(nn.Module):
    """核心卷积模块（带窗口池化+残差）"""

    def __init__(self, in_channels, out_channels, kernel_size=75, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size, dilation=dilation,
                              padding=self.padding, groups=groups, bias=bias)
        self.window = WindowPool()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        B, C, T = x.size()
        x0 = self.window(x)
        x = x0 + x  # 残差连接，避免特征丢失
        x = rearrange(x, "b (h c) t -> (b c) h t", h=1)
        out = self.act(self.norm(self.conv(x)))
        out = rearrange(out, "(b c) h t -> b (h c) t", b=B)
        return out


class MultiScalePooling(nn.Module):
    """多尺度池化（适配1000采样点的时间维度）"""

    def __init__(self, pool_kernels=[75, 115, 155]):
        super().__init__()
        self.pool_layers = nn.ModuleList([Pooling1D(pool_size=k) for k in pool_kernels])

    def forward(self, x):
        pooled_outputs = [pool(x) for pool in self.pool_layers]
        return pooled_outputs[0], pooled_outputs[1], pooled_outputs[2]


class GeMP1D(nn.Module):
    """门控几何平均池化（增强有用通道，抑制噪声）"""

    def __init__(self, p=4., eps=1e-6, learn_p=False, num_channels=288, epsilon=1e-5):
        super().__init__()
        self._p = p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.p.requires_grad = learn_p
        # 门控参数（适配32×9=288通道）
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon

    def timeavg(self, x):
        return x.clamp(min=self.eps).log().mean(dim=-1).exp().unsqueeze(2)

    def forward(self, x):
        o = self.timeavg(x)
        embedding = (o + self.epsilon).pow(0.5) * self.alpha
        # 门控归一化，增强手写想象相关特征
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate


class RMSPool1D(nn.Module):
    """均方根池化（提取EEG能量特征）"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        square_mean = F.avg_pool1d(x ** 2, self.kernel_size, self.stride, self.padding)
        return torch.sqrt(square_mean)


class RMSfusion(nn.Module):
    """多尺度RMS融合（适配1000采样点）"""

    def __init__(self, kernel_sizes=[50, 100, 200]):
        super().__init__()
        self.RMS_layers = nn.ModuleList([
            nn.Sequential(
                RMSPool1D(kernel_size=k, stride=int(k / 2)),
                nn.Flatten(start_dim=1)
            ) for k in kernel_sizes
        ])

    def forward(self, x, x1, x2):
        out = [self.RMS_layers[i](feat) for i, feat in enumerate([x, x1, x2])]
        return torch.concat(out, dim=1)


# ===================== 编码器模块 =====================
class EAEEG_Encoder(nn.Module):
    """EEG-EA编码器（适配32通道+1000采样点）"""

    def __init__(self, chans=32, multiple=9, pool_kernels=[50, 100, 200], dataset='handwriting'):
        super().__init__()
        self.dataset = dataset
        # 核心卷积（大核75捕捉手写想象慢变化特征）
        self.conv = Conv(in_channels=chans, out_channels=multiple, kernel_size=75)
        # 降维卷积（32×9=288 → 46通道，适配手写想象特征维度）
        self.conv1 = Conv1(in_channels=chans * multiple, out_channels=46, kernel_size=1)
        # 多尺度池化
        self.pool = MultiScalePooling()
        # 门控模块（适配288通道）
        self.GeMP1D = GeMP1D(num_channels=chans * multiple)
        # 多尺度RMS融合
        self.fftmix = RMSfusion(kernel_sizes=pool_kernels)

    def forward(self, x):
        # 卷积特征提取
        x = self.conv(x)
        # 通道归一化（消除不同电极/被试的幅值差异）
        out_mean = torch.mean(x, 2, True)
        out_var = torch.mean(x ** 2, 2, True)
        x = (x - out_mean) / torch.sqrt(out_var + 1e-5)
        # 门控增强
        x = self.GeMP1D(x)
        # 降维
        x = self.conv1(x)
        # 多尺度池化 + RMS融合
        x, x1, x2 = self.pool(x)
        x = self.fftmix(x, x1, x2)
        return x


# ===================== 主模型封装（和EEGConformer完全兼容） =====================
class EEGEA(nn.Module):
    """
    适配手写想象实验的EEG-EA主模型
    输入：(B, 1, 32, 1000) → 自动转换为(B, 32, 1000)
    输出：训练模式返回logits，评估模式返回(features, logits)（和EEGConformer一致）
    """

    def __init__(self, chans=32, samples=1000, multiple=9, num_classes=10,
                 dataset='handwriting', pool_kernels=[50, 100, 200]):
        super().__init__()
        # 保存核心参数（适配配置文件传参）
        self.chans = chans
        self.samples = samples
        self.num_classes = num_classes

        # 初始化编码器
        self.encoder = EAEEG_Encoder(
            chans=chans,
            multiple=multiple,
            pool_kernels=pool_kernels,
            dataset=dataset
        )

        # 预计算特征维度（避免运行时错误）
        with torch.no_grad():
            dummy_input = torch.ones((1, chans, samples))
            self.feat_dim = self.encoder(dummy_input).shape[-1]

        # 原型分类头（ISP：类间分离原型，适配10分类）
        self.isp = nn.Parameter(torch.randn(num_classes, self.feat_dim), requires_grad=True)
        self.icp = nn.Parameter(torch.randn(num_classes, self.feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)  # 初始化

        # 存储特征（用于评估/测试）
        self.features = None

    def get_features(self):
        """获取最后一层特征（和EEGConformer接口一致）"""
        if self.features is not None:
            return self.features
        raise RuntimeError("先执行forward()提取特征！")

    def forward(self, x):
        """
        核心前向逻辑（完全兼容EEGConformer的训练/评估流程）
        """
        # 自动适配4维输入（EEGConformer的输入格式：B,1,32,1000 → B,32,1000）
        if len(x.shape) == 4:
            x = x.squeeze(1)

        # 特征提取
        features = self.encoder(x)
        self.features = features

        # 原型分类（ISP归一化，增强类间分离）
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)

        # 兼容EEGConformer的输出逻辑
        if self.training:
            return logits  # 训练模式仅返回logits
        else:
            return features, logits  # 评估模式返回(特征, logits)


# ===================== 测试代码（验证适配性） =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型（适配你的手写想象实验参数）
    model = EEGEA(
        chans=32,
        samples=1000,
        num_classes=10,
        multiple=9,
        pool_kernels=[50, 100, 200]
    ).to(device)

    # 测试4维输入（模拟训练代码的输入）
    test_input = torch.randn(8, 1, 32, 1000).to(device)

    # 训练模式
    model.train()
    outputs = model(test_input)
    print("训练模式输出形状:", outputs.shape)  # 应为 (8, 10)

    # 评估模式
    model.eval()
    features, outputs = model(test_input)
    print("评估模式特征形状:", features.shape)  # 特征维度
    print("评估模式输出形状:", outputs.shape)  # 应为 (8, 10)

    # 打印可训练参数数量（轻量化，适合批量训练）
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1000:.2f}K")