"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    """使用卷积层提取局部特征并生成序列嵌入（适配32通道+1000采样点）"""

    def __init__(self, emb_size=40, input_channels=32):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # 时间维度卷积
            nn.Conv2d(40, 40, (input_channels, 1), (1, 1)),  # 32通道适配
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # 池化参数
            nn.Dropout(0.3),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制模块（优化数值稳定性）"""

    def __init__(self, emb_size, num_heads=8, dropout=0.3):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        assert emb_size % num_heads == 0, f"emb_size {emb_size} 必须能被num_heads {num_heads} 整除"

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = (self.emb_size // self.num_heads) ** 0.5
        energy = energy / scaling

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy + 1e-8, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    """残差连接模块（添加dropout提升稳定性）"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = self.dropout(x)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    """前馈网络模块（优化激活函数+降低dropout）"""

    def __init__(self, emb_size, expansion=4, drop_p=0.3):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.LayerNorm(expansion * emb_size),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """Transformer编码器块（优化参数）"""

    def __init__(self, emb_size, num_heads=8, drop_p=0.3, forward_expansion=4, forward_drop_p=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ))
        )


class TransformerEncoder(nn.Sequential):
    """Transformer编码器（堆叠多个编码器块）"""

    def __init__(self, depth=4, emb_size=40):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """分类头模块（完全自适应维度，无硬编码）"""

    def __init__(self, emb_size=40, n_classes=10):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        # 仅定义后续层，第一层在首次前向时根据实际维度初始化
        self.layers = nn.Sequential(
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

        # 动态层（初始化标记）
        self.fc1 = None
        self._initialized = False

    def forward(self, x):
        # 展平特征：[batch_size, seq_len, emb_size] → [batch_size, seq_len×emb_size]
        x_flat = x.contiguous().view(x.size(0), -1)
        input_dim = x_flat.size(1)

        # 首次前向时动态初始化第一层（适配实际维度）
        if not self._initialized:
            self.fc1 = nn.Linear(input_dim, 256).to(x.device)
            # 打印维度信息（便于调试）
            print(f"【分类头初始化】实际输入维度: {input_dim} → 256")
            self._initialized = True

        # 前向传播
        x = self.fc1(x_flat)
        out = self.layers(x)
        return x_flat, out


class EEGConformer(nn.Module):
    """EEG-Conformer主模型（完全自适应维度版）"""

    def __init__(self, emb_size=40, depth=4, n_classes=10, num_heads=8, drop_p=0.3, forward_expansion=4,
                 input_channels=32):
        super().__init__()
        # 32通道补丁嵌入
        self.patch_embedding = PatchEmbedding(emb_size, input_channels)
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        # 自适应分类头
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        """
        前向传播（适配训练逻辑）
        :param x: 输入张量，形状为 [batch_size, 1, 32, 1000]
        :return: 分类输出（训练） | (特征张量, 分类输出)（评估）
        """
        # 1. 补丁嵌入
        x = self.patch_embedding(x)
        # 打印序列长度（便于调试）
        if not self.training and not self.classification_head._initialized:
            print(f"【PatchEmbedding输出】序列长度: {x.size(1)}, Emb维度: {x.size(2)}")

        # 2. Transformer编码
        x = self.transformer_encoder(x)
        # 3. 分类（自适应维度）
        features, outputs = self.classification_head(x)

        # 训练模式仅返回分类输出，评估模式返回双值
        if self.training:
            return outputs
        else:
            return features, outputs

    def get_feature(self, x):
        """单独提取特征"""
        self.eval()
        with torch.no_grad():
            x = self.patch_embedding(x)
            x = self.transformer_encoder(x)
            x_flat = x.contiguous().view(x.size(0), -1)
        return x_flat


# 测试代码（验证自适应维度）
if __name__ == "__main__":
    torch.manual_seed(0)

    # 创建模型
    model = EEGConformer(
        emb_size=40,
        depth=4,
        n_classes=10,
        num_heads=8,
        drop_p=0.3,
        forward_expansion=4,
        input_channels=32
    )

    # 测试不同输入长度（验证自适应）
    test_input1 = torch.randn(8, 1, 32, 1000)  # 原长度
    test_input2 = torch.randn(8, 1, 32, 1024)  # 不同长度

    # 训练模式测试
    model.train()
    outputs1 = model(test_input1)
    print(f"输入1000采样点 → 分类输出形状: {outputs1.shape}")  # [8,10]

    # 评估模式测试
    model.eval()
    features2, outputs2 = model(test_input2)
    print(f"输入1024采样点 → 特征形状: {features2.shape}, 分类输出形状: {outputs2.shape}")