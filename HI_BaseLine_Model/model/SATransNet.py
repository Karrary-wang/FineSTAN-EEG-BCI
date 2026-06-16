"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== CNN 特征提取模块 ====================
class PatchEmbeddingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (32, 1), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 10)),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# ==================== 注意力模块 ====================
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 16*3)
        self.out_proj = nn.Linear(16, 16)
        self.attn_drop = nn.Dropout(0.1)
        self.scale = (8)**-0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 2, 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, 16)
        return self.out_proj(out)

# ==================== Transformer ====================
class FeedForward(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Dropout(0.1)
        )

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(16)
        self.attn = SparseAttention()
        self.norm2 = nn.LayerNorm(16)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 25, 16))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(x + self.pos_emb[:, :x.size(1)])

# ==================== 主模型 ====================
class SATransNet(nn.Module):
    def __init__(self, n_classes=5, **kwargs):
        super().__init__()
        self.cnn = PatchEmbeddingCNN()
        self.pos_enc = PositionalEncoding()
        self.transformer = TransformerBlock()
        self.norm = nn.LayerNorm(16)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(25*16, n_classes))

    def forward(self, x):
        x = self.cnn(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)

if __name__ == "__main__":
    test = torch.randn(2, 1, 32, 1000)
    model = SATransNet(n_classes=5)
    print("输入:", test.shape)
    print("输出:", model(test).shape)