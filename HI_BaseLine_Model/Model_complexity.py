import torch
import math
from torch import nn
from torch import Tensor
from typing import Union, Tuple
import time
import numpy as np
from ptflops import get_model_complexity_info


# -------------------------- EEGNet 模型完整定义 --------------------------
class EEGNetModel(nn.Module):
    def __init__(self, chans=32, classes=5, time_points=1000, temp_kernel=25,
                 f1=8, f2=16, d=2, pk1=16, pk2=8, dropout_rate=0.5,
                 max_norm1=1, max_norm2=1):
        super().__init__()
        linear_size = (time_points // (pk1 * pk2)) * f2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        self._apply_max_norm(self.block2[0], max_norm1)
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer: nn.Module, max_norm: float):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x: Tensor, return_feature: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feature = self.flatten(x)
        output = self.fc(feature)

        if return_feature:
            return output, feature
        return output


# -------------------------- 通用统计工具函数（口径完全统一） --------------------------
def count_trainable_params(model):
    """原生PyTorch统计可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数量: {total_params:,}")
    print(f"参数量 (K): {total_params / 1000:.2f} K\n")
    return total_params


def get_model_flops(model, input_shape):
    """ptflops统计FLOPs"""
    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    flops = macs * 2
    print(f"FLOPs (M): {flops / 1e6:.2f} M\n")
    return flops


def measure_inference_time(model, input_tensor, warmup=50, repeat=1000):
    """统计单样本平均推理时间 ms"""
    model.eval()
    with torch.no_grad():
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


# -------------------------- 主执行入口 --------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}\n")

    # 1. 模型初始化：统一10分类基准，参数与训练配置保持一致
    model = EEGNetModel(
        chans=32,
        classes=10,
        time_points=1000
    ).to(device)
    model.eval()

    # 2. 输入配置（batch=1 学术通用标准）
    input_shape = (1, 32, 1000)  # ptflops专用，不含batch维度
    test_input = torch.randn(1, 1, 32, 1000).to(device)

    # 3. 执行统计
    print("===== EEGNet 模型复杂度统计结果 =====")
    count_trainable_params(model)
    get_model_flops(model, input_shape)
    measure_inference_time(model, test_input)
    print("=====================================")