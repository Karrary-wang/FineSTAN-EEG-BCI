import torch
import numpy as np
from torch.utils.data import Dataset

class eegDataset(Dataset):
    """
    EEG 数据集类，用于 PyTorch DataLoader。
    Parameters:
        data (np.ndarray): 形状为 (N, C, T) 的 EEG 数据
        label (np.ndarray): 长度为 N 的标签数组（应为整数）
        transform (callable, optional): 可选的数据变换函数（如标准化、滤波）
    """
    def __init__(self, data, label, transform=None):
        # 1. 校验数据维度（确保EEG数据为3维 (N, C, T)）
        assert data.ndim == 3, f"EEG data must be 3D (N, C, T), got {data.ndim}D"
        # 2. 校验数据与标签样本数一致
        assert len(data) == len(label), f"Data length ({len(data)}) != Label length ({len(label)})"
        # 3. 校验标签为整数类型（避免非整数标签导致后续分类错误）
        assert np.issubdtype(label.dtype, np.integer), f"Label must be integer type, got {label.dtype}"

        # 转为 PyTorch 张量并指定类型（保持与模型/损失函数的适配）
        self.data = torch.from_numpy(data).float()  # float32 适配模型输入
        self.labels = torch.from_numpy(label).long()  # int64 适配 CrossEntropyLoss
        self.transform = transform

    def __getitem__(self, index):
        """根据索引返回单个样本（支持数据变换）"""
        data = self.data[index]
        label = self.labels[index]

        # 应用数据变换（支持张量输入的变换函数）
        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.data)