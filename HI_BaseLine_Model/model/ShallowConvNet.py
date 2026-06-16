"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self, classes, chans, time_points, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes
        in_channels = chans
        time_step = time_points
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1)
            )

        # 自动计算全连接输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, time_step)
            out = self.layer1(dummy_input)
            out = torch.square(out)
            out = F.avg_pool2d(out, (1, 75), 15)
            self.n_outputs = out.numel() // out.shape[0]

        self.clf = nn.Linear(self.n_outputs, self.classes_num)

    def forward(self, x):
        # 删掉这一行：x = x.unsqueeze(1)  数据外部已经补好通道维
        x = self.layer1(x)
        x = torch.square(x)
        x = F.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x + 1e-8)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.flatten(1)
        x = self.clf(x)
        return x