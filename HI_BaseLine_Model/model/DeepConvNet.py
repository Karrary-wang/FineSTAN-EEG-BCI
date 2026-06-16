"""
作者：王帆
时间：2022. 9.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self, classes, chans, time_points, batch_norm=True, batch_norm_alpha=0.1, kernal=10):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes
        in_channels = chans
        time_step = time_points
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1, bias=False),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )

        # 自动计算全连接输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, time_step)
            out = self.convnet(dummy_input)
            self.n_outputs = out.numel() // out.shape[0]

        self.clf = nn.Sequential(
            nn.Linear(self.n_outputs, self.classes_num),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        # 已删除多余的 x = x.unsqueeze(1)，避免维度报错
        output = self.convnet(x)
        output = output.flatten(1)
        output = self.clf(output)
        return output
