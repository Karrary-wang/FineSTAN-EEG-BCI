"""
作者：王帆
时间：2026. 1.7
"""
# Task ：hand writing   a d e f j  n o s t v  ten classification session1--train data
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# -------------------------- 基础配置（仅需修改这里） --------------------------
# 路径配置（确认是实际文件所在路径）
eeg_data_root = r'D:\Handwriting_Imagery_Raw_Dara\EL-HI-EEG_BIDS'
event_data_root = r'D:\Handwriting_Imagery_Raw_Dara\EL_HI_Event_Txt'
save_path = r'D:\EEG_HandWriting_Project_Code\FineSTAN\dataset\bci_handwriting_data_English'

# 要处理的被试编号（改这里切换被试，如02、03）
sub_id = '17'
ses_id = '01'

# -------------------------- 1. 检查并读取事件文件 --------------------------
# 打印事件文件夹内所有文件，方便排查
all_files = os.listdir(event_data_root)

# 筛选带T的训练集事件文件（兼容带/不带.txt后缀）
event_files = [
    f for f in all_files
    if (f.startswith(f'A0{sub_id}T') or f.startswith(f'A{sub_id}T'))
    and ('_Event' in f)
]

# 取第一个匹配的事件文件
event_file = event_files[0]
event_path = os.path.join(event_data_root, event_file)

# 读取事件数据
events = []
with open(event_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            sample, event_type = map(int, line.split())
            events.append([sample, 0, event_type])
events = np.array(events)

# -------------------------- 2. 读取并预处理BDF数据 --------------------------
# 拼接BDF文件路径
eeg_dir = os.path.join(eeg_data_root, f'sub-{sub_id}', f'ses-{ses_id}', 'eeg')

# 查找BDF文件
bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf')]

bdf_file = bdf_files[0]
bdf_path = os.path.join(eeg_dir, bdf_file)

# 加载BDF数据
raw_data = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

# ========== 核心修改：仅保留32个EEG通道，删除刺激通道 ==========
# pick_types(eeg=True) 会只保留EEG类型通道，自动过滤STIM刺激通道
raw_data.pick_types(eeg=True)

# 后续预处理（单位转换+滤波）
raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)  # 转换单位：V → uV
raw_data.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)  # 1-40Hz带通滤波

# -------------------------- 3. 生成Epochs并提取数据 --------------------------
# 事件ID映射（1-10对应手写字符）
event_id = {str(i): i for i in range(1, 11)}
# 生成Epochs（时间窗：0~4秒，无基线校正）
epochs = mne.Epochs(
    raw_data, events, event_id,
    tmin=0, tmax=4, baseline=None,
    preload=True, verbose=False
)
# 降采样到250Hz
epochs_resampled = epochs.resample(250, npad="auto", verbose=False)

# 提取数据和标签
data = epochs_resampled.get_data()  # 形状：[试次数, 32, 1000]
labels = epochs_resampled.events[:, -1]  # 每个试次对应的标签

# -------------------------- 4. 按事件文件名保存（核心修改） --------------------------
# 提取事件文件前缀（去掉.txt和_Event，如A01T.txt_Event → A01T）
event_file_prefix = event_file.replace('.txt', '').replace('_Event', '')
# 确保保存路径存在
os.makedirs(save_path, exist_ok=True)
# 生成最终保存路径（格式：A01T_data.npy / A01T_label.npy）
data_save_path = os.path.join(save_path, f'{event_file_prefix}_data.npy')
label_save_path = os.path.join(save_path, f'{event_file_prefix}_label.npy')

# 保存数据和标签
np.save(data_save_path, data)
np.save(label_save_path, labels)

# -------------------------- 打印数据和标签信息 --------------------------
print("="*60)
# 打印数据大小（形状）
print(f"数据形状（试次数 × 通道数 × 时间点）：{data.shape}")
# 打印标签大小（形状）
print(f"️标签形状（试次数）：{labels.shape}")
