"""
作者：王帆
时间：2022. 9.8
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# -------------------------- 基础配置（仅需修改这里） --------------------------
# 路径配置（和训练集一致）
eeg_data_root = r'D:\Handwriting_Imagery_Raw_Dara\EL-HI-EEG_BIDS'
event_data_root = r'D:\Handwriting_Imagery_Raw_Dara\EL_HI_Event_Txt'
save_path = r'D:\EEG_HandWriting_Project_Code\FineSTAN\dataset\bci_handwriting_data_English'

# 要处理的被试编号（保持01，仅改下面2处）
sub_id = '17'
# ========== 修改1：测试集会话编号改为02 ==========
ses_id = '02'

# -------------------------- 1. 检查并读取事件文件 --------------------------
all_files = os.listdir(event_data_root)

# ========== 修改2：筛选带E的测试集事件文件（替换原T） ==========
event_files = [
    f for f in all_files
    if (f.startswith(f'A0{sub_id}E') or f.startswith(f'A{sub_id}E'))  # 匹配A01E
    and ('_Event' in f)
]

# 容错：未找到测试集事件文件时提示
if not event_files:
    print(f"\n 未找到A0{sub_id}E相关的测试集事件文件！")
    exit()

# 取第一个匹配的事件文件
event_file = event_files[0]
event_path = os.path.join(event_data_root, event_file)

# 读取事件数据（逻辑和训练集完全一致）
events = []
with open(event_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            sample, event_type = map(int, line.split())
            events.append([sample, 0, event_type])
events = np.array(events)

# -------------------------- 2. 读取并预处理BDF数据 --------------------------
# 拼接BDF文件路径（自动适配ses-02）
eeg_dir = os.path.join(eeg_data_root, f'sub-{sub_id}', f'ses-{ses_id}', 'eeg')

# 查找BDF文件
bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf')]

bdf_file = bdf_files[0]
bdf_path = os.path.join(eeg_dir, bdf_file)

# 加载BDF数据（和训练集预处理逻辑完全一致）
raw_data = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
# 保留32个EEG通道（和训练集一致）
raw_data.pick_types(eeg=True)
# 单位转换+滤波（和训练集参数完全一致，保证数据对齐）
raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)
raw_data.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)

# -------------------------- 3. 生成Epochs并提取数据 --------------------------
# 事件ID映射（和训练集一致）
event_id = {str(i): i for i in range(1, 11)}
# 生成Epochs（时间窗/降采样和训练集一致）
epochs = mne.Epochs(
    raw_data, events, event_id,
    tmin=0, tmax=4, baseline=None,
    preload=True, verbose=False
)
epochs_resampled = epochs.resample(250, npad="auto", verbose=False)

# 提取数据和标签（格式和训练集一致）
data = epochs_resampled.get_data()
labels = epochs_resampled.events[:, -1]

# -------------------------- 4. 按事件文件名保存 --------------------------
# 提取前缀（自动生成A01E，无需手动改）
event_file_prefix = event_file.replace('.txt', '').replace('_Event', '')
os.makedirs(save_path, exist_ok=True)
data_save_path = os.path.join(save_path, f'{event_file_prefix}_data.npy')
label_save_path = os.path.join(save_path, f'{event_file_prefix}_label.npy')

# 保存数据和标签
np.save(data_save_path, data)
np.save(label_save_path, labels)

# -------------------------- 打印信息 --------------------------
print(f"测试集（A0{sub_id}E）预处理完成！")
print(f"数据形状（试次数 × 通道数 × 时间点）：{data.shape}")
print(f"标签形状（试次数）：{labels.shape}")
