"""
作者：王帆
时间：2026. 1.8
"""
# Task：hand writing  heng shu dian pie zhe  five classification (汉字笔画分类)
import mne
import numpy as np
import os
# -------------------------- 基础配置（仅需修改这里） --------------------------
# 路径配置（严格对齐英文字母代码的BIDS格式路径结构）
eeg_data_root = r'D:\Handwriting_Imagery_Raw_Dara\CCS-SV-HI-EEG_BIDS'  # 汉字笔画EEG BIDS根路径
event_data_root = r'D:\Handwriting_Imagery_Raw_Dara\CCS_HI_Event_Txt'  # 汉字笔画事件文件根路径
save_path = r'D:\EEG_HandWriting_Project_Code\FineSTAN\dataset\bci_handwriting_data'  # 处理后数据保存路径

# 严格对齐英文字母代码：补充被试编号、会话编号（如需切换直接修改此处）
sub_id = '03'
ses_id = '01'

# 预处理相关参数（保留原汉字笔画代码的核心参数）
factor_new = 1e-3
init_block_size = 1000  # sampling frequency *times

# -------------------------- 1. 检查并读取事件文件 --------------------------
# 打印事件文件夹内所有文件，方便排查（对齐英文字母代码的排查逻辑）
all_files = os.listdir(event_data_root)

# 筛选带T的训练集事件文件（兼容A0{sub_id}T，匹配_Event和.txt后缀，对齐英文字母筛选逻辑）
event_files = [
    f for f in all_files
    if f.startswith(f'A{sub_id}T')
    and ('_Event' in f)
    and f.endswith('.txt')
]

# 取第一个匹配的事件文件（对齐英文字母代码的文件选取逻辑）
event_file = event_files[0]
event_path = os.path.join(event_data_root, event_file)

# 读取事件数据（保留原汉字笔画的事件读取逻辑，对齐英文字母代码的空行跳过优化）
events = []
with open(event_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行，提升鲁棒性
            sample, event_type = map(int, line.split())
            events.append([sample, 0, event_type])
events = np.array(events)

# -------------------------- 2. 读取并预处理BDF数据 --------------------------
# 严格对齐英文字母代码：拼接BIDS格式的EEG目录路径（sub-{sub_id}/ses-{ses_id}/eeg）
eeg_dir = os.path.join(eeg_data_root, f'sub-{sub_id}', f'ses-{ses_id}', 'eeg')

# 严格对齐英文字母代码：在BIDS格式的eeg目录下查找BDF文件
bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf')]
bdf_file = bdf_files[0]
bdf_path = os.path.join(eeg_dir, bdf_file)

# 加载BDF数据（对齐英文字母代码的加载参数）
raw_data = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

# 仅保留32个EEG通道，删除刺激通道（严格对齐英文字母代码的通道筛选逻辑）
raw_data.pick_types(eeg=True)

# 后续预处理（保留原汉字笔画的单位转换+滤波逻辑，对齐英文字母代码的格式）
raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)  # 转换单位：V → uV
raw_data.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)  # 1-40Hz带通滤波

# -------------------------- 3. 生成Epochs并提取数据 --------------------------
# 事件ID映射（保留汉字笔画的5分类映射：1-5对应横、竖、点、撇、折，对齐英文字母代码的映射格式）
event_id = {str(i): i for i in range(1, 6)}
# 生成Epochs（严格对齐英文字母代码的参数格式，保留汉字笔画的0~4秒时间窗）
epochs = mne.Epochs(
    raw_data, events, event_id,
    tmin=0, tmax=4, baseline=None,
    preload=True, verbose=False
)
# 降采样到250Hz（严格对齐英文字母代码的降采样逻辑）
epochs_resampled = epochs.resample(250, npad="auto", verbose=False)

# 提取数据和标签（保留原汉字笔画的数据提取逻辑，对齐英文字母代码的变量格式）
data = epochs_resampled.get_data()  # 形状：[试次数, n_channels, n_times]
true_labels = epochs_resampled.events[:, -1]  # 每个试次对应的汉字笔画标签

# -------------------------- 4. 按事件文件名保存（严格对齐英文字母代码的保存格式） --------------------------
# 提取事件文件前缀（去掉.txt和_Event，统一文件命名格式，对齐英文字母代码）
event_file_prefix = event_file.replace('.txt', '').replace('_Event', '')
# 确保保存路径存在（不存在则自动创建，对齐英文字母代码的鲁棒性优化）
os.makedirs(save_path, exist_ok=True)
# 生成最终保存路径（格式：A001T_data.npy / A001T_label.npy，对齐英文字母代码）
data_save_path = os.path.join(save_path, f'{event_file_prefix}_data.npy')
label_save_path = os.path.join(save_path, f'{event_file_prefix}_label.npy')

# 保存数据和标签（对齐英文字母代码的保存逻辑）
np.save(data_save_path, data)
np.save(label_save_path, true_labels)

# -------------------------- 打印数据和标签信息（严格对齐英文字母代码格式） --------------------------
print("="*60)
# 打印数据大小（形状）
print(f"数据形状（试次数 × 通道数 × 时间点）：{data.shape}")
# 打印标签大小（形状）
print(f"标签形状（试次数）：{true_labels.shape}")
print("="*60)