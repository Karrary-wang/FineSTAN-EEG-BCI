"""
作者：王帆
时间：2022. 9.8
"""
# Task ：hand writing  heng shu dian pie zhe  five classification session1--train data
import mne
from mne import event
from mne import epochs
from mne import label
from mne.io.fiff import raw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

factor_new = 1e-3
init_block_size = 1000  # 计划采用降采样 将1000降到250Hz  sampling frequency *times
subject_events = {}
data_path = 'F:/Hand_Writing_Imagery_Pinyin_Data_New/Event_&_ID'   # data path
data_files = ['A0' + str(i) + 'E.bdf' for i in range(5, 6)]  # only one T/E.bdf files

save_path = 'F:/FineSTAN/dataset/bci_handwriting_data_Pinyin_New'   # 获得.py的数据和标签
event_description = {'1': "a", '2': "o", '3': "e", '4': "i", '5': "u", '6': "ü"}  # task
# event .txt 文件
files = os.listdir(data_path)
filtered_files = [file for file in files if file.endswith('A05E_Event.txt')]
data_files_bdf = [file for file in files if file.endswith('A05E.bdf')]
index = 0
# create MNE Event
# 遍历所有的事件文件，并将事件信息存储到字典中 这部分主要建造events
for file in filtered_files:
    index=index+1
    file_path = os.path.join(data_path, file)
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            sample, event_type = map(int, line.split())
            events.append([sample, 0, event_type])
    events = np.array(events)  # event 文件 包含 切数据的索引和真实标签
    subject_event = os.path.splitext(file)[0]  # 获取文件名去掉后缀作为被试名称
    subject_events[index] = events  # 将events存储到字典中，以subject_name作为键
    # print(subject_event)


# Read data
for idx, Data_file in enumerate(data_files_bdf):
    # read data
    raw_data = mne.io.read_raw_bdf(os.path.join(data_path, Data_file), preload=True, verbose=False)

    # print(raw_data)
    raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)  # 将jiang  data v  uv
    # data filter
    raw_data.filter(l_freq=1, h_freq=40,
                    fir_design='firwin')  # fir_design='firwin'  method='iir'   l_freq 和 h_freq 是 MNE 函数所定义的参数名 带通滤波
    # filtered_data = exponential_running_standardize(raw_data.get_data().T, factor_new=factor_new, init_block_size=init_block_size)
    # raw_data = mne.io.RawArray(filtered_data.T, raw_data.info)
    # 降采样到 250Hz
    #raw_data.resample(250)
    # 创建 epochs
    event_id = dict({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})  # 1-6 stand for  a、o、e、i、u、ü
    # epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3, reject=None, baseline=None, preload=True)
    raw_events=subject_events[idx+1]
    # raw_data.notch_filter(freqs=50)
    tmin, tmax = 0, 4

    raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True,  baseline=None,
                            preload=True)
    # 降采样到 250Hz
    epochs_resampled = raw_epochs.copy().resample(250, npad="auto")  # 降采样250Hz

    true_labels = epochs_resampled.events[:, -1]   # 标签映射到 1、2、3、4、5、6
    data = epochs_resampled.get_data(
        copy=False)  # [n_epochs, n_channels, n_times] #raw_epochs.get_data() 是从 Epochs 对象中提取具体 EEG 数据的基础方法
    # 如果用到降采样就把下面这行代码注释掉
    #data = data[:, :, :-1]
    print(data.shape)

    np.save(os.path.join(save_path, Data_file[:-4] + '_data.npy'), data)
    np.save(os.path.join(save_path, Data_file[:-4] + '_label.npy'), true_labels)