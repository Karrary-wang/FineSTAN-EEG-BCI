import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os
import resampy

def load_BCI42_data(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')

    data = np.load(data_path)
    label = np.load(label_path).squeeze()-1  # 去除数组中维度为 1 的所有维度  并且将标签改为0 1 2 3 4  五个类别

    print(data_file, 'load success')

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def load_HGD_data(dataset_path, data_file, label_file):
    data = []
    label = []
    #Todo 文件名根据需要更改
    data_path = os.path.join(dataset_path, data_file)
    label_path = os.path.join(dataset_path, label_file)

    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    print(data_file, 'load success')

    # Shuffle shuffle_data 函数的作用通常是对数据进行随机化（打乱顺序）。这是为了避免数据在训练模型时因为顺序的原因产生偏差或模型过拟合到某个特定的输入顺序
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def shuffle_data(data, label):   # 对输入数据和标签进行随机打乱，以便在训练机器学习模型时提高模型的泛化能力
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label