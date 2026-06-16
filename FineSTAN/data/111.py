"""
作者：王帆
时间：2022. 9.8
"""
import numpy as np

# 定义两个标签文件的路径
label_path_e = r'D:\EEG_HandWriting_Project_Code\FineSTAN\dataset\bci_handwriting_data_English\A15E_label.npy'
label_path_t = r'D:\EEG_HandWriting_Project_Code\FineSTAN\dataset\bci_handwriting_data_English\A15T_label.npy'

# 加载两个标签文件
labels_e = np.load(label_path_e)
labels_t = np.load(label_path_t)

# 打印A01E_label的完整信息
print("="*50)
print("A01E_label 完整信息")
print("="*50)
print("shape：", labels_e.shape)
print("所有标签值：")
print(labels_e)  # 直接打印整个标签数组

# 打印A01T_label的完整信息
print("\n" + "="*50)
print("A01T_label 完整信息")
print("="*50)
print("shape：", labels_t.shape)
print("所有标签值：")
print(labels_t)  # 直接打印整个标签数组