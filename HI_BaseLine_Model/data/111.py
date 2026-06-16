"""
作者：王帆
时间：2022. 9.8
"""
import numpy as np  # 导入numpy库

# 1. 定义文件的完整路径（注意用r开头表示“原始字符串”，避免路径中的\转义错误）
file_path = r"F:\HandWriting_Imagery_Deep_model_Collect\EEG-TransNet-main\dataset\bci_iv_2a\A01E_data.npy"

# 2. 加载文件（numpy.load直接读取.npy文件）
data = np.load(file_path)

# 3. 查看shape（直接打印data的shape属性）
print("A01E_data 的 shape：", data.shape)