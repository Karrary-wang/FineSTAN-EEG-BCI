import numpy as np
import random
import os


def standardize_labels(label):
    """标准化标签为 0～n-1（n为类别数）"""
    # 处理空标签场景（避免报错）
    if len(label) == 0:
        return label
    unique_labels = np.sort(np.unique(label))
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        print(f"Remapping labels from {unique_labels} to 0～{len(unique_labels) - 1}")
        label_map = {old: new for new, old in enumerate(unique_labels)}
        # 优化：使用np.vectorize提升大规模标签映射效率
        label_vectorized = np.vectorize(lambda x: label_map[x])
        label = label_vectorized(label)
    return label.astype(np.int64)  # 确保标签为整数类型，适配后续模型训练


def shuffle_data(data, label, seed=None):
    """打乱数据和标签（保持一一对应，支持固定种子复现）"""
    # 校验数据和标签长度一致
    assert len(data) == len(label), f"Data length ({len(data)}) != Label length ({len(label)})"
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(data))
    return data[indices], label[indices]


def load_BCI42_data(dataset_path, data_file, seed=None):
    """加载BCI42数据集（自动标准化标签，校验维度）"""
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')

    # 校验文件是否存在（避免文件路径错误）
    assert os.path.exists(data_path), f"Data file not found: {data_path}"
    assert os.path.exists(label_path), f"Label file not found: {label_path}"

    data = np.load(data_path)
    label = np.load(label_path).squeeze()
    label = standardize_labels(label)

    assert data.ndim == 3, f"Data must be (N, C, T), got {data.shape}"
    data, label = shuffle_data(data, label, seed=seed)

    print(f"{data_file} loaded successfully | Data shape: {data.shape}, Labels: {np.unique(label)}")
    return data, label


def load_HGD_data(dataset_path, data_file, label_file, seed=None):
    """加载HGD数据集（自动标准化标签，校验维度）"""
    data_path = os.path.join(dataset_path, data_file)
    label_path = os.path.join(dataset_path, label_file)

    # 校验文件是否存在（避免文件路径错误）
    assert os.path.exists(data_path), f"Data file not found: {data_path}"
    assert os.path.exists(label_path), f"Label file not found: {label_path}"

    data = np.load(data_path)
    label = np.load(label_path).squeeze()
    label = standardize_labels(label)

    assert data.ndim == 3, f"Data must be (N, C, T), got {data.shape}"
    data, label = shuffle_data(data, label, seed=seed)

    print(f"{data_file} loaded successfully | Data shape: {data.shape}, Labels: {np.unique(label)}")
    return data, label