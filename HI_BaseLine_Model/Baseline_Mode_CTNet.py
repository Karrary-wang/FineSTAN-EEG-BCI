
"""import torch
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from matplotlib import rcParams
from pathlib import Path
from matplotlib.font_manager import FontProperties

import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.ctnet import CTNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 配置 ==========
SUBJECT_ID = 15
TASK_TYPE = "CCS-HI"
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\epoch_500\cv_2026-03-09--15-09"
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data"
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\sub15"


# ========== 工具函数 ==========
def load_ctnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)

    # 修复位置编码不匹配问题
    cnn_output_shape = None

    def hook(module, input, output):
        nonlocal cnn_output_shape
        cnn_output_shape = output.shape

    # 找到最后一个卷积层
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    handle = last_conv.register_forward_hook(hook)

    # 运行dummy输入获取CNN输出形状
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        try:
            _, _ = model(dummy_input)
        except RuntimeError:
            pass

    handle.remove()

    # 计算实际序列长度并重建位置编码
    actual_seq_len = cnn_output_shape[2] * cnn_output_shape[3]
    model.position.encoding = torch.nn.Parameter(
        torch.randn(1, actual_seq_len, model.position.encoding.shape[2]),
        requires_grad=False
    )

    # 加载权重（跳过位置编码，其他参数完全匹配）
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and k != 'position.encoding':
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)

    # 最终验证
    with torch.no_grad():
        _, _ = model(dummy_input)

    return model


def init_ctnet_args(config_path):
    """"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    return cv_config['network_args']


def worker_init(worker_id):
    np.random.seed(42 + worker_id)


def extract_features_with_hook(model, dataloader):
    model.eval()
    features_list = []
    labels_list = []
    fc_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fc_layer = module
    handle = fc_layer.register_forward_hook(lambda m, i, o: features_list.append(i[0].cpu().numpy()))
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            _, _ = model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)


# ========== 主流程 ==========
if __name__ == '__main__':
    # 1. 加载数据
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    test_data = test_data[:, np.newaxis, :, :]  # CTNet输入格式: (N, 1, C, T)
    print(f"数据 | 形状：{test_data.shape} | 类别：{np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=device.type == 'cuda',
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=False
    )

    # 2. 加载模型（直接使用cv_config.yaml中的完整参数初始化）
    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_ctnet_args(config_path)
    model = CTNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_ctnet_weights(model, weights_path, device)
    print("CTNet模型加载完成")

    # 3. 提取特征
    features, labels = extract_features_with_hook(model, test_loader)
    print(f"特征 | 样本数：{features.shape[0]} | 维度：{features.shape[1]}")

    # 4. PCA + t-SNE（与FineSTAN完全一致，修复scikit-learn版本兼容问题）
    features = PCA(n_components=30, random_state=42).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        n_iter=5000  # 旧版本scikit-learn使用n_iter而非max_iter
    )
    tsne_embedding = tsne.fit_transform(features)

    # 5. 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    # 6. 绘图
    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))
    colors = sns.color_palette("tab10", num_classes)

    if TASK_TYPE == "CCS-HI":
        class_names = ['一', '丨', '丿', '㇏', 'ㄥ']
    elif TASK_TYPE == "SV-HI":
        class_names = ['a', 'o', 'e', 'i', 'u', 'ü']
    elif TASK_TYPE == "ELL-HI":
        class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    else:
        class_names = [str(i) for i in range(num_classes)]

    for i in range(num_classes):
        idx = labels == i
        plt.scatter(
            tsne_embedding_scaled[idx, 0],
            tsne_embedding_scaled[idx, 1],
            color=colors[i],
            label=class_names[i],
            s=40,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )

    plt.title('CTNet', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(prop=chinese_font, loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    # 保存
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_ctnet_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"已保存：{save_path}")

    plt.show()"""

import torch
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from matplotlib import rcParams
from pathlib import Path
from matplotlib.font_manager import FontProperties

import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.ctnet import CTNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 已修正配置 ==========
SUBJECT_ID = 10
TASK_TYPE = "SV-HI"
# 你指定的模型路径
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\sub_10_SV\cv_2026-03-09--15-09"
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data_Pinyin"
# 保存路径对应被试10
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\sub10"
# ====================================


def load_ctnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)

    cnn_output_shape = None

    def hook(module, input, output):
        nonlocal cnn_output_shape
        cnn_output_shape = output.shape

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    handle = last_conv.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        try:
            _, _ = model(dummy_input)
        except RuntimeError:
            pass

    handle.remove()

    actual_seq_len = cnn_output_shape[2] * cnn_output_shape[3]
    model.position.encoding = torch.nn.Parameter(
        torch.randn(1, actual_seq_len, model.position.encoding.shape[2]),
        requires_grad=False
    )

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and k != 'position.encoding':
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)

    with torch.no_grad():
        _, _ = model(dummy_input)

    return model


def init_ctnet_args(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    return cv_config['network_args']


def worker_init(worker_id):
    np.random.seed(42 + worker_id)


def extract_features_with_hook(model, dataloader):
    model.eval()
    features_list = []
    labels_list = []
    fc_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fc_layer = module
    handle = fc_layer.register_forward_hook(lambda m, i, o: features_list.append(i[0].cpu().numpy()))
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            _, _ = model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)


if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    test_data = test_data[:, np.newaxis, :, :]
    print(f"Data shape: {test_data.shape} | Classes: {np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=device.type == 'cuda',
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=False
    )

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_ctnet_args(config_path)
    model = CTNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_ctnet_weights(model, weights_path, device)
    print("CTNet model loaded successfully")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"Features | Samples: {features.shape[0]} | Dimensions: {features.shape[1]}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        n_iter=5000
    )
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))
    colors = sns.color_palette("tab10", num_classes)

    # SV-HI 6分类类名
    if TASK_TYPE == "SV-HI":
        class_names = ['a', 'o', 'e', 'i', 'u', 'ü']
    elif TASK_TYPE == "ELL-HI":
        class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    else:
        class_names = [str(i) for i in range(num_classes)]

    for i in range(num_classes):
        idx = labels == i
        plt.scatter(
            tsne_embedding_scaled[idx, 0],
            tsne_embedding_scaled[idx, 1],
            color=colors[i],
            label=class_names[i],
            s=40,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )

    plt.title('CTNet', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_ctnet_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    plt.show()


# -*- coding: utf-8 -*-
"""import torch
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from matplotlib import rcParams
from pathlib import Path

import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.ctnet import CTNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 已修改为你指定的参数 ==========
SUBJECT_ID = 12  # 被试改为12
TASK_TYPE = "ELL-HI"
# 模型路径100%用你给的
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\sub_12_ELL\epoch_400\cv_2026-03-08--18-33"
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data_English"
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\sub12"  # 保存路径对应sub12
# ============================================


def load_ctnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)

    cnn_output_shape = None

    def hook(module, input, output):
        nonlocal cnn_output_shape
        cnn_output_shape = output.shape

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    handle = last_conv.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        try:
            _, _ = model(dummy_input)
        except RuntimeError:
            pass

    handle.remove()

    actual_seq_len = cnn_output_shape[2] * cnn_output_shape[3]
    model.position.encoding = torch.nn.Parameter(
        torch.randn(1, actual_seq_len, model.position.encoding.shape[2]),
        requires_grad=False
    )

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and k != 'position.encoding':
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)

    with torch.no_grad():
        _, _ = model(dummy_input)

    return model


def init_ctnet_args(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    return cv_config['network_args']


def worker_init(worker_id):
    np.random.seed(42 + worker_id)


def extract_features_with_hook(model, dataloader):
    model.eval()
    features_list = []
    labels_list = []
    fc_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fc_layer = module
    handle = fc_layer.register_forward_hook(lambda m, i, o: features_list.append(i[0].cpu().numpy()))
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            _, _ = model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)


if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    test_data = test_data[:, np.newaxis, :, :]
    print(f"Data shape: {test_data.shape} | Classes: {np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=device.type == 'cuda',
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=False
    )

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_ctnet_args(config_path)
    model = CTNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_ctnet_weights(model, weights_path, device)
    print("CTNet model loaded successfully")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"Features | Samples: {features.shape[0]} | Dimensions: {features.shape[1]}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        n_iter=5000
    )
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))
    colors = sns.color_palette("tab10", num_classes)

    class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']

    for i in range(num_classes):
        idx = labels == i
        plt.scatter(
            tsne_embedding_scaled[idx, 0],
            tsne_embedding_scaled[idx, 1],
            color=colors[i],
            label=class_names[i],
            s=40,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )

    plt.title('CTNet', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(loc='best', framealpha=1, fontsize=10)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_ctnet_ell-hi_sub{SUBJECT_ID}_final1.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    plt.show() """


