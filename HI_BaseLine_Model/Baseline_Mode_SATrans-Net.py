
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

# 固定随机种子，保证可复现性
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入SATransNet模型与数据工具
from model.SATransNet import SATransNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# 绘图全局设置（与CTNet/FineSTAN统一）
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 配置区域（已按你提供的信息修改） ==========
SUBJECT_ID = 15
TASK_TYPE = "CCS-HI"
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATrans-Net_sub15\epoch_600\cv_2026-04-24--11-26"
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data"
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATransNet_sub15"


# =====================================================


# ========== 工具函数 ==========
def load_satransnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 过滤匹配的权重，避免形状不匹配报错
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if
                           k in model_state_dict and v.shape == model_state_dict[k].shape}

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"成功加载权重，匹配参数数：{len(filtered_state_dict)}/{len(model_state_dict)}")
    return model


def init_satransnet_args(config_path):
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

    # 找到最后一个Linear层（分类头）
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fc_layer = module

    # 注册前向钩子，提取该层输入
    handle = fc_layer.register_forward_hook(lambda m, i, o: features_list.append(i[0].cpu().numpy()))

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            model(x)
            labels_list.append(y.numpy())

    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)


# ========== 主流程 ==========
if __name__ == '__main__':
    # 1. 加载数据（sub15的CCS-HI笔画数据）
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    test_data = test_data[:, np.newaxis, :, :]  # SATransNet输入格式: (N, 1, C, T)
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

    # 2. 初始化并加载SATransNet模型
    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_satransnet_args(config_path)
    model = SATransNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_satransnet_weights(model, weights_path, device)
    print("SATransNet模型加载完成")

    # 3. 提取特征
    features, labels = extract_features_with_hook(model, test_loader)
    print(f"特征 | 样本数：{features.shape[0]} | 维度：{features.shape[1]}")

    # 4. PCA + t-SNE（与CTNet/FineSTAN参数完全一致，保证对比公平）
    features = PCA(n_components=30, random_state=42).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        n_iter=5000  # 适配旧版本scikit-learn
    )
    tsne_embedding = tsne.fit_transform(features)

    # 5. 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    # 6. 绘图（与其他模型格式/配色统一）
    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))
    colors = sns.color_palette("tab10", num_classes)

    # 笔画5分类类名
    if TASK_TYPE == "CCS-HI":
        class_names = ['一', '丨', '丿', '㇏', 'ㄥ']
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

    plt.title('SATransNet', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(prop=chinese_font, loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    # 保存图片
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_satransnet_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"已保存：{save_path}")

    plt.show()
    """

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

# 1. 替换为 SATransNet 模型
from model.SATransNet import SATransNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 核心配置修改 ==========
SUBJECT_ID = 10
TASK_TYPE = "SV-HI"
# 你指定的 SATransNet 模型路径
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATrans-Net_sub10\epoch_600\cv_2026-04-25--05-20"
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data_Pinyin"
# 保存路径（对应SATransNet+被试10）
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATransNet_sub10"
# ====================================

# 2. 替换为 SATransNet 权重加载函数
def load_satransnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    # 去除多GPU训练的 module. 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 过滤匹配的权重
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"SATransNet 权重加载完成 | 匹配参数：{len(filtered_state_dict)}/{len(model_state_dict)}")
    return model

# 初始化模型参数
def init_model_args(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    return cv_config['network_args']

def worker_init(worker_id):
    np.random.seed(42 + worker_id)

# 特征提取函数（保持不变）
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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

# ========== 主流程 ==========
if __name__ == '__main__':
    # 加载数据
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

    # 加载 SATransNet 模型
    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_model_args(config_path)
    model = SATransNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_satransnet_weights(model, weights_path, device)

    # 提取特征
    features, labels = extract_features_with_hook(model, test_loader)
    print(f"Features | Samples: {features.shape[0]} | Dimensions: {features.shape[1]}")

    # PCA + t-SNE
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

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    # 绘图
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

    # 3. 标题改为 SATransNet
    plt.title('SATrans-Net', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    # 保存图片
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_satransnet_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    plt.show()


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

# 导入 SATransNet 模型
from model.SATransNet import SATransNet
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 核心配置（已修改为 sub12 英文字母任务）==========
SUBJECT_ID = 12
TASK_TYPE = "ELL-HI"
# 你指定的 SATransNet 模型路径
MODEL_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATransNet_sub12\epoch_300\cv_2026-04-26--11-24"
# 英文字母数据集路径
DATA_ROOT = r"F:\EEG-TransNet-main\dataset\bci_handwriting_data_English"
# 保存路径
SAVE_DIR = r"F:\EEG-TransNet-main\BaseLine_Model_t-SNE\SATransNet_sub12"
# ==========================================================

# SATransNet 权重加载函数
def load_satransnet_weights(model, weights_path, device):
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    # 去除多GPU训练的 module. 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 过滤匹配的权重
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"SATransNet 权重加载完成 | 匹配参数：{len(filtered_state_dict)}/{len(model_state_dict)}")
    return model

# 初始化模型参数
def init_model_args(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)
    return cv_config['network_args']

def worker_init(worker_id):
    np.random.seed(42 + worker_id)

# 特征提取函数
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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

# ========== 主流程 ==========
if __name__ == '__main__':
    # 加载 sub12 英文字母数据
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

    # 加载 SATransNet 模型
    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    network_args = init_model_args(config_path)
    model = SATransNet(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_satransnet_weights(model, weights_path, device)

    # 提取特征
    features, labels = extract_features_with_hook(model, test_loader)
    print(f"Features | Samples: {features.shape[0]} | Dimensions: {features.shape[1]}")

    # PCA + t-SNE
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

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    # 绘图
    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))
    colors = sns.color_palette("tab10", num_classes)

    # ELL-HI 英文字母10分类类名
    if TASK_TYPE == "ELL-HI":
        class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    elif TASK_TYPE == "SV-HI":
        class_names = ['a', 'o', 'e', 'i', 'u', 'v']
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

    # 图表标题
    plt.title('SATrans-Net', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    # 保存图片
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_satransnet_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    plt.show()"""