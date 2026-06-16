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

# 固定随机种子
import random
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型与数据
from model.finestan import FineSTAN
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
SUBJECT_ID = 15
TASK_TYPE = "CCS-HI"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub15\epoch_400\cv_2026-05-04--15-28"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub15"

def load_model_weights(model, weights_path, device):
    weights_path = Path(weights_path).resolve()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def init_model_args(train_config, test_data, test_labels):
    network_args = train_config['network_args'].copy()
    selected_channel_idx = train_config.get('selected_channel_idx', list(range(test_data.shape[1])))
    network_args.update({
        'num_classes': len(np.unique(test_labels)),
        'num_samples': test_data.shape[2],
        'num_channels': test_data.shape[1],
        'selected_channel_idx': selected_channel_idx
    })
    return network_args

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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
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

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)

    network_args = init_model_args(cv_config, test_data, test_labels)
    model = FineSTAN(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_model_weights(model, weights_path, device)
    print("模型加载完成")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"特征 | 样本数：{features.shape[0]} | 维度：{features.shape[1]}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        max_iter=5000
    )
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("tab10", 5)

    if TASK_TYPE == "CCS-HI":
        class_names = ['一', '丨', '丿', '㇏', 'ㄥ']
    else:
        class_names = [str(i) for i in range(len(np.unique(labels)))]

    for i in range(5):
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

    plt.title('FineSTAN (Ours)', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_{TASK_TYPE.lower()}_sub{SUBJECT_ID}_final.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"已保存：{save_path}")

    plt.show()"""


"""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import torch
import os
import sys
from data.data_utils import load_BCI42_data

# 固定随机种子
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 配置参数
SUBJECT_ID = 15
TASK_TYPE = "CCS-HI"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub15"

test_data, test_labels = load_BCI42_data(DATA_ROOT, f"A{SUBJECT_ID:02d}E")
print(f"原始数据 | 形状：{test_data.shape} | 类别数：{len(np.unique(test_labels))}")

features_raw = test_data.reshape(test_data.shape[0], -1)
features_raw_pca = PCA(n_components=30, random_state=42).fit_transform(features_raw)

tsne_embedding_raw = TSNE(
    n_components=2,
    perplexity=15,
    learning_rate=100,
    init='pca',
    random_state=42,
    max_iter=5000
).fit_transform(features_raw_pca)

tsne_embedding_raw_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(tsne_embedding_raw)

plt.figure(figsize=(6, 6))
colors = sns.color_palette("tab10", 5)
class_names = ['一', '丨', '丿', '㇏', 'ㄥ']

for i in range(5):
    idx = test_labels == i
    plt.scatter(
        tsne_embedding_raw_scaled[idx, 0],
        tsne_embedding_raw_scaled[idx, 1],
        color=colors[i],
        label=class_names[i],
        s=40,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

plt.title('Raw Data', fontsize=16, fontweight='bold')
plt.xlabel('dim_1', fontsize=14)
plt.ylabel('dim_2', fontsize=14)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.legend(loc='lower left', framealpha=1)

plt.grid(alpha=0.3, linestyle='-')
plt.tight_layout()

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, f'CCS_tsne_raw_data_sub{SUBJECT_ID}_final.png')
plt.savefig(save_path, dpi=900, bbox_inches='tight')
print(f"原始数据t-SNE图已保存：{save_path}")

plt.show()"""


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

# 固定随机种子
import random
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.finestan import FineSTAN
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
SUBJECT_ID = 7
TASK_TYPE = "ELL-HI"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub07\epoch_100\cv_2026-05-03--15-55"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data_English"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub07"

def load_model_weights(model, weights_path, device):
    weights_path = Path(weights_path).resolve()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def init_model_args(train_config, test_data, test_labels):
    network_args = train_config['network_args'].copy()
    selected_channel_idx = train_config.get('selected_channel_idx', list(range(test_data.shape[1])))
    network_args.update({
        'num_classes': 10,
        'num_samples': test_data.shape[2],
        'num_channels': test_data.shape[1],
        'selected_channel_idx': selected_channel_idx
    })
    return network_args

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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    print(f"数据形状: {test_data.shape} | 类别: {np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init)

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)

    network_args = init_model_args(cv_config, test_data, test_labels)
    model = FineSTAN(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_model_weights(model, weights_path, device)
    print("模型加载完成")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"特征形状: {features.shape}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=100, init='pca', random_state=42, max_iter=5000)
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("tab10", 10)
    class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']

    for i in range(10):
        idx = labels == i
        plt.scatter(tsne_embedding_scaled[idx, 0], tsne_embedding_scaled[idx, 1],
                    color=colors[i], label=class_names[i], s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

    plt.title('FineSTAN (Ours)', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'tsne_ell_hi_sub7_final.png')
    print(f"已保存: {save_path}")
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.show()"""


"""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import torch
import os
import sys
from data.data_utils import load_BCI42_data

# 固定随机种子
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 配置参数
SUBJECT_ID = 7
TASK_TYPE = "ELL-HI"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data_English"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub07"

test_data, test_labels = load_BCI42_data(DATA_ROOT, f"A{SUBJECT_ID:02d}E")
print(f"原始数据 | 形状：{test_data.shape} | 类别数：{len(np.unique(test_labels))}")

features_raw = test_data.reshape(test_data.shape[0], -1)
features_raw_pca = PCA(n_components=30, random_state=42).fit_transform(features_raw)

tsne_embedding_raw = TSNE(
    n_components=2,
    perplexity=15,
    learning_rate=100,
    init='pca',
    random_state=42,
    max_iter=5000
).fit_transform(features_raw_pca)

tsne_embedding_raw_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(tsne_embedding_raw)

plt.figure(figsize=(6, 6))
colors = sns.color_palette("tab10", 10)
class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']

for i in range(10):
    idx = test_labels == i
    plt.scatter(
        tsne_embedding_raw_scaled[idx, 0],
        tsne_embedding_raw_scaled[idx, 1],
        color=colors[i],
        label=class_names[i],
        s=40,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

plt.title('Raw Data', fontsize=16, fontweight='bold')
plt.xlabel('dim_1', fontsize=14)
plt.ylabel('dim_2', fontsize=14)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.legend(loc='lower left', framealpha=1)

plt.grid(alpha=0.3, linestyle='-')
plt.tight_layout()

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, f'ELL_tsne_raw_data_sub{SUBJECT_ID}_final.png')
print(f"原始数据t-SNE图已保存：{save_path}")
plt.savefig(save_path, dpi=900, bbox_inches='tight')

plt.show()"""


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

# 固定随机种子
import random
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.finestan import FineSTAN
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
SUBJECT_ID = 10
TASK_TYPE = "SV-HI"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub10\epoch_400\cv_2026-05-28--06-55"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data_Pinyin"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub10"

def load_model_weights(model, weights_path, device):
    weights_path = Path(weights_path).resolve()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def init_model_args(train_config, test_data, test_labels):
    network_args = train_config['network_args'].copy()
    selected_channel_idx = train_config.get('selected_channel_idx', list(range(test_data.shape[1])))
    network_args.update({
        'num_classes': 6,
        'num_samples': test_data.shape[2],
        'num_channels': test_data.shape[1],
        'selected_channel_idx': selected_channel_idx
    })
    return network_args

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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    print(f"数据形状: {test_data.shape} | 类别: {np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init)

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)

    network_args = init_model_args(cv_config, test_data, test_labels)
    model = FineSTAN(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_model_weights(model, weights_path, device)
    print("模型加载完成")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"特征形状: {features.shape}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=100, init='pca', random_state=42, max_iter=5000)
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("tab10", 6)
    class_names = ['a', 'o', 'e', 'i', 'u', 'ü']

    for i in range(6):
        idx = labels == i
        plt.scatter(tsne_embedding_scaled[idx, 0], tsne_embedding_scaled[idx, 1],
                    color=colors[i], label=class_names[i], s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

    plt.title('FineSTAN (Ours)', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'tsne_sv_hi_sub{SUBJECT_ID}_final.png')
    print(f"已保存: {save_path}")
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.show()"""


"""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import torch
import os
import sys
from data.data_utils import load_BCI42_data

# 固定随机种子
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 配置参数
SUBJECT_ID = 10
TASK_TYPE = "SV-HI"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data_Pinyin"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub10"

test_data, test_labels = load_BCI42_data(DATA_ROOT, f"A{SUBJECT_ID:02d}E")
print(f"原始数据 | 形状：{test_data.shape} | 类别数：{len(np.unique(test_labels))}")

features_raw = test_data.reshape(test_data.shape[0], -1)
features_raw_pca = PCA(n_components=30, random_state=42).fit_transform(features_raw)

tsne_embedding_raw = TSNE(
    n_components=2,
    perplexity=15,
    learning_rate=100,
    init='pca',
    random_state=42,
    max_iter=5000
).fit_transform(features_raw_pca)

tsne_embedding_raw_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(tsne_embedding_raw)

plt.figure(figsize=(6, 6))
colors = sns.color_palette("tab10", 6)
class_names = ['a', 'o', 'e', 'i', 'u', 'ü']

for i in range(6):
    idx = test_labels == i
    plt.scatter(
        tsne_embedding_raw_scaled[idx, 0],
        tsne_embedding_raw_scaled[idx, 1],
        color=colors[i],
        label=class_names[i],
        s=40,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

plt.title('Raw Data', fontsize=16, fontweight='bold')
plt.xlabel('dim_1', fontsize=14)
plt.ylabel('dim_2', fontsize=14)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.legend(loc='lower left', framealpha=1)

plt.grid(alpha=0.3, linestyle='-')
plt.tight_layout()

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, f'SV_tsne_raw_data_sub{SUBJECT_ID}_final.png')
print(f"原始数据t-SNE图已保存：{save_path}")
plt.savefig(save_path, dpi=900, bbox_inches='tight')

plt.show()"""


# -*- coding: utf-8 -*-
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

# 固定随机种子
import random
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型与数据
from model.finestan import FineSTAN
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# 绘图全局配置
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
SUBJECT_ID = 7
TASK_TYPE = "ELL-HI"
MODEL_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub07\epoch_400\cv_2026-05-03--15-55"
DATA_ROOT = r"D:\EEG_HandWriting_Project_Code\EEG-TransNet-main\dataset\bci_handwriting_data_English"
SAVE_DIR = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\output\t-SNE_Plot_choice_model\sub07"

def load_model_weights(model, weights_path, device):
    weights_path = Path(weights_path).resolve()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def init_model_args(train_config, test_data):
    network_args = train_config['network_args'].copy()
    selected_channel_idx = train_config.get('selected_channel_idx', list(range(test_data.shape[1])))
    network_args.update({
        'num_classes': 10,
        'num_samples': test_data.shape[2],
        'num_channels': test_data.shape[1],
        'selected_channel_idx': selected_channel_idx
    })
    return network_args

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
            model(x)
            labels_list.append(y.numpy())
    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)

if __name__ == '__main__':
    test_datafile = f"A{SUBJECT_ID:02d}E"
    test_data, test_labels = load_BCI42_data(DATA_ROOT, test_datafile)
    print(f"Data shape: {test_data.shape} | Classes: {np.unique(test_labels)}")

    test_dataset = eegDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        worker_init_fn=worker_init,
        drop_last=False
    )

    config_path = os.path.join(MODEL_DIR, "cv_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cv_config = yaml.safe_load(f)

    network_args = init_model_args(cv_config, test_data)
    model = FineSTAN(**network_args).to(device)

    weights_path = os.path.join(MODEL_DIR, "cv_global_best.pth")
    model = load_model_weights(model, weights_path, device)
    print("FineSTAN loaded successfully")

    features, labels = extract_features_with_hook(model, test_loader)
    print(f"Features shape: {features.shape}")

    features = PCA(n_components=30, random_state=42).fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=100,
        init='pca',
        random_state=42,
        max_iter=5000
    )
    tsne_embedding = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_embedding_scaled = scaler.fit_transform(tsne_embedding)

    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("tab10", 10)
    class_names = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']

    for i in range(10):
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

    plt.title('FineSTAN (Ours)', fontsize=16, fontweight='bold')
    plt.xlabel('dim_1', fontsize=14)
    plt.ylabel('dim_2', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='best', framealpha=1)
    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'tsne_ell_hi_sub7_1.png')
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


