"""
作者：王帆
时间：2022. 9.8
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from scipy import signal

# ============ 1. 路径&模块导入（和训练代码一致） ============
PROJECT_ROOT = r"F:\EEG-TransNet-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.FBCNet import FBCNetModel
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset

# ============ 2. 和训练完全一致的频段/滤波参数 ============
freq_bands = [
    [1, 4], [4, 8], [8, 12], [12, 16], [16, 20],
    [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]
]
nBands = len(freq_bands)
sfreq = 250
order = 4

def bandpass_filter(data, l_freq, h_freq, sfreq, order=4):
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data, axis=-1)

def convert_to_fb_data(data_3d):
    n_epoch, n_chan, n_time = data_3d.shape
    fb_data = np.zeros((n_epoch, n_chan, n_time, nBands), dtype=np.float32)
    for band_idx, (l_freq, h_freq) in enumerate(freq_bands):
        for ch in range(n_chan):
            fb_data[:, ch, :, band_idx] = bandpass_filter(
                data_3d[:, ch, :], l_freq, h_freq, sfreq, order
            )
    return fb_data

# ============ 3. 工具函数 ============
def setRandom(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dictToYaml(filePath, dictToWrite):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)

# ============ 4. 单受试者测试函数 ============
def test_subject(config, sub_id, test_datafile, model_path):
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f"\n------------------------------")
    print(f"测试受试者：{sub_id} | 设备：{device}")
    print(f"模型路径：{model_path}")

    # 1. 加载测试数据（原始3维）
    data, labels = load_BCI42_data(config['data_path'], test_datafile)
    print(f"原始测试数据形状: {data.shape}, 标签形状: {labels.shape}")

    # 2. 自动频段分解（和训练一致）
    if data.ndim == 3:
        print("自动多频段滤波分解...")
        data = convert_to_fb_data(data)
        print(f"分解后: {data.shape}")
    elif data.ndim == 4 and data.shape[-1] == nBands:
        print("已为4维频段数据，跳过滤波")
    else:
        raise RuntimeError(f"不支持数据维度: {data.shape}")

    # 3. 补维到模型输入：(N,1,32,1000,10)
    data = data[:, np.newaxis, :, :, :]
    print(f"模型输入形状: {data.shape}")

    # 4. DataLoader
    test_loader = DataLoader(
        eegDataset(data, labels),
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # 5. 加载模型
    net = FBCNetModel(**config['network_args']).to(device)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)
            outputs = net(batch_data)
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # 6. 计算指标
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    print(f"\n【{sub_id}】测试ACC：{acc:.4f}")
    print("混淆矩阵：")
    print(cm)

    return {
        'sub_id': sub_id,
        'acc': float(acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

# ============ 5. 批量遍历所有受试者测试 ============
def scan_subjects_test(data_root):
    data_root = Path(data_root)
    subject_list = []
    for file in data_root.iterdir():
        if file.is_file() and file.name.startswith('A') and file.name.endswith('E_data.npy'):
            base_name = file.name.replace('_data.npy', '')
            sub_num = base_name[1:3]
            sub_id = f'sub{sub_num}'
            test_datafile = base_name
            subject_list.append((sub_id, test_datafile))
    subject_list.sort(key=lambda x: int(x[0].replace('sub', '')))
    print(f"共识别到 {len(subject_list)} 个测试受试者：{[s[0] for s in subject_list]}")
    return subject_list

# ============ 主程序 ============
if __name__ == '__main__':
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data_English'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_FBCNet.yaml'
    # 你训练时用的 epochs（用来定位模型文件夹）
    TRAIN_EPOCHS = 100

    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)
    config['data_path'] = DATA_ROOT
    config.setdefault('device', 'cuda:0')
    config.setdefault('batch_size', 32)

    # 受试者列表（找 *E_data.npy）
    subject_list = scan_subjects_test(DATA_ROOT)

    # 结果保存目录
    timestamp = time.strftime('%Y-%m-%d--%H-%M')
    out_test_dir = os.path.join(config['out_folder'], config['network'], f'test_{timestamp}')
    os.makedirs(out_test_dir, exist_ok=True)

    all_results = []
    for sub_id, test_datafile in subject_list:
        # 自动定位该受试者最佳 final_model.pth
        model_path = os.path.join(
            config['out_folder'],
            config['network'],
            sub_id,
            f'epoch_{TRAIN_EPOCHS}',
            f'final_{timestamp}',  # 注意：训练和测试timestamp要一致，或手动指定
            'final_model.pth'
        )
        # 如果上面路径不对，你可以手动写死：
        # model_path = r"F:\EEG-TransNet-main\output\FBCNetModel\sub01\epoch_100\final_xxxx\final_model.pth"

        try:
            res = test_subject(config, sub_id, test_datafile, model_path)
            all_results.append(res)
        except Exception as e:
            print(f"❌ {sub_id} 测试失败：{e}")
            import traceback
            traceback.print_exc()

    # 汇总所有受试者结果
    mean_acc = np.mean([r['acc'] for r in all_results])
    std_acc = np.std([r['acc'] for r in all_results])
    print(f"\n==============================")
    print(f"所有受试者平均ACC：{mean_acc:.4f} ± {std_acc:.4f}")

    # 保存汇总结果
    summary = {
        'timestamp': timestamp,
        'train_epochs': TRAIN_EPOCHS,
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
        'subject_results': all_results
    }
    dictToYaml(os.path.join(out_test_dir, 'test_summary.yaml'), summary)
    print(f"✅ 测试结果已保存到：{out_test_dir}")