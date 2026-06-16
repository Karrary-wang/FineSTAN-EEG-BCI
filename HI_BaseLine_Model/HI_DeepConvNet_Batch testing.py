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
from pathlib import Path
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
# 导入DeepConvNet模型
from model.DeepConvNet import DeepConvNet


def dictToYaml(filePath, dictToWrite):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)


def scan_subjects_data(data_root):
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
    if not subject_list:
        raise ValueError(f"在 {data_root} 中未找到AxxE_data.npy格式的测试数据文件！")
    print(f"共识别到 {len(subject_list)} 个受试者：{[s[0] for s in subject_list]}")
    return subject_list


def scan_model_paths(out_root, network_name):
    model_info_list = []
    out_root = Path(out_root)
    net_root = out_root / network_name
    if not net_root.exists():
        raise ValueError(f"模型根目录不存在：{net_root}")

    for sub_dir in net_root.iterdir():
        if not sub_dir.is_dir() or not sub_dir.name.startswith('sub'):
            continue
        sub_id = sub_dir.name
        for epoch_dir in sub_dir.iterdir():
            if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch_'):
                continue
            epoch_tag = epoch_dir.name
            for cv_dir in epoch_dir.iterdir():
                if not cv_dir.is_dir() or not cv_dir.name.startswith('cv_'):
                    continue
                model_path = cv_dir / 'cv_global_best.pth'
                config_path = cv_dir / 'cv_config.yaml'
                if model_path.exists() and config_path.exists():
                    model_info_list.append({
                        'sub_id': sub_id,
                        'epoch_tag': epoch_tag,
                        'model_path': str(model_path),
                        'config_path': str(config_path)
                    })
    if not model_info_list:
        raise ValueError("未扫描到有效模型文件 cv_global_best.pth")
    return model_info_list


def test_single_model(config, model_path, datafile, sub_id, epoch_tag):
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f"\n【{sub_id} | {epoch_tag}】开始测试，设备：{device}")

    # 数据维度处理，与训练保持一致
    data, labels = load_BCI42_data(config['data_path'], datafile)
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]
    elif len(data.shape) == 4 and data.shape[1] != 1:
        data = np.transpose(data, (0, 3, 1, 2))

    test_dataset = eegDataset(data, labels)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                              shuffle=False, pin_memory=True, num_workers=0)

    # 初始化模型并加载权重
    net = DeepConvNet(**config['network_args']).to(device)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            outputs = net(batch_data)
            pred = torch.max(outputs, 1)[1]
            total_correct += (pred == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    acc = total_correct / total_samples
    print(f"【{sub_id} | {epoch_tag}】测试准确率：{acc:.4f}")
    return acc


if __name__ == '__main__':
    # 路径和配置，与训练脚本严格对应
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data_English'
    OUT_ROOT = r'F:/EEG-TransNet-main/output/bci_handwriting_data_English'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_DeepConvNet.yaml'
    # 模型文件夹名称，必须和训练配置里 network 一致
    NETWORK_NAME = 'DeepConvNet'

    # 加载全局配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        global_config = yaml.full_load(f)

    # 扫描测试集与模型文件
    subject_list = scan_subjects_data(DATA_ROOT)
    model_info_list = scan_model_paths(OUT_ROOT, NETWORK_NAME)

    result_summary = {}
    for sub_id, test_datafile in subject_list:
        sub_models = [m for m in model_info_list if m['sub_id'] == sub_id]
        if not sub_models:
            print(f"\n【{sub_id}】未找到对应模型，跳过")
            continue

        result_summary[sub_id] = {}
        for model_info in sub_models:
            epoch_tag = model_info['epoch_tag']
            model_path = model_info['model_path']
            cfg_path = model_info['config_path']

            # 加载训练时保存的配置，保证参数完全匹配
            with open(cfg_path, 'r', encoding='utf-8') as f:
                model_config = yaml.full_load(f)

            test_acc = test_single_model(model_config, model_path, test_datafile, sub_id, epoch_tag)
            result_summary[sub_id][epoch_tag] = test_acc

    # 汇总输出所有结果
    print("\n==================== 全部测试结果汇总 ====================")
    for sub_id, epoch_res in result_summary.items():
        print(f"\n{sub_id}:")
        for epoch_tag, acc in epoch_res.items():
            print(f"  {epoch_tag} : ACC = {acc:.4f}")