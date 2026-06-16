"""
作者：王帆
时间：2022. 9.8
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import random
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# ===================== 路径配置 & 模型导入 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.MCTD import MCTD  # 导入MCTD模型（需确保MCTD.py在model目录下）


# ===================== 工具函数（完全复用） =====================
def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def dictToYaml(filePath, dictToWrite):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)


def scan_subjects_data(data_root):
    data_root = Path(data_root)
    subject_list = []
    for file in data_root.iterdir():
        if file.is_file() and file.name.startswith('A') and file.name.endswith('T_data.npy'):
            base_name = file.name.replace('_data.npy', '')
            sub_num = base_name[1:3]
            sub_id = f'sub{sub_num}'
            train_datafile = base_name
            subject_list.append((sub_id, train_datafile))
    subject_list.sort(key=lambda x: int(x[0].replace('sub', '')))
    if not subject_list:
        raise ValueError(f"在 {data_root} 中未找到AxxT_data.npy格式的训练数据文件！")
    print(f"共识别到 {len(subject_list)} 个受试者：{[s[0] for s in subject_list]}")
    return subject_list


# ===================== 核心训练函数（适配MCTD多尺度输出） =====================
def train_cv(config, sub_id, datafile, epochs):
    out_folder = config['out_folder']
    timestamp = config.get('timestamp', time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    out_path = os.path.normpath(
        os.path.join(out_folder, config['network'], sub_id, f'epoch_{epochs}', f'cv_{timestamp}'))
    os.makedirs(out_path, exist_ok=True)
    dictToYaml(os.path.join(out_path, 'cv_config.yaml'), config)

    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(
        f"【{sub_id} | Epoch:{epochs}】设备：{device} | GPU数量：{torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    # 加载并适配数据维度
    data, labels = load_BCI42_data(config['data_path'], datafile)
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]
    elif len(data.shape) == 4 and data.shape[1] != 1:
        data = np.transpose(data, (0, 3, 1, 2))

    k_folds = config.get('k_folds', 5)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config['random_seed'])

    fold_best_acc, fold_best_epoch = [], []
    global_best_acc = 0.0
    global_best_model_path = os.path.join(out_path, 'cv_global_best.pth')
    print(f"【{sub_id} | Epoch:{epochs}】CV路径：{out_path} | 数据形状：{data.shape}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\n【{sub_id} | Epoch:{epochs}】第 {fold + 1}/{k_folds} 折训练启动")
        train_data, train_labels = data[train_idx], labels[train_idx]
        val_data, val_labels = data[val_idx], labels[val_idx]

        def worker_init(worker_id):
            np.random.seed(config['random_seed'] + worker_id)

        train_loader = DataLoader(eegDataset(train_data, train_labels), batch_size=config['batch_size'],
                                  shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init)
        val_loader = DataLoader(eegDataset(val_data, val_labels), batch_size=config['batch_size'],
                                shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=worker_init)

        # 初始化MCTD模型
        net = MCTD(**config['network_args']).to(device)

        # 模型预热：验证输入输出维度（MCTD返回(特征列表, 输出列表)）
        net.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 32, 1000).to(device)
            feats, outputs = net(dummy_input)
            # 验证多尺度输出维度
            assert len(outputs) == 5, "MCTD应返回5个尺度的输出"
        net.train()

        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            net = nn.DataParallel(net)
        current_net = net.module if isinstance(net, nn.DataParallel) else net

        optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=float(config['weight_decay']))
        loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.05))

        best_val_acc, best_epoch = 0.0, 0
        for epoch in range(epochs):
            net.train()
            train_loss, train_correct, total_train = 0.0, 0, 0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device, dtype=torch.float32)
                batch_labels = batch_labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                # MCTD返回(特征列表, 输出列表)，取输出列表平均作为最终输出
                _, outputs = net(batch_data)
                outputs_avg = torch.stack(outputs).mean(dim=0)  # 5个尺度输出平均
                loss = loss_func(outputs_avg, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_data.size(0)
                train_correct += (torch.max(outputs_avg, 1)[1] == batch_labels).sum().item()
                total_train += batch_labels.size(0)

            # 验证阶段
            net.eval()
            val_loss, val_correct, total_val = 0.0, 0, 0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device, dtype=torch.float32)
                    batch_labels = batch_labels.to(device, dtype=torch.long)
                    _, outputs = net(batch_data)
                    outputs_avg = torch.stack(outputs).mean(dim=0)
                    loss = loss_func(outputs_avg, batch_labels)
                    val_loss += loss.item() * batch_data.size(0)
                    val_correct += (torch.max(outputs_avg, 1)[1] == batch_labels).sum().item()
                    total_val += batch_labels.size(0)

            train_acc = train_correct / total_train
            val_acc = val_correct / total_val
            print(f"【{sub_id} | Epoch:{epochs}】折{fold + 1} | 轮{epoch + 1:3d}/{epochs:3d} | "
                  f"训练损失：{train_loss / total_train:.4f} | 训练ACC：{train_acc:.4f} | 验证ACC：{val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(current_net.state_dict(), os.path.join(out_path, f'fold_{fold + 1}_best.pth'))

        print(f"【{sub_id} | Epoch:{epochs}】第{fold + 1}折最佳ACC：{best_val_acc:.4f}")
        fold_best_acc.append(best_val_acc)
        fold_best_epoch.append(best_epoch)

        if best_val_acc > global_best_acc:
            global_best_acc = best_val_acc
            torch.save(current_net.state_dict(), global_best_model_path)
            print(f"【{sub_id} | Epoch:{epochs}】全局最佳更新！当前最佳ACC：{global_best_acc:.4f}")

    cv_mean_acc = np.mean(fold_best_acc)
    cv_std_acc = np.std(fold_best_acc)
    avg_best_epoch = int(np.round(np.mean(fold_best_epoch)))
    print(f"\n【{sub_id} | Epoch:{epochs}】CV汇总 | 平均ACC：{cv_mean_acc:.4f}±{cv_std_acc:.4f} | "
          f"全局最佳ACC：{global_best_acc:.4f} | 平均最佳轮数：{avg_best_epoch}")

    cv_summary = {'sub_id': sub_id, 'global_best_acc': global_best_acc,
                  'cv_mean_acc': float(cv_mean_acc), 'cv_std_acc': float(cv_std_acc),
                  'fold_best_acc': fold_best_acc, 'fold_best_epoch': fold_best_epoch,
                  'avg_best_epoch': avg_best_epoch, 'train_epochs': epochs}
    dictToYaml(os.path.join(out_path, 'cv_summary.yaml'), cv_summary)
    return out_path, avg_best_epoch


def train_final(config, sub_id, datafile, avg_best_epoch, epochs):
    out_folder = config['out_folder']
    timestamp = config.get('timestamp', time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    out_path = os.path.normpath(
        os.path.join(out_folder, config['network'], sub_id, f'epoch_{epochs}', f'final_{timestamp}'))
    os.makedirs(out_path, exist_ok=True)

    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    full_data, full_labels = load_BCI42_data(config['data_path'], datafile)
    if len(full_data.shape) == 3:
        full_data = full_data[:, np.newaxis, :, :]
    elif len(full_data.shape) == 4 and full_data.shape[1] != 1:
        full_data = np.transpose(full_data, (0, 3, 1, 2))
    print(f"【{sub_id} | Epoch:{epochs}】最终模型训练 | 路径：{out_path} | 设备：{device} | 数据形状：{full_data.shape}")

    def worker_init(worker_id):
        np.random.seed(config['random_seed'] + worker_id)

    full_loader = DataLoader(eegDataset(full_data, full_labels), batch_size=config['batch_size'],
                             shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init)

    # 初始化MCTD模型
    net = MCTD(**config['network_args']).to(device)

    # 模型预热
    net.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        feats, outputs = net(dummy_input)
    net.train()

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        net = nn.DataParallel(net)
    current_net = net.module if isinstance(net, nn.DataParallel) else net

    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=float(config['weight_decay']))
    loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.05))

    print(f"\n【{sub_id} | Epoch:{epochs}】开始最终模型训练（共{avg_best_epoch}轮）")
    for epoch in range(avg_best_epoch):
        net.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for batch_data, batch_labels in full_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            # MCTD多尺度输出平均
            _, outputs = net(batch_data)
            outputs_avg = torch.stack(outputs).mean(dim=0)
            loss = loss_func(outputs_avg, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_data.size(0)
            train_correct += (torch.max(outputs_avg, 1)[1] == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        train_acc = train_correct / total_train
        print(f"【{sub_id} | Epoch:{epochs}】最终模型 | 轮{epoch + 1:3d}/{avg_best_epoch:3d} | "
              f"训练损失：{train_loss / total_train:.4f} | 训练ACC：{train_acc:.4f}")

    final_model_path = os.path.join(out_path, 'final_model.pth')
    torch.save(current_net.state_dict(), final_model_path)
    final_config = config.copy()
    final_config.update({'epochs': avg_best_epoch, 'sub_id': sub_id, 'datafile': datafile, 'cv_train_epochs': epochs})
    print(f"【{sub_id} | Epoch:{epochs}】最终模型路径：{final_model_path}")
    dictToYaml(os.path.join(out_path, 'final_train_config.yaml'), final_config)
    return out_path


# ===================== 主函数（修改为单epoch值运行） =====================
if __name__ == '__main__':
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data'
    # MCTD配置文件路径（可复用CTNet配置文件，或新建Hand_Writing_MCTD.yaml）
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_MCTD.yaml'

    # ========== 关键修改1：替换遍历列表为单个epoch值 ==========
    # 注释掉原有的列表，改为指定单个值（比如300）
    # EPOCHS_LIST = list(range(100, 500, 100))  # 注释掉原有列表
    TARGET_EPOCH = 500  # 你想要运行的具体epoch值，可修改为任意数值（如200、500等）

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)

    # MCTD默认配置
    config.setdefault('batch_size', 16)
    config.setdefault('lr', 0.0001)  # MCTD建议学习率1e-4
    config.setdefault('random_seed', 0)
    config.setdefault('weight_decay', 1e-5)
    config.setdefault('label_smoothing', 0.01)
    config.setdefault('timestamp', time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    config.setdefault('network', 'MCTD')  # 网络名称改为MCTD
    config.setdefault('device', 'cuda:0')
    config.setdefault('k_folds', 5)
    config.setdefault('network_args', {})

    # MCTD核心参数（适配32通道、1000采样点、250Hz手写想象数据）
    config['network_args'].setdefault('in_chans', 32)  # 32电极通道
    config['network_args'].setdefault('emb_size', 40)  # 特征嵌入维度
    config['network_args'].setdefault('n_classes', 10)  # 手写想象10分类
    config['network_args'].setdefault('depth', 6)  # Transformer编码器层数

    config['data_path'] = DATA_ROOT

    # 扫描受试者数据
    subject_list = scan_subjects_data(DATA_ROOT)
    # 已完成训练的受试者列表（复用原CTNet列表）
    completed_subs = []

    # 批量训练每个受试者
    for sub_id, train_datafile in subject_list:
        if sub_id in completed_subs:
            print(f"【{sub_id}】已完成，跳过")
            continue

        # ========== 关键修改2：移除epoch列表遍历，直接使用单个TARGET_EPOCH ==========
        # 注释掉原有的for train_epochs in EPOCHS_LIST循环，改为直接使用TARGET_EPOCH
        print(f"\n=====================================")
        print(f"开始训练 MCTD ：【{sub_id}】 | Epoch数：{TARGET_EPOCH}")
        print(f"=====================================")

        setRandom(config['random_seed'] + int(sub_id.replace('sub', '')) + TARGET_EPOCH)

        try:
            cv_out_path, avg_best_epoch = train_cv(config, sub_id, train_datafile, TARGET_EPOCH)
            final_out_path = train_final(config, sub_id, train_datafile, avg_best_epoch, TARGET_EPOCH)
            print(f"\n【{sub_id} | Epoch:{TARGET_EPOCH}】训练完成 | CV路径：{cv_out_path} | 最终模型路径：{final_out_path}")
        except Exception as e:
            print(f"\n【{sub_id} | Epoch:{TARGET_EPOCH}】训练失败：{str(e)}")
            import traceback

            traceback.print_exc()
            continue