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
from sklearn.metrics import cohen_kappa_score, f1_score

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.SATransNet import SATransNet

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
        raise ValueError('未找到训练数据')
    print(f'共识别到 {len(subject_list)} 个受试者：{[s[0] for s in subject_list]}')
    return subject_list

def train_cv(config, sub_id, datafile, epochs):
    out_folder = config['out_folder']
    timestamp = time.strftime('%Y-%m-%d--%H-%M')
    out_path = os.path.normpath(
        os.path.join(out_folder, 'SATransNet', sub_id, f'epoch_{epochs}', f'cv_{timestamp}')
    )
    os.makedirs(out_path, exist_ok=True)
    dictToYaml(os.path.join(out_path, 'cv_config.yaml'), config)

    device = torch.device(config.get('device', 'cuda:0'))
    print(f'[{sub_id}] 设备：{device}')

    data, labels = load_BCI42_data(config['data_path'], datafile)
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]

    k_folds = config.get('k_folds', 5)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config['random_seed'])

    fold_best_acc = []
    fold_best_epoch = []
    global_best_acc = 0.0
    global_best_path = os.path.join(out_path, 'cv_global_best.pth')

    print(f'[{sub_id}] 数据形状：{data.shape}')

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f'\n[{sub_id}] 第 {fold+1}/{k_folds} 折')
        train_X, val_X = data[train_idx], data[val_idx]
        train_y, val_y = labels[train_idx], labels[val_idx]

        def worker_init(worker_id):
            np.random.seed(config['random_seed'] + worker_id)

        train_loader = DataLoader(
            eegDataset(train_X, train_y),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init,
            pin_memory=True
        )
        val_loader = DataLoader(
            eegDataset(val_X, val_y),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        net = SATransNet(**config['network_args']).to(device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        current_net = net.module if isinstance(net, nn.DataParallel) else net

        optimizer = optim.AdamW(
            net.parameters(),
            lr=config['lr'],
            weight_decay=float(config['weight_decay'])
        )

        loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            net.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.long)
                optimizer.zero_grad()
                out = net(batch_X)
                loss = loss_func(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, pred = torch.max(out, 1)
                train_total += batch_y.size(0)
                train_correct += (pred == batch_y).sum().item()

            net.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_pred = []
            all_label = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device, dtype=torch.float32)
                    batch_y = batch_y.to(device, dtype=torch.long)
                    out = net(batch_X)
                    loss = loss_func(out, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    _, pred = torch.max(out, 1)
                    val_total += batch_y.size(0)
                    val_correct += (pred == batch_y).sum().item()
                    all_pred.extend(pred.cpu().numpy())
                    all_label.extend(batch_y.cpu().numpy())

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            kappa = cohen_kappa_score(all_label, all_pred)
            f1 = f1_score(all_label, all_pred, average='macro', zero_division=0)

            print(
                f'[{sub_id}] 折{fold+1} 轮{epoch+1:2d}/{epochs} | '
                f'训练Loss:{train_loss/train_total:.4f} 训练ACC:{train_acc:.4f} | '
                f'验证ACC:{val_acc:.4f} Kappa:{kappa:.4f} F1:{f1:.4f}'
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(current_net.state_dict(), os.path.join(out_path, f'fold_{fold+1}_best.pth'))

        print(f'[{sub_id}] 第{fold+1}折最佳：ACC={best_val_acc:.4f} Epoch={best_epoch}')
        fold_best_acc.append(best_val_acc)
        fold_best_epoch.append(best_epoch)

        if best_val_acc > global_best_acc:
            global_best_acc = best_val_acc
            torch.save(current_net.state_dict(), global_best_path)

    mean_acc = np.mean(fold_best_acc)
    std_acc = np.std(fold_best_acc)
    avg_best_epoch = int(np.round(np.mean(fold_best_epoch)))
    print(f'\n[{sub_id}] 5折平均：{mean_acc:.4f}±{std_acc:.4f}')

    dictToYaml(os.path.join(out_path, 'cv_summary.yaml'), {
        'sub_id': sub_id,
        'best_acc': float(global_best_acc),
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
        'avg_best_epoch': avg_best_epoch
    })
    return out_path, avg_best_epoch

def train_final(config, sub_id, datafile, avg_best_epoch, epochs):
    out_folder = config['out_folder']
    timestamp = config['timestamp']
    out_path = os.path.normpath(
        os.path.join(out_folder, 'SATransNet', sub_id, f'epoch_{epochs}', f'final_{timestamp}')
    )
    os.makedirs(out_path, exist_ok=True)
    device = torch.device(config.get('device', 'cuda:0'))

    data, labels = load_BCI42_data(config['data_path'], datafile)
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]

    def worker_init(worker_id):
        np.random.seed(config['random_seed'] + worker_id)

    loader = DataLoader(
        eegDataset(data, labels),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True
    )

    net = SATransNet(**config['network_args']).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    current_net = net.module if isinstance(net, nn.DataParallel) else net

    optimizer = optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=float(config['weight_decay']))
    loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

    print(f'\n[{sub_id}] 最终训练（{avg_best_epoch}轮）')
    for epoch in range(avg_best_epoch):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in loader:
            # ===================== 修复位置 =====================
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = net(X)
            loss = loss_func(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        acc = correct / total
        print(f'[{sub_id}] 最终轮{epoch+1:2d}/{avg_best_epoch} | Loss:{total_loss/total:.4f} ACC:{acc:.4f}')

    torch.save(current_net.state_dict(), os.path.join(out_path, 'final_model.pth'))
    print(f'[{sub_id}] 最终模型保存完成！')
    return out_path

if __name__ == '__main__':
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data_English'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_SATrans-Net.yaml'

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)

    config['timestamp'] = time.strftime('%Y-%m-%d--%H-%M')
    config['data_path'] = DATA_ROOT
    EPOCHS_LIST = list(range(100, 700, 100))
    subject_list = scan_subjects_data(DATA_ROOT)

    for sub_id, datafile in subject_list:
        for ep in EPOCHS_LIST:
            print(f'\n=====================================')
            print(f' 训练 {sub_id} Epoch={ep}')
            print(f'=====================================')
            setRandom(config['random_seed'] + int(sub_id.replace('sub','')) + ep)
            try:
                cv_path, best_ep = train_cv(config, sub_id, datafile, ep)
                train_final(config, sub_id, datafile, best_ep, ep)
                print(f'[{sub_id}] 训练完成！')
            except Exception as e:
                print(f'失败：{e}')
                import traceback
                traceback.print_exc()