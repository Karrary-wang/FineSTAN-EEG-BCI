import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import random
import copy
import gc
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.finestan import FineSTAN, compute_channel_importance, select_motor_channels


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

    def to_python_type(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python_type(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    safe_config = to_python_type(copy.deepcopy(dictToWrite))
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(safe_config, f, allow_unicode=True)


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
        raise ValueError(f"路径 {data_root} 下未找到训练数据文件")

    print(f"共识别 {len(subject_list)} 名受试者: {[s[0] for s in subject_list]}")
    return subject_list


def auto_init_model(config, data, labels, device):
    network_args = config['network_args'].copy()
    num_channels = network_args.get('num_channels', data.shape[1])
    num_samples = network_args.get('num_samples', data.shape[2])
    num_classes = network_args.get('num_classes', 10)
    top_k = network_args.get('top_k_channels', 16)

    sample_ratio = 0.2
    data_tensor = torch.from_numpy(data[:int(sample_ratio * len(data))]).float().to(device)
    label_tensor = torch.from_numpy(labels[:int(sample_ratio * len(labels))]).long().to(device)
    importance = compute_channel_importance(data_tensor, label_tensor)
    selected_channel_idx = select_motor_channels(importance, top_k=top_k).cpu().numpy()
    print(f"通道选择完成，选取 {top_k} 个通道: {selected_channel_idx.tolist()}")

    valid_params = [
        'num_classes', 'num_samples', 'num_channels', 'embed_dim', 'pool_size',
        'pool_stride', 'num_heads', 'fc_ratio', 'depth', 'attn_drop', 'fc_drop',
        'classify_drop', 'pool_dropout', 'attn_norm_type', 'temp_coeff_init',
        'global_weight_init', 'selected_channel_idx', 'window_size', 'seg_fusion_init'
    ]
    network_args = {k: v for k, v in network_args.items() if k in valid_params}
    network_args.update({
        'num_classes': num_classes,
        'num_samples': num_samples,
        'num_channels': num_channels,
        'selected_channel_idx': selected_channel_idx
    })
    return network_args, selected_channel_idx


def train_cv(config, sub_id, datafile, epochs, fixed_split_seed=None):
    out_folder = config['out_folder']
    timestamp = time.strftime('%Y-%m-%d--%H-%M', time.localtime())
    out_path = os.path.normpath(
        os.path.join(out_folder, config['network'], sub_id, f'epoch_{epochs}', f'cv_{timestamp}'))
    os.makedirs(out_path, exist_ok=True)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and config.get('preferred_device') == 'gpu') else 'cpu')
    print(f"\n[{sub_id} | Epoch:{epochs}] 训练设备: {device}")

    data, labels = load_BCI42_data(config['data_path'], datafile)
    k_folds = config.get('k_folds', 5)
    split_seed = fixed_split_seed if fixed_split_seed else config['random_seed']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=split_seed)

    fold_best_acc, fold_best_epoch = [], []
    global_best_acc = 0.0
    global_best_model_path = os.path.join(out_path, 'cv_global_best.pth')
    weight_save_dir = os.path.join(out_path, 'v3_weights')
    os.makedirs(weight_save_dir, exist_ok=True)

    config['network_args'], selected_channel_idx = auto_init_model(config, data, labels, device)
    config['selected_channel_idx'] = selected_channel_idx
    dictToYaml(os.path.join(out_path, 'cv_config.yaml'), config)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\n----- {sub_id} 第 {fold + 1}/{k_folds} 折 -----")
        train_data_fold, train_labels_fold = data[train_idx], labels[train_idx]
        val_data_fold, val_labels_fold = data[val_idx], labels[val_idx]

        def worker_init(worker_id):
            np.random.seed(split_seed + worker_id)

        train_loader = DataLoader(
            eegDataset(train_data_fold, train_labels_fold),
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=worker_init
        )
        val_loader = DataLoader(
            eegDataset(val_data_fold, val_labels_fold),
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=worker_init
        )

        net = FineSTAN(**config['network_args']).to(device)
        loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.05))

        param_groups = [
            {'params': [p for n, p in net.named_parameters() if 'scale_logits' not in n]},
            {
                'params': [p for n, p in net.named_parameters() if 'scale_logits' in n],
                'lr': config['lr'] * 10,
                'weight_decay': 1e-3
            }
        ]
        optimizer = optim.Adam(param_groups, lr=config['lr'], weight_decay=float(config['weight_decay']))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_acc, best_epoch = 0.0, 0
        for epoch in range(epochs):
            net.train()
            train_loss, train_correct, total_train = 0.0, 0, 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device, dtype=torch.float32)
                batch_labels = batch_labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = net(batch_data)
                loss = loss_func(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_data.size(0)
                train_correct += (torch.max(outputs, 1)[1] == batch_labels).sum().item()
                total_train += batch_labels.size(0)

            net.eval()
            val_loss, val_correct, total_val = 0.0, 0, 0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device, dtype=torch.float32)
                    batch_labels = batch_labels.to(device, dtype=torch.long)
                    outputs = net(batch_data)
                    loss = loss_func(outputs, batch_labels)

                    val_loss += loss.item() * batch_data.size(0)
                    val_correct += (torch.max(outputs, 1)[1] == batch_labels).sum().item()
                    total_val += batch_labels.size(0)

            scheduler.step()
            train_acc = train_correct / total_train
            val_acc = val_correct / total_val
            lr = scheduler.get_last_lr()[0]

            print(f"轮{epoch+1:3d}/{epochs:3d} | LR:{lr:.6f} | 训练损失:{train_loss/total_train:.4f} | 训练ACC:{train_acc:.4f} | 验证ACC:{val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(out_path, f'fold_{fold + 1}_best.pth'))

        try:
            fold_scale_weights = net.get_scale_weights()
            np.save(os.path.join(weight_save_dir, f'fold_{fold + 1}_scale_weights.npy'), fold_scale_weights)
        except (AttributeError, Exception):
            pass

        print(f"本折最佳验证ACC: {best_val_acc:.4f}")
        fold_best_acc.append(best_val_acc)
        fold_best_epoch.append(best_epoch)

        if best_val_acc > global_best_acc:
            global_best_acc = best_val_acc
            torch.save(net.state_dict(), global_best_model_path)
            try:
                np.save(os.path.join(weight_save_dir, 'global_best_scale_weights.npy'), net.get_scale_weights())
            except (AttributeError, Exception):
                pass

        del net, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    cv_mean_acc = np.mean(fold_best_acc)
    cv_std_acc = np.std(fold_best_acc)
    avg_best_epoch = int(np.round(np.mean(fold_best_epoch)))
    print(f"\n[{sub_id}] 五折交叉验证汇总 | 平均ACC: {cv_mean_acc:.4f}±{cv_std_acc:.4f}")

    cv_summary = {
        'sub_id': sub_id,
        'split_seed': split_seed,
        'global_best_acc': global_best_acc,
        'cv_mean_acc': float(cv_mean_acc),
        'cv_std_acc': float(cv_std_acc),
        'fold_best_acc': fold_best_acc,
        'fold_best_epoch': fold_best_epoch,
        'avg_best_epoch': avg_best_epoch,
        'weight_save_dir': weight_save_dir,
        'selected_channel_idx': selected_channel_idx,
        'train_epochs': epochs
    }
    dictToYaml(os.path.join(out_path, 'cv_summary.yaml'), cv_summary)
    return out_path, avg_best_epoch, selected_channel_idx


def train_final(config, sub_id, datafile, avg_best_epoch, epochs, selected_channel_idx):
    out_folder = config['out_folder']
    timestamp = time.strftime('%Y-%m-%d--%H-%M', time.localtime())
    out_path = os.path.normpath(
        os.path.join(out_folder, config['network'], sub_id, f'epoch_{epochs}', f'final_{timestamp}'))
    os.makedirs(out_path, exist_ok=True)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and config.get('preferred_device') == 'gpu') else 'cpu')
    full_data, full_labels = load_BCI42_data(config['data_path'], datafile)

    network_args = config['network_args'].copy()
    num_channels = network_args.get('num_channels', full_data.shape[1])
    num_samples = network_args.get('num_samples', full_data.shape[2])
    num_classes = network_args.get('num_classes', 5)

    valid_params = [
        'num_classes', 'num_samples', 'num_channels', 'embed_dim', 'pool_size',
        'pool_stride', 'num_heads', 'fc_ratio', 'depth', 'attn_drop', 'fc_drop',
        'classify_drop', 'pool_dropout', 'attn_norm_type', 'temp_coeff_init',
        'global_weight_init', 'selected_channel_idx', 'window_size', 'seg_fusion_init'
    ]
    network_args = {k: v for k, v in network_args.items() if k in valid_params}
    network_args.update({
        'num_classes': num_classes,
        'num_samples': num_samples,
        'num_channels': num_channels,
        'selected_channel_idx': selected_channel_idx
    })
    config['network_args'] = network_args
    config['selected_channel_idx'] = selected_channel_idx
    dictToYaml(os.path.join(out_path, 'final_config.yaml'), config)

    def worker_init(worker_id):
        np.random.seed(config['random_seed'] + worker_id)

    full_loader = DataLoader(
        eegDataset(full_data, full_labels),
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=worker_init
    )

    net = FineSTAN(**config['network_args']).to(device)
    loss_func = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.05))

    param_groups = [
        {'params': [p for n, p in net.named_parameters() if 'scale_logits' not in n]},
        {
            'params': [p for n, p in net.named_parameters() if 'scale_logits' in n],
            'lr': config['lr'] * 10,
            'weight_decay': 1e-3
        }
    ]
    optimizer = optim.Adam(param_groups, lr=config['lr'], weight_decay=float(config['weight_decay']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=avg_best_epoch, eta_min=1e-6)

    train_log = []
    final_train_acc = 0.0
    final_train_loss = 0.0

    print(f"\n----- {sub_id} 最终模型训练（共 {avg_best_epoch} 轮） -----")
    for epoch in range(avg_best_epoch):
        net.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for batch_data, batch_labels in full_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(batch_data)
            loss = loss_func(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_data.size(0)
            train_correct += (torch.max(outputs, 1)[1] == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        scheduler.step()
        train_acc = train_correct / total_train
        lr = scheduler.get_last_lr()[0]
        train_loss_avg = train_loss / total_train

        train_log.append({
            'epoch': epoch + 1,
            'lr': lr,
            'train_loss': train_loss_avg,
            'train_acc': train_acc
        })

        print(f"轮{epoch+1:3d}/{avg_best_epoch:3d} | LR:{lr:.6f} | 训练损失:{train_loss_avg:.4f} | 训练ACC:{train_acc:.4f}")

        if epoch == avg_best_epoch - 1:
            final_train_acc = train_acc
            final_train_loss = train_loss_avg

    final_model_path = os.path.join(out_path, 'final_model.pth')
    torch.save(net.state_dict(), final_model_path)
    try:
        final_scale_weights = net.get_scale_weights()
        np.save(os.path.join(out_path, 'final_scale_weights.npy'), final_scale_weights)
    except (AttributeError, Exception):
        pass

    final_train_summary = {
        'sub_id': sub_id,
        'total_epochs': avg_best_epoch,
        'final_train_acc': final_train_acc,
        'final_train_loss': final_train_loss,
        'train_log': train_log
    }
    dictToYaml(os.path.join(out_path, 'final_train_summary.yaml'), final_train_summary)

    del net, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


if __name__ == '__main__':
    # 路径配置
    DATA_ROOT = r'D:/EEG_HandWriting_Project_Code/FineSTAN/dataset/bci_handwriting_data_English'
    CONFIG_PATH = r'D:/EEG_HandWriting_Project_Code/FineSTAN/config/HI_Ablation_experiment.yaml'
    EPOCHS_LIST = list(range(100, 600, 100))

    # 加载配置
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = yaml.full_load(f)

    # 全局默认超参
    config.setdefault('batch_size', 64)
    config.setdefault('lr', 0.0002)
    config.setdefault('random_seed', 0)
    config.setdefault('weight_decay', 1e-4)
    config.setdefault('label_smoothing', 0.05)
    config.setdefault('preferred_device', 'gpu')
    config.setdefault('k_folds', 5)
    config['data_path'] = DATA_ROOT
    # 固定为新模型
    config['network'] = 'FineSTAN'

    # 扫描受试者
    subject_list = scan_subjects_data(DATA_ROOT)
    completed_subs = []

    for sub_id, train_datafile in subject_list:
        if sub_id in completed_subs:
            print(f"【{sub_id}】已完成，跳过")
            continue

        for train_epochs in EPOCHS_LIST:
            print(f"\n{'='*60}")
            print(f"模型: {config['network']} | 受试者: {sub_id} | 总轮数: {train_epochs}")
            print(f"{'='*60}")

            seed = config['random_seed'] + int(sub_id.replace('sub', '')) + train_epochs
            setRandom(seed)

            try:
                cv_out_path, avg_best_epoch, selected_channel_idx = train_cv(config, sub_id, train_datafile, train_epochs, config['random_seed'])
                final_out_path = train_final(config, sub_id, train_datafile, avg_best_epoch, train_epochs, selected_channel_idx)
                print(f"【{sub_id} | {train_epochs}】训练完成！")
            except Exception as e:
                print(f"【{sub_id} | {train_epochs}】训练失败：{str(e)}")
                gc.collect()
                torch.cuda.empty_cache()
                continue