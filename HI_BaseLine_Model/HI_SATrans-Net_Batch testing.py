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
import random
from pathlib import Path
from torch.utils.data import DataLoader

# ===================== 路径配置 & 模型导入 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 导入数据工具
try:
    from data.data_utils import load_BCI42_data
    from data.dataset import eegDataset
except ImportError:
    print("警告: 未找到 data.data_utils 或 data.dataset，请确保目录结构正确。")
    sys.exit(1)

# 导入 SATransNet 模型
try:
    from model.SATransNet import SATransNet
except ImportError:
    print("错误: 无法导入 SATransNet 模型。请检查 model/SATransNet.py 是否存在。")
    sys.exit(1)

# ===================== 工具函数 =====================
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

def load_model_weights(model, weight_path, device):
    try:
        state_dict = torch.load(weight_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"权重文件加载失败 {weight_path}: {e}")

    # 模型预热
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        _ = model(dummy_input)

    # 过滤匹配的权重
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # 加载
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"成功加载权重：{os.path.basename(weight_path)} | 匹配参数：{len(filtered_state_dict)}/{len(model_state_dict)}")
    return model

def scan_subjects_models(out_root, network_name='SATransNet'):
    out_root = Path(out_root)
    subject_models = {}

    for sub_dir in out_root.glob(f"{network_name}/sub*"):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name
        epoch_dirs = [d for d in sub_dir.glob("epoch_*") if d.is_dir()]
        epoch_list = []
        model_paths = {}

        for epoch_dir in epoch_dirs:
            try:
                epoch_num = int(epoch_dir.name.replace('epoch_', ''))
            except ValueError:
                continue

            global_best_path = list(epoch_dir.glob("cv_*/cv_global_best.pth"))
            if global_best_path:
                epoch_list.append(epoch_num)
                model_paths[epoch_num] = global_best_path[0]

        if epoch_list:
            epoch_list.sort()
            subject_models[sub_id] = {
                'epochs': epoch_list,
                'model_paths': model_paths
            }
    return subject_models

# ===================== 核心测试函数 =====================
def test_subject(config, sub_id, datafile, epoch_num, model_path):
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f"\n----- 测试被试 {sub_id} Epoch:{epoch_num} -----")

    # 加载数据
    try:
        data, labels = load_BCI42_data(config['data_path'], datafile)
        if len(data.shape) == 3:
            data = data[:, np.newaxis, :, :]
    except Exception as e:
        print(f"被试 {sub_id} 数据加载失败：{str(e)}")
        return None

    # 数据加载器
    test_loader = DataLoader(
        eegDataset(data, labels),
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # 模型初始化 + 加载
    try:
        net = SATransNet(**config['network_args']).to(device)
        net = load_model_weights(net, model_path, device)
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return None

    # 测试
    net.eval()
    test_correct, total_test = 0, 0
    test_loss = 0.0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            outputs = net(batch_data)
            loss = loss_func(outputs, batch_labels)

            test_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs, 1)
            total_test += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

    if total_test == 0:
        return None

    test_acc = test_correct / total_test
    avg_loss = test_loss / total_test
    print(f"测试完成 | ACC：{test_acc:.4f} | 损失：{avg_loss:.4f}")

    return {
        'sub_id': sub_id,
        'epoch': epoch_num,
        'test_acc': test_acc,
        'test_loss': avg_loss,
        'total_samples': total_test,
        'correct_samples': test_correct,
        'model_path': str(model_path)
    }

# ===================== 主函数 =====================
if __name__ == '__main__':
    # ================= 配置区域 =================
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data_English'
    OUT_ROOT = r'F:/EEG-TransNet-main/output/bci_handwriting_data_English'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_SATrans-Net.yaml'
    NETWORK_NAME = 'SATransNet'

    # 加载配置
    if not os.path.exists(CONFIG_PATH):
        print(f"提示: 配置文件 {CONFIG_PATH} 不存在，将使用默认参数。")
        config = {}
    else:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.full_load(f)

    # 默认配置
    config.setdefault('batch_size', 16)
    config.setdefault('device', 'cuda:0')
    config.setdefault('random_seed', 0)
    config.setdefault('network_args', {})
    config['data_path'] = DATA_ROOT

    # SATransNet 核心参数 (默认值，若yaml中有则覆盖)
    args = config['network_args']
    args.setdefault('n_classes', 6)
    args.setdefault('heads', 4)
    args.setdefault('emb_size', 32)
    args.setdefault('depth', 3)
    args.setdefault('eeg1_f1', 16)
    args.setdefault('eeg1_kernel_size', 64)
    args.setdefault('eeg1_D', 2)
    args.setdefault('eeg1_pooling_size1', 10)
    args.setdefault('eeg1_pooling_size2', 4)
    args.setdefault('eeg1_dropout_rate', 0.2)
    args.setdefault('eeg1_number_channel', 32)
    args.setdefault('flatten_eeg1', 800)

    setRandom(config['random_seed'])

    # ================= 开始测试 =================
    print(f"开始批量测试所有被试的所有 epoch 模型（版本：{NETWORK_NAME}）...")
    subject_models = scan_subjects_models(OUT_ROOT, NETWORK_NAME)

    if not subject_models:
        print(f"错误: 在 {OUT_ROOT} 中未找到 {NETWORK_NAME} 模型！")
        sys.exit(1)

    print(f"共扫描到 {len(subject_models)} 个被试的模型。")
    all_test_results = []

    # 逐个被试测试
    for sub_id in subject_models.keys():
        print(f"\n=====================================")
        print(f"处理被试：{sub_id}")
        print(f"=====================================")

        sub_num = sub_id.replace('sub', '')
        test_datafile = f"A{sub_num}E"
        epoch_list = subject_models[sub_id]['epochs']
        model_paths = subject_models[sub_id]['model_paths']

        print(f"被试 {sub_id} 有效 epoch：{epoch_list}")

        # 测试每个 epoch
        for epoch_num in epoch_list:
            model_path = model_paths[epoch_num]
            try:
                result = test_subject(config, sub_id, test_datafile, epoch_num, model_path)
                if result:
                    all_test_results.append(result)
            except Exception as e:
                print(f"测试失败：{str(e)}")
                continue

    # ================= 结果汇总 =================
    if all_test_results:
        timestamp = time.strftime('%Y-%m-%d--%H-%M')
        result_filename = f"batch_test_results_{NETWORK_NAME}_{timestamp}.yaml"
        result_path = os.path.join(OUT_ROOT, NETWORK_NAME, result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        serializable_results = []
        for res in all_test_results:
            res_copy = res.copy()
            res_copy['test_acc'] = float(res_copy['test_acc'])
            res_copy['test_loss'] = float(res_copy['test_loss'])
            serializable_results.append(res_copy)

        summary_data = {
            'test_time': timestamp,
            'network': NETWORK_NAME,
            'total_subjects': len(subject_models),
            'total_tests': len(all_test_results),
            'results': serializable_results
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary_data, f, allow_unicode=True)

        print(f"\n=====================================")
        print(f"批量测试完成！")
        print(f"=====================================")
        print(f"测试被试数：{len(subject_models)}")
        print(f"有效测试数：{len(all_test_results)}")
        print(f"结果保存：{result_path}")

        # 统计
        avg_acc = np.mean([r['test_acc'] for r in all_test_results])
        std_acc = np.std([r['test_acc'] for r in all_test_results])
        print(f"所有模型平均ACC：{avg_acc:.4f} ± {std_acc:.4f}")

        # 每个被试最佳
        sub_best = {}
        for r in all_test_results:
            sid = r['sub_id']
            if sid not in sub_best or r['test_acc'] > sub_best[sid]['test_acc']:
                sub_best[sid] = r

        print(f"\n各被试最佳结果：")
        best_acc_list = []
        for sid, r in sorted(sub_best.items()):
            print(f" {sid} | 最佳Epoch：{r['epoch']:3d} | ACC：{r['test_acc']:.4f}")
            best_acc_list.append(r['test_acc'])

        if best_acc_list:
            print(f"\n跨被试平均最佳ACC：{np.mean(best_acc_list):.4f} ± {np.std(best_acc_list):.4f}")

    else:
        print("\n无有效测试结果！")