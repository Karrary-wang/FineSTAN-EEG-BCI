import sys
import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import time
import glob
from pathlib import Path
from torch.utils.data import DataLoader

# ===================== 路径配置 & 模型导入 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.EEGConformer import EEGConformer  # 导入最新的EEGConformer


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
    """
    安全加载模型权重，适配动态层和参数不匹配问题
    """
    # 加载权重文件
    state_dict = torch.load(weight_path, map_location=device)

    # 预热模型（初始化动态层）
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)
        _ = model(dummy_input)  # 触发classification_head.fc1初始化

    # 加载权重（忽略不匹配的键，适配结构变化）
    model_state_dict = model.state_dict()
    # 过滤掉权重中不存在于模型的键
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    # 过滤掉模型中不存在于权重的键（可选）
    # filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if model_state_dict[k].shape == v.shape}

    # 更新模型权重
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)

    print(f"✅ 成功加载权重：{weight_path} | 匹配参数数：{len(filtered_state_dict)}")
    return model


def scan_subjects_models(out_root, network_name='EEGConformer'):
    """
    扫描所有被试的训练模型路径
    """
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
            epoch_num = int(epoch_dir.name.replace('epoch_', ''))
            # 找全局最佳模型
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
    """
    测试单个被试单个epoch的模型
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f"\n----- 测试被试 {sub_id} 的 Epoch:{epoch_num} -----")

    # 加载测试数据
    try:
        data, labels = load_BCI42_data(config['data_path'], datafile)
        if len(data.shape) == 3:
            data = data[:, np.newaxis, :, :]
        elif len(data.shape) == 4 and data.shape[1] != 1:
            data = np.transpose(data, (0, 3, 1, 2))
    except Exception as e:
        print(f"❌ 被试 {sub_id} Epoch:{epoch_num} 数据加载失败：{str(e)}")
        return None

    # 创建测试数据集
    test_dataset = eegDataset(data, labels)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, pin_memory=True, num_workers=0)

    # 初始化模型
    try:
        net = EEGConformer(**config['network_args']).to(device)
        # 安全加载权重
        net = load_model_weights(net, model_path, device)
    except Exception as e:
        print(f"❌ 被试 {sub_id} Epoch:{epoch_num} 模型加载失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

    # 测试模型
    net.eval()
    test_correct, total_test = 0, 0
    test_loss = 0.0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            # 评估模式返回(特征, 输出)
            _, outputs = net(batch_data)
            loss = loss_func(outputs, batch_labels)

            test_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs, 1)
            total_test += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

    # 计算结果
    test_acc = test_correct / total_test
    avg_loss = test_loss / total_test
    print(f"✅ 被试 {sub_id} Epoch:{epoch_num} 测试完成 | 测试ACC：{test_acc:.4f} | 平均损失：{avg_loss:.4f}")

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
    # 配置参数
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data_English'
    OUT_ROOT = r'F:/EEG-TransNet-main/output/bci_handwriting_data_English'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_conformer.yaml'
    NETWORK_NAME = 'EEGConformer'

    # 导入必要模块
    import random  # 补充导入setRandom需要的random

    # 加载配置文件
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)

    # 默认配置
    config.setdefault('batch_size', 16)
    config.setdefault('device', 'cuda:0')
    config.setdefault('random_seed', 0)
    config.setdefault('network_args', {})
    # 模型参数（和训练时保持一致）
    config['network_args'].setdefault('emb_size', 40)
    config['network_args'].setdefault('depth', 4)
    config['network_args'].setdefault('n_classes', 10)
    config['network_args'].setdefault('num_heads', 8)
    config['network_args'].setdefault('drop_p', 0.3)
    config['network_args'].setdefault('forward_expansion', 4)
    config['network_args'].setdefault('input_channels', 32)
    config['data_path'] = DATA_ROOT

    # 设置随机种子
    setRandom(config['random_seed'])

    # 扫描所有被试的模型
    print("开始批量测试所有被试的所有epoch模型（模型版本：{}）...".format(NETWORK_NAME))
    subject_models = scan_subjects_models(OUT_ROOT, NETWORK_NAME)
    if not subject_models:
        raise ValueError(f"在 {OUT_ROOT} 中未找到 {NETWORK_NAME} 的模型文件！")

    # 存储所有测试结果
    all_test_results = []

    # 逐个测试被试
    for sub_id in subject_models.keys():
        print(f"\n=====================================")
        print(f"开始处理被试：{sub_id}")
        print(f"=====================================")

        # 构造测试数据文件名（AxxE）
        sub_num = sub_id.replace('sub', '')
        test_datafile = f"A{sub_num}E"

        # 获取该被试的epoch列表
        epoch_list = subject_models[sub_id]['epochs']
        model_paths = subject_models[sub_id]['model_paths']
        print(f"被试 {sub_id} 共找到 {len(epoch_list)} 个有效epoch：{epoch_list}")

        # 加载测试数据（验证）
        try:
            data, labels = load_BCI42_data(DATA_ROOT, test_datafile)
            print(f"Data shape:  {data.shape}")
            print(f"Label shape:  {labels.shape}")
        except Exception as e:
            print(f"❌ 被试 {sub_id} 数据加载失败：{str(e)}")
            continue

        # 逐个测试epoch
        for epoch_num in epoch_list:
            model_path = model_paths[epoch_num]
            try:
                result = test_subject(config, sub_id, test_datafile, epoch_num, model_path)
                if result:
                    all_test_results.append(result)
            except Exception as e:
                print(f"❌ 被试 {sub_id} Epoch:{epoch_num} 测试失败：{str(e)}")
                import traceback

                traceback.print_exc()
                continue

    # 保存测试结果
    if all_test_results:
        timestamp = time.strftime('%Y-%m-%d--%H-%M', time.localtime())
        result_path = os.path.join(OUT_ROOT, NETWORK_NAME, f"batch_test_results_{timestamp}.yaml")
        # 转换为可序列化的格式
        serializable_results = []
        for res in all_test_results:
            serializable_res = res.copy()
            serializable_res['test_acc'] = float(serializable_res['test_acc'])
            serializable_res['test_loss'] = float(serializable_res['test_loss'])
            serializable_results.append(serializable_res)

        # 写入yaml文件
        with open(result_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'test_time': timestamp,
                'total_subjects': len(subject_models),
                'total_tests': len(all_test_results),
                'results': serializable_results
            }, f, allow_unicode=True)

        # 打印汇总信息
        print(f"\n=====================================")
        print(f"批量测试完成！")
        print(f"=====================================")
        print(f"测试被试数：{len(subject_models)}")
        print(f"有效测试数：{len(all_test_results)}")
        print(f"结果保存路径：{result_path}")

        # 计算平均准确率
        avg_acc = np.mean([res['test_acc'] for res in all_test_results])
        print(f"所有模型平均测试ACC：{avg_acc:.4f}")

        # 找出每个被试的最佳epoch
        sub_best_results = {}
        for res in all_test_results:
            sub_id = res['sub_id']
            if sub_id not in sub_best_results or res['test_acc'] > sub_best_results[sub_id]['test_acc']:
                sub_best_results[sub_id] = res

        print(f"\n各被试最佳测试结果：")
        for sub_id, res in sub_best_results.items():
            print(f"✅ {sub_id} | 最佳Epoch：{res['epoch']} | ACC：{res['test_acc']:.4f}")
    else:
        print(f"\n❌ 批量测试无有效结果！")