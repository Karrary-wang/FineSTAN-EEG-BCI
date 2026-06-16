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
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.MCTD import MCTD  # 导入MCTD模型（需确保MCTD.py在model目录下）

# ===================== 工具函数（适配MCTD） =====================
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


def load_model_weights(model, weight_path, device):
    """
    安全加载MCTD模型权重，适配动态分类头和参数不匹配问题
    """
    try:
        # 加载权重文件（兼容CPU/GPU）
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except:
        state_dict = torch.load(weight_path, map_location=device)

    # 关键：预热模型，初始化MCTD的动态分类头（必须执行）
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 32, 1000).to(device)  # 匹配MCTD输入维度
        feats, outputs = model(dummy_input)  # 触发分类头动态初始化

    # 过滤权重：只加载模型中存在的参数（避免动态层参数不匹配）
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v

    # 加载权重（strict=False 兼容动态层）
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f" 成功加载MCTD权重：{weight_path} | 匹配参数数：{len(filtered_state_dict)}/{len(state_dict)}")
    return model


def scan_subjects_models(out_root, network_name='MCTD'):
    """
    扫描所有被试的MCTD训练模型路径（优化：兼容不同命名格式）
    """
    out_root = Path(out_root)
    subject_models = {}

    # 遍历网络目录下的所有被试文件夹
    for sub_dir in out_root.glob(f"{network_name}/sub*"):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name
        epoch_dirs = [d for d in sub_dir.glob("epoch_*") if d.is_dir()]
        if not epoch_dirs:
            print(f" 被试 {sub_id} 未找到epoch目录，跳过")
            continue

        model_paths = {}
        # 遍历每个epoch的模型
        for epoch_dir in epoch_dirs:
            try:
                epoch_num = int(epoch_dir.name.replace('epoch_', ''))
            except:
                print(f" 无效的epoch目录名：{epoch_dir.name}，跳过")
                continue

            # 优先找全局最佳模型，其次找final模型
            global_best_paths = list(epoch_dir.glob("cv_*/cv_global_best.pth"))
            final_model_paths = list(epoch_dir.glob("final_*/final_model.pth"))

            if global_best_paths:
                model_paths[epoch_num] = global_best_paths[0]
            elif final_model_paths:
                model_paths[epoch_num] = final_model_paths[0]
            else:
                print(f" 被试 {sub_id} Epoch:{epoch_num} 未找到有效模型文件")
                continue

        if model_paths:
            epoch_list = sorted(model_paths.keys())
            subject_models[sub_id] = {
                'epochs': epoch_list,
                'model_paths': model_paths
            }

    return subject_models


# ===================== 核心测试函数（适配MCTD多尺度输出） =====================
def test_subject(config, sub_id, datafile, epoch_num, model_path):
    """
    测试单个被试单个epoch的MCTD模型（核心：多尺度输出平均）
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    print(f"\n----- 测试被试 {sub_id} | Epoch:{epoch_num} -----")

    # 1. 加载并预处理测试数据
    try:
        data, labels = load_BCI42_data(config['data_path'], datafile)
        # 适配MCTD输入维度：[样本数, 1, 32通道, 1000采样点]
        if len(data.shape) == 3:
            data = data[:, np.newaxis, :, :]  # [N, 32, 1000] → [N, 1, 32, 1000]
        elif len(data.shape) == 4 and data.shape[1] != 1:
            data = np.transpose(data, (0, 3, 1, 2))  # 调整通道维度
        print(f" 数据加载完成 | 数据形状：{data.shape} | 标签数：{len(labels)}")
    except Exception as e:
        print(f" 被试 {sub_id} 数据加载失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

    # 2. 创建测试数据加载器
    test_dataset = eegDataset(data, labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0  # 避免多线程随机种子问题
    )

    # 3. 初始化并加载MCTD模型
    try:
        net = MCTD(**config['network_args']).to(device)
        net = load_model_weights(net, model_path, device)
    except Exception as e:
        print(f" 被试 {sub_id} 模型加载失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

    # 4. 模型测试（核心：MCTD多尺度输出平均）
    net.eval()
    test_correct, total_test = 0, 0
    test_loss = 0.0
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.01)  # 和训练保持一致

    with torch.no_grad():  # 禁用梯度计算，加速测试
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            batch_data = batch_data.to(device, dtype=torch.float32)
            batch_labels = batch_labels.to(device, dtype=torch.long)

            # MCTD返回：(多尺度特征列表, 多尺度输出列表)
            _, outputs = net(batch_data)
            # 关键：5个尺度输出取平均作为最终预测
            outputs_avg = torch.stack(outputs).mean(dim=0)

            # 计算损失和准确率
            loss = loss_func(outputs_avg, batch_labels)
            test_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs_avg, 1)
            total_test += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

            # 打印批次进度
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (predicted == batch_labels).sum().item() / batch_labels.size(0)
                print(f" 批次 {batch_idx + 1}/{len(test_loader)} | 批次ACC：{batch_acc:.4f}")

    # 5. 计算最终结果
    test_acc = test_correct / total_test
    avg_loss = test_loss / total_test
    print(f"----- 被试 {sub_id} Epoch:{epoch_num} 测试结果 -----")
    print(f" 总样本数：{total_test} | 正确数：{test_correct}")
    print(f" 测试ACC：{test_acc:.4f} | 平均损失：{avg_loss:.4f}")

    return {
        'sub_id': sub_id,
        'epoch': epoch_num,
        'test_acc': float(test_acc),
        'test_loss': float(avg_loss),
        'total_samples': total_test,
        'correct_samples': test_correct,
        'model_path': str(model_path),
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }


# ===================== 主函数（适配MCTD参数） =====================
if __name__ == '__main__':
    # 基础配置（需根据你的路径修改）
    DATA_ROOT = r'F:/EEG-TransNet-main/dataset/bci_handwriting_data'
    OUT_ROOT = r'F:/EEG-TransNet-main/output/bci_handwriting_data'
    CONFIG_PATH = r'F:/EEG-TransNet-main/config/Hand_Writing_MCTD.yaml'
    NETWORK_NAME = 'MCTD'  # 固定为MCTD

    # 加载配置文件
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f) or {}  # 兼容空配置

    # 默认配置（和训练保持一致）
    config.setdefault('batch_size', 16)
    config.setdefault('device', 'cuda:0')
    config.setdefault('random_seed', 0)
    config.setdefault('network_args', {})

    # MCTD核心参数（必须和训练代码一致）
    config['network_args'].setdefault('in_chans', 32)  # 32电极通道
    config['network_args'].setdefault('emb_size', 40)  # 特征嵌入维度
    config['network_args'].setdefault('n_classes', 10)  # 手写想象10分类
    config['network_args'].setdefault('depth', 6)  # Transformer编码器层数
    config['data_path'] = DATA_ROOT

    # 设置随机种子（保证可复现）
    setRandom(config['random_seed'])

    # 扫描所有被试的MCTD模型
    print("=" * 60)
    print(f"开始批量测试 {NETWORK_NAME} 模型...")
    print(f"模型输出根目录：{OUT_ROOT}")
    print("=" * 60)
    subject_models = scan_subjects_models(OUT_ROOT, NETWORK_NAME)

    if not subject_models:
        raise ValueError(f"在 {OUT_ROOT} 中未找到 {NETWORK_NAME} 的模型文件！")
    print(f"共找到 {len(subject_models)} 个被试的有效模型：{list(subject_models.keys())}")

    # 存储所有测试结果
    all_test_results = []

    # 逐个测试被试
    for sub_id in sorted(subject_models.keys()):
        print(f"\n=====================================")
        print(f"开始处理被试：{sub_id}")
        print(f"=====================================")

        # 构造测试数据文件名（AxxE，对应评估集）
        sub_num = sub_id.replace('sub', '')
        test_datafile = f"A{sub_num}E"

        # 获取该被试的epoch列表和模型路径
        epoch_list = subject_models[sub_id]['epochs']
        model_paths = subject_models[sub_id]['model_paths']
        print(f"被试 {sub_id} 可测试Epoch：{epoch_list}")

        # 验证数据是否存在
        try:
            load_BCI42_data(DATA_ROOT, test_datafile)
        except Exception as e:
            print(f" 被试 {sub_id} 测试数据不存在：{test_datafile} | 错误：{str(e)}")
            continue

        # 逐个测试每个epoch的模型
        for epoch_num in epoch_list:
            model_path = model_paths[epoch_num]
            if not os.path.exists(model_path):
                print(f" 被试 {sub_id} Epoch:{epoch_num} 模型文件不存在：{model_path}")
                continue

            # 执行测试
            result = test_subject(config, sub_id, test_datafile, epoch_num, model_path)
            if result:
                all_test_results.append(result)

    # 保存并汇总测试结果
    if all_test_results:
        # 生成结果保存路径
        timestamp = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        result_dir = os.path.join(OUT_ROOT, NETWORK_NAME)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"batch_test_results_{timestamp}.yaml")

        # 保存结果到YAML（确保可序列化）
        with open(result_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'test_config': config,
                'test_time': timestamp,
                'total_subjects': len(subject_models),
                'total_tests': len(all_test_results),
                'average_test_acc': float(np.mean([res['test_acc'] for res in all_test_results])),
                'results': all_test_results
            }, f, allow_unicode=True, sort_keys=False)

        # 打印汇总信息
        print("\n" + "=" * 60)
        print(f"批量测试完成！")
        print("=" * 60)
        print(f"测试被试总数：{len(subject_models)}")
        print(f"有效测试数：{len(all_test_results)}")
        print(f"所有模型平均ACC：{np.mean([res['test_acc'] for res in all_test_results]):.4f}")
        print(f"结果保存路径：{result_path}")

        # 打印各被试最佳结果
        print("\n各被试最佳测试结果：")
        print("-" * 60)
        sub_best = {}
        for res in all_test_results:
            sub = res['sub_id']
            if sub not in sub_best or res['test_acc'] > sub_best[sub]['test_acc']:
                sub_best[sub] = res

        for sub, res in sorted(sub_best.items()):
            print(f"{sub:<6} | 最佳Epoch：{res['epoch']:<4} | ACC：{res['test_acc']:.4f} | 损失：{res['test_loss']:.4f}")
    else:
        print("\n⚠️ 批量测试无有效结果！请检查模型文件和数据路径。")