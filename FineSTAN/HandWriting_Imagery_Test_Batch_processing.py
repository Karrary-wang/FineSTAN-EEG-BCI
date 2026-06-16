import numpy as np
import torch
import os
import yaml
import time
import glob
import traceback
from pathlib import Path
from torch.utils.data import DataLoader
from data.data_utils import load_BCI42_data
from data.dataset import eegDataset
from model.finestan import FineSTAN


def load_model_weights(model, weights_path, device):
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{weights_path}")

    try:
        load_kwargs = {'map_location': device}
        if torch.__version__ >= '2.0.0':
            load_kwargs['weights_only'] = True
        state_dict = torch.load(str(weights_path), **load_kwargs)
    except Exception as e:
        raise RuntimeError(f"加载权重失败 {weights_path}：{e}")

    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()
                      if k.startswith('module.') or k in model.state_dict()}
    model.load_state_dict(new_state_dict, strict=True)
    return model


def test_single_model(model, test_loader, device):
    test_correct, total_test = 0, 0
    infer_times = []

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device, dtype=torch.float32, non_blocking=device.type == 'cuda')
            batch_labels = batch_labels.to(device, dtype=torch.long, non_blocking=device.type == 'cuda')

            start = time.perf_counter()
            outputs = model(batch_data)
            infer_times.append(time.perf_counter() - start)

            _, predicted = torch.max(outputs, 1)
            total_test += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

    acc = test_correct / total_test if total_test > 0 else 0.0
    avg_time = np.mean(infer_times) if infer_times else 0.0
    return acc, avg_time


def init_model_args(train_config, test_data, test_labels):
    network_args = train_config['network_args'].copy()
    selected_channel_idx = train_config.get('selected_channel_idx', [])

    if not selected_channel_idx:
        num_channels = test_data.shape[1]
        selected_channel_idx = list(range(num_channels))

    network_args.update({
        'num_classes': len(np.unique(test_labels)),
        'num_samples': test_data.shape[2],
        'num_channels': test_data.shape[1],
        'selected_channel_idx': selected_channel_idx
    })
    return network_args, selected_channel_idx


def scan_subject_all_epochs(subject_dir):
    subject_dir = Path(subject_dir).resolve()
    epoch_info = {}
    print(f"\n=== 扫描被试文件夹：{subject_dir} ===")

    epoch_folders = sorted(
        glob.glob(os.path.join(subject_dir, "epoch_*")),
        key=lambda x: int(os.path.basename(x).replace("epoch_", ""))
        if os.path.basename(x).replace("epoch_", "").isdigit() else 0
    )
    print(f"匹配到的epoch文件夹：{epoch_folders}")

    for epoch_folder in epoch_folders:
        epoch_folder = Path(epoch_folder)
        try:
            epoch_num = int(epoch_folder.name.replace("epoch_", ""))
        except (ValueError, AttributeError):
            print(f"跳过{epoch_folder}：无法提取epoch数字")
            continue

        cv_folders = sorted(glob.glob(os.path.join(epoch_folder, "cv_*")))
        final_folders = sorted(glob.glob(os.path.join(epoch_folder, "final_*")))

        if not cv_folders or not final_folders:
            print(f"跳过epoch {epoch_num}：缺少cv_*或final_*文件夹")
            continue

        cv_folder = Path(cv_folders[-1])
        final_folder = Path(final_folders[-1])
        cv_summary_path = cv_folder / "cv_summary.yaml"

        if not cv_summary_path.exists():
            print(f"跳过epoch {epoch_num}：缺失CV训练汇总文件 {cv_summary_path}")
            continue

        with open(cv_summary_path, 'r', encoding='utf-8') as f:
            cv_summary = yaml.safe_load(f)
        cv_train_avg_best_epoch = cv_summary.get('avg_best_epoch', 0)
        cv_train_mean_acc = cv_summary.get('cv_mean_acc', 0.0)

        paths = {
            'cv_model': cv_folder / "cv_global_best.pth",
            'final_model': final_folder / "final_model.pth",
            'cv_config': cv_folder / "cv_config.yaml",
            'final_config': final_folder / "final_config.yaml",
            'final_folder': str(final_folder),
            'cv_train_avg_best_epoch': cv_train_avg_best_epoch,
            'cv_train_mean_acc': cv_train_mean_acc
        }

        missing_files = [k for k, v in paths.items()
                         if k not in ['cv_train_avg_best_epoch', 'cv_train_mean_acc'] and not Path(v).exists()]
        if missing_files:
            print(f"跳过epoch {epoch_num}：缺失文件 {missing_files}")
            continue

        epoch_info[epoch_num] = paths
        print(f"成功识别epoch {epoch_num}：CV训练平均最佳epoch={cv_train_avg_best_epoch}，CV训练平均ACC={cv_train_mean_acc:.4f}")

    return epoch_info


def find_best_epoch_in_subject(subject_epoch_results):
    if not subject_epoch_results:
        return None

    best_final_item = max(subject_epoch_results.items(), key=lambda x: (x[1]['final_acc'], -x[0]))
    cv_train_best_epoch_test_item = None
    max_cv_train_mean_acc = 0.0

    for epoch_num, result in subject_epoch_results.items():
        if result['cv_train_mean_acc'] > max_cv_train_mean_acc:
            max_cv_train_mean_acc = result['cv_train_mean_acc']
            cv_train_best_epoch_test_item = (epoch_num, result)

    return {
        'best_final_epoch_num': best_final_item[0],
        'best_final_acc': best_final_item[1]['final_acc'],
        'cv_train_best_epoch_num': cv_train_best_epoch_test_item[0] if cv_train_best_epoch_test_item else 0,
        'cv_train_best_epoch_cv_test_acc': cv_train_best_epoch_test_item[1]['cv_acc'] if cv_train_best_epoch_test_item else 0.0,
        'cv_train_best_epoch_final_test_acc': cv_train_best_epoch_test_item[1]['final_acc'] if cv_train_best_epoch_test_item else 0.0,
        'cv_train_best_mean_acc': max_cv_train_mean_acc,
        'best_better_acc': best_final_item[1]['better_acc']
    }


def batch_test_all_subjects(config, data_root, output_root, network_name):
    data_root = Path(data_root).resolve()
    network_dir = Path(output_root).resolve() / network_name
    print(f"\n=== 开始批量测试 | 网络路径：{network_dir} ===")

    subject_dirs = sorted(
        glob.glob(os.path.join(network_dir, "sub*")),
        key=lambda x: int(os.path.basename(x).replace("sub", ""))
        if os.path.basename(x).replace("sub", "").isdigit() else 0
    )

    if not subject_dirs:
        raise ValueError(f"未找到被试文件夹：{network_dir}")

    all_subjects_summary = {}
    # 新增：存储每个被试ID+最佳Final准确率
    subject_acc_list = []
    device = torch.device(
        'cuda:0' if (torch.cuda.is_available() and config.get('preferred_device') == 'gpu') else 'cpu')
    print(f"测试设备：{device}")

    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir)
        print(f"\n=====================================")
        print(f"开始处理被试：{sub_id}")
        print(f"=====================================")

        epoch_info = scan_subject_all_epochs(subject_dir)
        if not epoch_info:
            print(f"被试 {sub_id} 无有效epoch数据，跳过")
            continue

        try:
            sub_num = sub_id.replace('sub', '').zfill(2)
            test_datafile = f"A{sub_num}E"
        except:
            print(f"被试 {sub_id} 编号格式错误，跳过")
            continue

        try:
            test_data, test_labels = load_BCI42_data(str(data_root), test_datafile)
            print(f"{test_datafile} 加载完成 | 数据形状：{test_data.shape}，标签类别数：{len(np.unique(test_labels))}")
        except Exception as e:
            print(f"被试 {sub_id} 数据加载失败：{str(e)}")
            traceback.print_exc()
            continue

        subject_epoch_results = {}
        for epoch_num, epoch_data in epoch_info.items():
            print(f"\n===== 测试被试 {sub_id} Epoch:{epoch_num} =====")
            try:
                with open(epoch_data['cv_config'], 'r', encoding='utf-8') as f:
                    cv_config = yaml.safe_load(f)
                with open(epoch_data['final_config'], 'r', encoding='utf-8') as f:
                    final_config = yaml.safe_load(f)

                cv_args, selected_channel_idx = init_model_args(cv_config, test_data, test_labels)
                final_args, _ = init_model_args(final_config, test_data, test_labels)

                def worker_init(worker_id):
                    np.random.seed(config.get('random_seed', 0) + worker_id)

                test_loader = DataLoader(
                    eegDataset(test_data, test_labels),
                    batch_size=config['batch_size'],
                    shuffle=False,
                    pin_memory=device.type == 'cuda',
                    num_workers=0,
                    worker_init_fn=worker_init,
                    drop_last=False
                )

                cv_model = FineSTAN(**cv_args).to(device)
                cv_model = load_model_weights(cv_model, epoch_data['cv_model'], device)
                cv_acc, cv_time = test_single_model(cv_model, test_loader, device)

                final_model = FineSTAN(**final_args).to(device)
                final_model = load_model_weights(final_model, epoch_data['final_model'], device)
                final_acc, final_time = test_single_model(final_model, test_loader, device)

                better_model = "CV模型" if cv_acc > final_acc else "Final模型"
                better_acc = max(cv_acc, final_acc)
                result = {
                    'sub_id': sub_id,
                    'epoch_num': epoch_num,
                    'test_datafile': test_datafile,
                    'cv_acc': float(cv_acc),
                    'cv_infer_time': float(cv_time),
                    'final_acc': float(final_acc),
                    'final_infer_time': float(final_time),
                    'better_model': better_model,
                    'better_acc': float(better_acc),
                    'device': str(device),
                    'batch_size': config['batch_size'],
                    'test_data_shape': test_data.shape,
                    'selected_channel_idx': selected_channel_idx,
                    'cv_train_avg_best_epoch': epoch_data['cv_train_avg_best_epoch'],
                    'cv_train_mean_acc': epoch_data['cv_train_mean_acc']
                }

                result_path = Path(epoch_data['final_folder']) / f"{sub_id}_{test_datafile}_epoch{epoch_num}_test_result.yaml"
                with open(result_path, 'w', encoding='utf-8') as f:
                    yaml.dump(result, f, allow_unicode=True, sort_keys=False)

                subject_epoch_results[epoch_num] = result
                print(f"【{sub_id} Epoch:{epoch_num}】CV_ACC：{cv_acc:.4f} | Final_ACC：{final_acc:.4f} | 最优：{better_acc:.4f}")

                del cv_model, final_model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"被试 {sub_id} Epoch:{epoch_num} 测试失败：{str(e)}")
                traceback.print_exc()
                continue

        if not subject_epoch_results:
            continue
        best_epoch_info = find_best_epoch_in_subject(subject_epoch_results)
        if not best_epoch_info:
            continue

        subject_summary = {
            'sub_id': sub_id,
            'all_epochs': subject_epoch_results,
            'best_epoch_info': best_epoch_info,
            'all_final_acc': {k: v['final_acc'] for k, v in subject_epoch_results.items()},
            'all_cv_acc': {k: v['cv_acc'] for k, v in subject_epoch_results.items()},
            'all_better_acc': {k: v['better_acc'] for k, v in subject_epoch_results.items()},
            'all_cv_train_mean_acc': {k: v['cv_train_mean_acc'] for k, v in subject_epoch_results.items()}
        }
        with open(Path(subject_dir) / f"{sub_id}_all_epochs_test_summary.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(subject_summary, f, allow_unicode=True, sort_keys=False)

        all_subjects_summary[sub_id] = best_epoch_info
        # 收集被试与最佳准确率
        subject_acc_list.append((sub_id, best_epoch_info['best_final_acc']))

        print(
            f"\n被试 {sub_id} "
            f"CV训练最佳epoch：{best_epoch_info['cv_train_best_epoch_num']}（训练平均ACC={best_epoch_info['cv_train_best_mean_acc']:.4f}） | "
            f"该epoch CV模型测试ACC：{best_epoch_info['cv_train_best_epoch_cv_test_acc']:.4f} | "
            f"该epoch Final模型测试ACC：{best_epoch_info['cv_train_best_epoch_final_test_acc']:.4f} | "
            f"最佳Final_Epoch：{best_epoch_info['best_final_epoch_num']} | 最佳Final_ACC：{best_epoch_info['best_final_acc']:.4f}"
        )

    # 统一打印所有被试准确率汇总（每行一个被试）
    print("\n" + "="*60)
    print("【所有被试 最佳Final模型测试准确率汇总】")
    print("="*60)
    for sub_name, acc in subject_acc_list:
        print(f"{sub_name} : {acc:.4f}")
    print("="*60)

    global_summary = {
        'tested_subjects': len(all_subjects_summary),
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'batch_size': config['batch_size']
    }

    if all_subjects_summary:
        best_final_acc_list = [v['best_final_acc'] for v in all_subjects_summary.values()]
        cv_train_best_mean_acc_list = [v['cv_train_best_mean_acc'] for v in all_subjects_summary.values()]
        cv_train_best_cv_test_acc_list = [v['cv_train_best_epoch_cv_test_acc'] for v in all_subjects_summary.values()]
        best_better_acc_list = [v['best_better_acc'] for v in all_subjects_summary.values()]
        global_summary.update({
            'avg_best_final_acc': float(np.mean(best_final_acc_list)),
            'std_best_final_acc': float(np.std(best_final_acc_list)),
            'max_best_final_acc': float(np.max(best_final_acc_list)),
            'min_best_final_acc': float(np.min(best_final_acc_list)),
            'avg_cv_train_best_mean_acc': float(np.mean(cv_train_best_mean_acc_list)),
            'std_cv_train_best_mean_acc': float(np.std(cv_train_best_mean_acc_list)),
            'avg_cv_train_best_epoch_cv_test_acc': float(np.mean(cv_train_best_cv_test_acc_list)),
            'std_cv_train_best_epoch_cv_test_acc': float(np.std(cv_train_best_cv_test_acc_list)),
            'avg_best_better_acc': float(np.mean(best_better_acc_list)),
            'std_best_better_acc': float(np.std(best_better_acc_list))
        })

    with open(network_dir / "global_all_subjects_test_summary.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(global_summary, f, allow_unicode=True, sort_keys=False)

    print(f"\n=====================================")
    print(f"测试完成！总测试被试数：{len(all_subjects_summary)}")
    if all_subjects_summary:
        print(f"平均CV训练最佳ACC：{global_summary['avg_cv_train_best_mean_acc']:.4f}±{global_summary['std_cv_train_best_mean_acc']:.4f}")
        print(f"平均CV训练最佳epoch的CV测试ACC：{global_summary['avg_cv_train_best_epoch_cv_test_acc']:.4f}±{global_summary['std_cv_train_best_epoch_cv_test_acc']:.4f}")
        print(f"平均最佳Final_ACC：{global_summary['avg_best_final_acc']:.4f}±{global_summary['std_best_final_acc']:.4f}")
    print("=====================================")


if __name__ == '__main__':
    DATA_ROOT = r'D:/EEG_HandWriting_Project_Code/FineSTAN/dataset/bci_handwriting_data_Pinyin'
    CONFIG_FILE = r'D:/EEG_HandWriting_Project_Code/FineSTAN/config/handwriteformer.yaml'
    OUTPUT_ROOT = r'D:/EEG_HandWriting_Project_Code/FineSTAN/output/bci_handwriting_data_Pinyin/hyperparam_Window'
    NETWORK_NAME = 'FineSTAN'

    try:
        config_file = Path(CONFIG_FILE).resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"加载配置失败：{e}")

    config = {
        **{'batch_size': 64, 'preferred_device': 'gpu', 'network_args': {}, 'random_seed': 0},
        **config
    }

    required_args = ['embed_dim', 'pool_size', 'pool_stride', 'num_heads', 'fc_ratio', 'depth']
    for arg in required_args:
        if arg not in config['network_args']:
            default_vals = {'embed_dim': 32, 'pool_size': 180, 'pool_stride': 30, 'num_heads': 8, 'fc_ratio': 4, 'depth': 4}
            config['network_args'][arg] = default_vals[arg]
            print(f"警告：network_args缺少参数 {arg}，使用默认值 {default_vals[arg]}")

    batch_test_all_subjects(config, DATA_ROOT, OUTPUT_ROOT, NETWORK_NAME)