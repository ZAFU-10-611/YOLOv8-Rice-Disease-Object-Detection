import argparse
from ultralytics import YOLO
import torch
import torch.nn as nn
import os
import sys
import warnings
import numpy as np
from pathlib import Path
import yaml


def parse_opt():
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--data', type=str, required=True, help='dataset yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0', help='cuda device or cpu')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')

    # ⭐ 注意力机制相关参数（方案A：动态插入CBAM）
    parser.add_argument('--attention', default=None,
                        choices=['cbam', None],
                        help='注意力机制: cbam (动态插入CBAM模块) 或 None')
    parser.add_argument('--use-attention-yaml', action='store_true',
                        help='(已弃用) 使用注意力机制YAML配置文件')

    # 高学习率训练相关参数
    parser.add_argument('--high-lr', action='store_true', help='启用高学习率训练模式')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率因子')
    parser.add_argument('--lr-warmup', type=int, default=3, help='学习率预热轮数')

    # 优化器和正则化
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--momentum', type=float, default=0.937, help='动量因子')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')

    # 数据增强
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic数据增强概率')
    parser.add_argument('--mixup', type=float, default=0.0, help='MixUp数据增强概率')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix数据增强概率')
    parser.add_argument('--fliplr', type=float, default=0.5, help='水平翻转概率')
    parser.add_argument('--degrees', type=float, default=0.0, help='旋转角度范围')
    parser.add_argument('--translate', type=float, default=0.0, help='平移比例')
    parser.add_argument('--scale', type=float, default=0.0, help='缩放比例')

    # 高级训练选项
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率调度')
    parser.add_argument('--close-mosaic', type=int, default=10, help='最后N个epoch关闭mosaic')
    parser.add_argument('--freeze', type=int, default=None,
                        help='冻结层数，如 10 表示冻结前10层')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    parser.add_argument('--resume', action='store_true', help='从上次训练恢复')

    # 损失函数权重
    parser.add_argument('--box', type=float, default=7.5, help='边界框损失权重')
    parser.add_argument('--cls', type=float, default=0.5, help='分类损失权重')
    parser.add_argument('--dfl', type=float, default=1.5, help='DFL损失权重')

    # 保存和日志
    parser.add_argument('--save-period', type=int, default=-1, help='每N个epoch保存一次')
    parser.add_argument('--save-best', action='store_true', help='只保存最佳模型')
    parser.add_argument('--verbose', action='store_true', help='输出详细训练信息')

    return parser.parse_args()


# ==================== 方案A核心：动态插入 CBAM ====================
def inject_cbam_into_yolo11(model):
    """
    动态将 CBAM 模块插入 YOLOv11 backbone 的 P3 (256ch) 和 P4 (512ch) 输出位置。
    只插入两个 CBAM：第一个 256 通道 C3k2 后，第一个 512 通道 C3k2 后。
    """
    from ultralytics.nn.modules import CBAM
    import torch.nn as nn

    # 获取模型序列
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        seq = model.model.model
    else:
        seq = model.model

    # 转换为列表以便插入
    layers = list(seq.children())
    insert_info = []  # 存放 (索引, 通道数, 名称)

    found_256 = False
    found_512 = False

    for idx, m in enumerate(layers):
        if m.__class__.__name__ == 'C3k2':
            out_ch = None
            # 获取输出通道数
            if hasattr(m, 'cv3') and hasattr(m.cv3, 'conv'):
                out_ch = m.cv3.conv.out_channels
            elif hasattr(m, 'cv2') and hasattr(m.cv2, 'conv'):
                out_ch = m.cv2.conv.out_channels

            if out_ch == 256 and not found_256:
                insert_info.append((idx, 256, 'P3'))
                found_256 = True
                print(f"✅ 定位 P3 插入点: 层 {idx} (输出通道 256)")
            elif out_ch == 512 and not found_512:
                insert_info.append((idx, 512, 'P4'))
                found_512 = True
                print(f"✅ 定位 P4 插入点: 层 {idx} (输出通道 512)")

            if found_256 and found_512:
                break

    if not insert_info:
        print("⚠️ 未找到合适的 C3k2 层，CBAM 插入失败。")
        return model

    # 从后向前插入，避免索引错乱
    for idx, ch, name in sorted(insert_info, key=lambda x: x[0], reverse=True):
        cbam = CBAM(ch, kernel_size=7)
        layers.insert(idx + 1, cbam)
        print(f"✅ 在层 {idx} ({name}) 后插入 CBAM (kernel_size=7)")

    # 重新构建 Sequential
    new_seq = nn.Sequential(*layers)

    # 替换回原模型
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model = new_seq
    else:
        model.model = new_seq

    # 尝试重置内部索引（非必须，但建议）
    try:
        # 对于 YOLO 封装，有时需要清理缓存
        if hasattr(model.model, '_modules'):
            # 避免 None 比较错误
            if model.model._modules is not None:
                model.model._modules.clear()
        if hasattr(model.model, '_reset_sequential'):
            model.model._reset_sequential()
    except Exception as e:
        # 忽略重置时的非关键错误
        print(f"⚠️ 模型状态重置时出现非致命警告: {e}")

    print("✅ CBAM 动态插入完成，模型结构已更新。")
    return model

def auto_select_attention_model(opt):
    """
    适配方案A：如果用户指定 --attention cbam，则后续动态插入 CBAM。
    """
    if opt.attention is None:
        return opt

    print("\n" + "="*60)
    print("🔄 注意力机制配置 (动态插入 CBAM)")
    print("="*60)

    # 检查模型是否为 .pt 预训练权重（推荐）
    if not opt.model.endswith('.pt'):
        print("⚠️ 警告：建议使用官方 .pt 权重（如 yolo11l.pt）以正确加载预训练参数。")
    print(f"✅ 将在加载模型后动态插入 CBAM 模块。")
    opt.use_attention_yaml = False  # 不再使用 YAML 模式
    return opt


def setup_attention_mechanism(opt):
    """保留原函数，但不再修改学习率，仅打印信息"""
    if opt.attention is None:
        return opt

    print("\n" + "="*60)
    print("🧠 注意力机制参数微调 (基于数据集大小)")
    print("="*60)

    dataset_size = get_dataset_size(opt.data)
    print(f"📊 数据集大小: {dataset_size} 张图像" if dataset_size else "⚠️ 无法检测数据集大小")

    # 对 CBAM 模型适当降低学习率（可选）
    if dataset_size and dataset_size < 1000:
        opt.lr0 = min(opt.lr0, 0.001)
        opt.weight_decay = max(opt.weight_decay, 0.001)
        print("📉 小数据集：学习率已自动保守化")
    print("✅ 注意力机制配置完成")
    return opt


def get_dataset_size(data_yaml_path):
    """检测数据集训练图像数量"""
    try:
        if not os.path.exists(data_yaml_path):
            return None
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        if 'train' not in data_config:
            return None
        train_path = data_config['train']
        if not os.path.isabs(train_path):
            yaml_dir = os.path.dirname(data_yaml_path)
            train_path = os.path.join(yaml_dir, train_path)
        if os.path.isdir(train_path):
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            return sum(1 for f in os.listdir(train_path) if os.path.splitext(f)[1].lower() in exts)
    except Exception as e:
        print(f"⚠️ 数据集大小检测失败: {e}")
    return None


def setup_high_lr_config(opt):
    if opt.high_lr:
        print("\n" + "="*60)
        print("🔧 启用高学习率训练模式 (优化1: 学习率智能配置)")
        print("="*60)

        dataset_size = get_dataset_size(opt.data)
        print(f"📊 检测到数据集大小: {dataset_size} 张图像" if dataset_size else "⚠️ 无法检测数据集大小")

        if dataset_size and dataset_size < 1000:
            print("📉 检测到小数据集模式（<1000张）")
            if 'yolov11' in opt.model:
                opt.lr0 = 0.0012
                opt.lrf = 1e-6
                opt.weight_decay = 0.0018
                opt.freeze = 10
            else:
                opt.lr0 = 0.002
                opt.lrf = 0.0001
                opt.weight_decay = 0.001
            opt.lr_warmup = 10
            opt.patience = 50
            opt.close_mosaic = 20
        else:
            if 'yolov11' in opt.model:
                opt.lr0 = 0.001
                opt.lrf = 1e-6
                opt.weight_decay = 0.0018
                opt.lr_warmup = 15
                opt.patience = 45
                print("   ⭐ YOLOv11L 架构检测，应用专用优化配置")
            else:
                opt.lr0 = 0.005
                opt.lrf = 0.0001
                opt.weight_decay = 0.0005
                opt.lr_warmup = 5
        opt.cos_lr = True

        print(f"✅ 学习率配置完成:")
        print(f"   - 初始学习率(lr0): {opt.lr0}")
        print(f"   - 最终学习率因子(lrf): {opt.lrf}")
        print(f"   - 权重衰减: {opt.weight_decay}")
        print(f"   - 预热轮数: {opt.lr_warmup}")
        print(f"   - 冻结层数: {opt.freeze if opt.freeze else '无'}")
        print(f"   - 早停耐心: {opt.patience}")
    return opt


def setup_augmentation(opt):
    if opt.augment:
        print("\n" + "="*60)
        print("🎨 启用激进数据增强 (优化2: 完整数据增强套件)")
        print("="*60)
        dataset_size = get_dataset_size(opt.data)
        opt.mosaic = 1.0
        if dataset_size and dataset_size < 1000:
            opt.mixup = 0.4
            opt.cutmix = 0.3
            opt.degrees = 25
            opt.translate = 0.3
            opt.scale = 0.5
            print("   配置: 小数据集 - 强增强")
        else:
            opt.mixup = 0.15
            opt.cutmix = 0.15
            opt.degrees = 15
            opt.translate = 0.2
            opt.scale = 0.3
            print("   配置: 中等/大数据集 - 平衡增强")
        opt.fliplr = 0.5
        print("✅ 数据增强配置完成")
    return opt


def optimize_for_small_dataset(opt):
    print("\n" + "="*60)
    print("🧠 启用小数据集智能优化 (优化3: 自适应参数调整)")
    print("="*60)
    dataset_size = get_dataset_size(opt.data)
    if dataset_size is None:
        print("⚠️ 无法检测数据集大小，跳过优化")
        return opt
    print(f"📊 数据集大小: {dataset_size} 张图像")
    if dataset_size < 1000:
        print("📍 触发小数据集优化（<1000张）")
        if opt.freeze is None:
            opt.freeze = min(int(dataset_size / 100), 20)
        opt.weight_decay = max(opt.weight_decay, 0.001)
        if opt.patience > 50:
            opt.patience = 50
        opt.close_mosaic = max(20, opt.epochs // 10)
        if opt.lr_warmup < 10:
            opt.lr_warmup = 10
    elif dataset_size < 5000:
        print("📍 触发中等数据集优化（1000-5000张）")
        if opt.freeze is None:
            opt.freeze = 8
        opt.weight_decay = max(opt.weight_decay, 0.001)
        if opt.patience > 50:
            opt.patience = 50
        opt.close_mosaic = max(40, int(opt.epochs * 0.15))
        if opt.lr_warmup < 10:
            opt.lr_warmup = 10
    else:
        print("📍 检测到大数据集（>5000张），使用标准配置")
        if opt.freeze is None:
            opt.freeze = 0
    print("✅ 小数据集优化应用完成")
    return opt


def analyze_four_class_rice_disease(save_dir):
    """保留原分析函数（略作调整）"""
    try:
        import pandas as pd
        results_csv = os.path.join(save_dir, 'results.csv')
        if not os.path.exists(results_csv):
            return
        df = pd.read_csv(results_csv)
        class_map_cols = [col for col in df.columns if 'mAP50' in col and col != 'metrics/mAP50']
        if not class_map_cols:
            return
        last_n = min(10, len(df))
        last_epochs = df.tail(last_n)
        print("\n📊 逐类性能分析（最后{}轮平均）:".format(last_n))
        for col in class_map_cols:
            if col in last_epochs.columns:
                avg = last_epochs[col].mean()
                name = col.split('(')[-1].replace(')', '').strip()
                print(f"   {name}: {avg:.4f}")
    except Exception as e:
        print(f"⚠️ 逐类分析失败: {e}")


def get_valid_training_args(opt):
    valid_args = [
        'data', 'epochs', 'batch', 'imgsz', 'device', 'workers',
        'project', 'name', 'exist_ok', 'pretrained', 'resume',
        'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
        'cos_lr', 'warmup_epochs', 'warmup_momentum',
        'mosaic', 'mixup', 'cutmix', 'fliplr', 'flipud',
        'degrees', 'translate', 'scale', 'shear', 'perspective',
        'hsv_h', 'hsv_s', 'hsv_v',
        'patience', 'close_mosaic', 'freeze',
        'box', 'cls', 'dfl', 'amp', 'overlap_mask',
        'save', 'save_period', 'plots', 'verbose',
        'conf', 'iou', 'max_det', 'half', 'dnn', 'cache',
        'seed', 'deterministic', 'single_cls', 'rect',
    ]
    args_dict = vars(opt)
    valid_args_dict = {}
    for arg in valid_args:
        if arg in args_dict and args_dict[arg] is not None:
            if arg == 'lr_warmup':
                valid_args_dict['warmup_epochs'] = args_dict[arg]
            else:
                valid_args_dict[arg] = args_dict[arg]
    return valid_args_dict


def main(opt):
    print("\n" + "="*80)
    print("🚀 YOLOv8 优化训练脚本 - 动态CBAM版".center(80))
    print("="*80)

    # 注意力机制选择（仅打印）
    opt = auto_select_attention_model(opt)

    # 参数优化
    opt = setup_high_lr_config(opt)
    opt = setup_augmentation(opt)
    opt = optimize_for_small_dataset(opt)
    opt = setup_attention_mechanism(opt)

    print(f"\n📋 最终训练配置:")
    print(f"  模型: {opt.model}")
    print(f"  数据: {opt.data}")
    print(f"  Epochs: {opt.epochs}")
    print(f"  批大小: {opt.batch}")
    print(f"  学习率: {opt.lr0}")

    print(f"\n📦 加载模型: {opt.model}")
    model = YOLO(opt.model)

    # ⭐ 动态插入 CBAM（如果指定了 --attention cbam）
    if opt.attention == 'cbam':
        print("\n" + "="*60)
        print("🧠 执行动态 CBAM 插入 (方案A)")
        print("="*60)
        model = inject_cbam_into_yolo11(model)

    train_args = get_valid_training_args(opt)

    print(f"\n🎯 开始训练...")
    try:
        results = model.train(**train_args)
        print(f"\n✅ 训练完成!")
        # 分析
        save_dir = results.save_dir if hasattr(results, 'save_dir') else opt.project
        analyze_four_class_rice_disease(save_dir)
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)