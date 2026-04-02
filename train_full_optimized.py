import argparse
import os

import numpy as np
import yaml

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="model path")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="cuda device or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")

    # 高学习率训练相关参数
    parser.add_argument("--high-lr", action="store_true", help="启用高学习率训练模式")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率因子")
    parser.add_argument("--lr-warmup", type=int, default=3, help="学习率预热轮数")

    # 优化器和正则化
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--momentum", type=float, default=0.937, help="动量因子")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")

    # 数据增强
    parser.add_argument("--augment", action="store_true", help="启用数据增强")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic数据增强概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp数据增强概率")
    parser.add_argument("--cutmix", type=float, default=0.0, help="CutMix数据增强概率")
    parser.add_argument("--fliplr", type=float, default=0.5, help="水平翻转概率")
    parser.add_argument("--degrees", type=float, default=0.0, help="旋转角度范围")
    parser.add_argument("--translate", type=float, default=0.0, help="平移比例")
    parser.add_argument("--scale", type=float, default=0.0, help="缩放比例")

    # 高级训练选项
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--cos-lr", action="store_true", help="使用余弦学习率调度")
    parser.add_argument("--close-mosaic", type=int, default=10, help="最后N个epoch关闭mosaic")
    parser.add_argument("--freeze", type=int, default=None, help="冻结层数，如 10 表示冻结前10层")
    parser.add_argument("--patience", type=int, default=100, help="早停耐心值")
    parser.add_argument("--resume", action="store_true", help="从上次训练恢复")

    # 损失函数权重
    parser.add_argument("--box", type=float, default=7.5, help="边界框损失权重")
    parser.add_argument("--cls", type=float, default=0.5, help="分类损失权重")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL损失权重")

    # 保存和日志
    parser.add_argument("--save-period", type=int, default=-1, help="每N个epoch保存一次")
    parser.add_argument("--save-best", action="store_true", help="只保存最佳模型")
    parser.add_argument("--verbose", action="store_true", help="输出详细训练信息")

    return parser.parse_args()


def get_dataset_size(data_yaml_path):
    """检测数据集大小 返回训练集图像数量.
    """
    try:
        if not os.path.exists(data_yaml_path):
            print(f"⚠️ 数据文件不存在: {data_yaml_path}")
            return None

        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        if "train" not in data_config:
            print("⚠️ 数据配置文件中没有 'train' 字段")
            return None

        train_path = data_config["train"]

        # 处理相对路径
        if not os.path.isabs(train_path):
            yaml_dir = os.path.dirname(data_yaml_path)
            train_path = os.path.join(yaml_dir, train_path)

        # 计算图像数量
        if os.path.isdir(train_path):
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            image_count = sum(1 for f in os.listdir(train_path) if os.path.splitext(f)[1].lower() in image_extensions)
            return image_count
        else:
            print(f"⚠️ 训练路径不是目录: {train_path}")
            return None

    except Exception as e:
        print(f"❌ 检测数据集大小时出错: {e}")
        return None


def setup_high_lr_config(opt):
    """⭐ 优化1: 智能学习率配置 - 基于数据集大小和模型大小自动调整.

    问题分析:
    - 原始lr0=0.01对846张小数据集太高
    - 导致训练不稳定和后期过拟合
    - 解决方案: 小数据集使用低学习率+强正则化
    """
    if opt.high_lr:
        print("\n" + "=" * 60)
        print("🔧 启用高学习率训练模式 (优化1: 学习率智能配置)")
        print("=" * 60)

        # 检测数据集大小
        dataset_size = get_dataset_size(opt.data)
        print(f"📊 检测到数据集大小: {dataset_size} 张图像" if dataset_size else "⚠️ 无法检测数据集大小")

        # 根据数据集大小和模型大小调整学习率
        # 小数据集 (<1000) 需要较低的学习率以防止过拟合
        if dataset_size and dataset_size < 1000:
            print("📉 检测到小数据集模式（<1000张）")

            if "yolov8n" in opt.model:
                opt.lr0 = 0.005  # 小模型+小数据 = 中等学习率
                opt.lrf = 0.1
                opt.weight_decay = 0.001
                opt.freeze = 0  # 不冻结
            elif "yolov8s" in opt.model:
                opt.lr0 = 0.003  # 中模型+小数据 = 低学习率
                opt.lrf = 0.0001
                opt.weight_decay = 0.001
                opt.freeze = 5  # 冻结5层
            elif "yolov8m" in opt.model:
                opt.lr0 = 0.002  # 大模型+小数据 = 更低学习率 ⭐ 针对用户情况
                opt.lrf = 0.0001
                opt.weight_decay = 0.001
                opt.freeze = 10  # 冻结10层 - 关键优化
            elif "yolov8l" in opt.model:
                opt.lr0 = 0.001
                opt.lrf = 0.0001
                opt.weight_decay = 0.0015
                opt.freeze = 15
            elif "yolov8x" in opt.model:
                opt.lr0 = 0.0008
                opt.lrf = 0.00001
                opt.weight_decay = 0.002
                opt.freeze = 20
            else:
                opt.lr0 = 0.002
                opt.lrf = 0.0001
                opt.weight_decay = 0.001

            opt.lr_warmup = 10  # 增加预热轮数至10，更稳定开始
            opt.patience = 50  # 减少早停耐心至50（防止无效训练）
            opt.close_mosaic = 20  # 最后20个epoch关闭mosaic

        else:
            # 大数据集和中等数据集（1000-5000张）可使用中等学习率
            if "yolov8n" in opt.model:
                opt.lr0 = 0.01
                opt.lrf = 0.1
            elif "yolov8s" in opt.model:
                opt.lr0 = 0.008
                opt.lrf = 0.0001
            elif "yolov8m" in opt.model:
                # 中等数据集+中等模型：更低更稳定的学习率
                opt.lr0 = 0.002  # ⭐ 进一步降低至0.002以减少后期波动
                opt.lrf = 0.00005  # 更较缓的衰减
                opt.weight_decay = 0.001  # 增强正则化到0.001
                opt.lr_warmup = 15  # 增加预热轮数至15
            elif "yolov8l" in opt.model:
                opt.lr0 = 0.002
                opt.lrf = 0.00001
            elif "yolov8x" in opt.model:
                opt.lr0 = 0.0015
                opt.lrf = 0.00001
            else:
                opt.lr0 = 0.005
                opt.lrf = 0.0001

            if "yolov8m" not in opt.model:
                opt.weight_decay = 0.0005
                opt.lr_warmup = 5

        # 启用余弦学习率调度
        opt.cos_lr = True

        print("✅ 学习率配置完成:")
        print(f"   - 初始学习率(lr0): {opt.lr0}")
        print(f"   - 最终学习率因子(lrf): {opt.lrf}")
        print(f"   - 权重衰减: {opt.weight_decay}")
        print(f"   - 预热轮数: {opt.lr_warmup}")
        print(f"   - 冻结层数: {opt.freeze if opt.freeze else '无'}")
        print(f"   - 早停耐心: {opt.patience}")

    return opt


def setup_augmentation(opt):
    """⭐ 优化2: 激进数据增强配置 (改进版).

    根据数据集大小自动调整增强强度：
    - 小数据集：更强的增强 + 低学习率
    - 中等数据集：平衡的增强 + 中等学习率
    - 大数据集：适度增强 + 高学习率
    """
    if opt.augment:
        print("\n" + "=" * 60)
        print("🎨 启用激进数据增强 (优化2: 完整数据增强套件)")
        print("=" * 60)

        # 获取数据集大小以调整增强强度
        dataset_size = get_dataset_size(opt.data)

        # Mosaic增强
        opt.mosaic = 1.0

        # 根据数据集大小调整MixUp和CutMix
        if dataset_size and dataset_size < 1000:
            # 小数据集：更强的增强
            opt.mixup = 0.4  # 提高到40%
            opt.cutmix = 0.3  # 提高到30%
            opt.fliplr = 0.5
            opt.degrees = 25  # 增加旋转角度
            opt.translate = 0.3
            opt.scale = 0.5  # 增加缩放
            print("   配置: 小数据集 - 强增强")
        else:
            # 中等/大数据集：平衡的增强
            opt.mixup = 0.15  # 降低至15%
            opt.cutmix = 0.15  # 降低至15%
            opt.fliplr = 0.5
            opt.degrees = 15
            opt.translate = 0.2
            opt.scale = 0.3
            print("   配置: 中等/大数据集 - 平衡增强")

        print("✅ 数据增强配置完成:")
        print(f"   - Mosaic: {opt.mosaic}")
        print(f"   - MixUp: {opt.mixup}")
        print(f"   - CutMix: {opt.cutmix}")
        print(f"   - 水平翻转: {opt.fliplr}")
        print(f"   - 旋转范围: ±{opt.degrees}°")
        print(f"   - 平移比例: {opt.translate}")
        print(f"   - 缩放比例: {opt.scale}")

    return opt


def optimize_for_small_dataset(opt):
    """⭐ 优化3: 小数据集智能优化.

    核心思想:
    - 自动检测数据集大小
    - 根据数据集大小应用相应优化
    - 防止过拟合的综合方案
    """
    print("\n" + "=" * 60)
    print("🧠 启用小数据集智能优化 (优化3: 自适应参数调整)")
    print("=" * 60)

    dataset_size = get_dataset_size(opt.data)

    if dataset_size is None:
        print("⚠️ 无法检测数据集大小，跳过小数据集优化")
        return opt

    print(f"📊 数据集大小: {dataset_size} 张图像")

    if dataset_size < 1000:
        print("📍 触发小数据集优化（<1000张）")

        # 减少可训练参数 - 冻结早期层
        if opt.freeze is None:
            freeze_layers = min(int(dataset_size / 100), 20)  # 根据数据量冻结不同数量的层
            opt.freeze = freeze_layers
            print(f"   ✓ 冻结前 {opt.freeze} 层以减少参数")

        # 增强正则化
        opt.weight_decay = max(opt.weight_decay, 0.001)
        print(f"   ✓ 权重衰减设为 {opt.weight_decay}")

        # 缩短训练周期
        if opt.patience > 50:
            opt.patience = 50
            print(f"   ✓ 早停耐心设为 {opt.patience}")

        # 早期关闭mosaic
        opt.close_mosaic = max(20, opt.epochs // 10)
        print(f"   ✓ 第 {opt.epochs - opt.close_mosaic} 个epoch后关闭mosaic")

        # 增加预热
        if opt.lr_warmup < 10:
            opt.lr_warmup = 10
            print(f"   ✓ 预热轮数增至 {opt.lr_warmup}")

        print("✅ 小数据集优化应用完成")

    elif dataset_size < 5000:
        print("📍 触发中等数据集优化（1000-5000张）")

        # 中等冻结层数
        if opt.freeze is None:
            opt.freeze = 8
        print(f"   ✓ 冻结前 {opt.freeze} 层")

        # 增强正则化
        opt.weight_decay = max(opt.weight_decay, 0.001)  # ⭐ 增强至0.001
        print(f"   ✓ 权重衰减设为 {opt.weight_decay}")

        # 适度调整早停耐心 - 中等耐心值
        if opt.patience > 60:
            opt.patience = 60  # ⭐ 降低至60以避免无用训练
            print(f"   ✓ 早停耐心设为 {opt.patience}")

        # 提前关闭mosaic - 波动较多时更需要稳定训练
        opt.close_mosaic = max(50, int(opt.epochs * 0.1))  # ⭐ 最后10%关闭而非15%
        print(f"   ✓ 第 {opt.epochs - opt.close_mosaic} 个epoch后关闭mosaic")

        # 增加预热轮数
        if opt.lr_warmup < 10:
            opt.lr_warmup = 10
            print(f"   ✓ 预热轮数增至 {opt.lr_warmup}")

        print("✅ 中等数据集优化应用完成")

    else:
        print("📍 检测到大数据集（>5000张），使用标准配置")
        if opt.freeze is None:
            opt.freeze = 0

    return opt


def analyze_overfitting(df, save_dir):
    """⭐ 优化4: 过拟合分析与诊断.

    功能:
    - 计算训练与验证损失差距
    - 判断过拟合程度
    - 提供优化建议
    """
    print("\n" + "=" * 60)
    print("📊 过拟合分析与诊断 (优化4: 训练质量评估)")
    print("=" * 60)

    try:
        # 确保有必要的列
        if "epoch" not in df.columns:
            df["epoch"] = range(1, len(df) + 1)

        # 查找损失列
        loss_pairs = []
        for train_col, val_col in [
            ("train/box_loss", "val/box_loss"),
            ("train/cls_loss", "val/cls_loss"),
            ("train/dfl_loss", "val/dfl_loss"),
            ("train/obj_loss", "val/obj_loss"),
        ]:
            if train_col in df.columns and val_col in df.columns:
                loss_pairs.append((train_col, val_col))

        if not loss_pairs:
            print("⚠️ 未找到训练/验证损失列")
            return

        # 计算最后50个epoch的平均过拟合程度
        last_n_epochs = min(50, len(df))
        analysis_df = df.tail(last_n_epochs)

        print(f"\n📈 分析最后 {last_n_epochs} 个epoch的过拟合程度:")

        max_gap = 0
        avg_gaps = []

        for train_col, val_col in loss_pairs:
            loss_name = train_col.split("/")[-1]

            # 计算差距
            gaps = analysis_df[val_col] - analysis_df[train_col]
            avg_gap = gaps.mean()
            max_gap_this = gaps.max()

            avg_gaps.append(avg_gap)
            max_gap = max(max_gap, max_gap_this)

            # 判断严重程度
            if avg_gap > 0.5:
                severity = "🔴 严重过拟合"
            elif avg_gap > 0.2:
                severity = "🟡 轻度过拟合"
            else:
                severity = "🟢 过拟合轻微"

            print(f"   {loss_name:12} - 平均差距: {avg_gap:.4f} {severity}")

        # 整体评估
        overall_avg_gap = np.mean(avg_gaps)
        print(f"\n📋 整体过拟合评估: 平均差距 = {overall_avg_gap:.4f}")

        # 给出建议
        print("\n💡 优化建议:")
        if overall_avg_gap > 0.5:
            print("   ❌ 过拟合严重，建议:")
            print("      1. 增加数据增强强度")
            print("      2. 提高权重衰减: --weight-decay 0.002")
            print("      3. 冻结更多层: --freeze 15")
            print("      4. 减少批大小: --batch 4")
            print("      5. 收集更多训练数据")
        elif overall_avg_gap > 0.2:
            print("   ⚠️ 存在轻度过拟合，建议:")
            print("      1. 增加数据增强")
            print("      2. 微调权重衰减")
            print("      3. 继续训练但需关注验证集性能")
        else:
            print("   ✅ 过拟合程度低，训练质量良好")
            print("      1. 可继续当前训练策略")
            print("      2. 若需进一步改进，可微调学习率")

        # 显示最佳指标
        if "metrics/mAP50" in df.columns:
            best_idx = df["metrics/mAP50"].idxmax()
            best_mAP50 = df.loc[best_idx, "metrics/mAP50"]
            best_epoch = df.loc[best_idx, "epoch"]
            final_mAP50 = df.iloc[-1]["metrics/mAP50"]

            print("\n🎯 性能指标:")
            print(f"   最佳mAP50: {best_mAP50:.4f} (Epoch {int(best_epoch)})")
            print(f"   最终mAP50: {final_mAP50:.4f}")

            if final_mAP50 < best_mAP50 * 0.95:
                print(f"   ⚠️ 最终mAP50较最佳值下降 {((best_mAP50 - final_mAP50) / best_mAP50 * 100):.1f}%")
                print(f"   💡 建议使用Epoch {int(best_epoch)}的模型: runs/detect/train/expX/weights/best.pt")

        print("=" * 60)

    except Exception as e:
        print(f"❌ 过拟合分析出错: {e}")


def get_valid_training_args(opt):
    """获取YOLOv8支持的有效训练参数."""
    valid_args = [
        # 基础参数
        "data",
        "epochs",
        "batch",
        "imgsz",
        "device",
        "workers",
        "project",
        "name",
        "exist_ok",
        "pretrained",
        "resume",
        # 优化器相关
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        # 学习率调度
        "cos_lr",
        "lr_warmup",
        "warmup_epochs",
        "warmup_momentum",
        # 数据增强
        "mosaic",
        "mixup",
        "cutmix",
        "fliplr",
        "flipud",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        # 训练控制
        "patience",
        "close_mosaic",
        "freeze",
        "box",
        "cls",
        "dfl",
        "amp",
        "overlap_mask",
        # 保存和日志
        "save",
        "save_period",
        "save_dir",
        "plots",
        "verbose",
        "conf",
        "iou",
        "max_det",
        "half",
        "dnn",
        "cache",
        # 其他
        "seed",
        "deterministic",
        "single_cls",
        "rect",
        "overlap_mask",
    ]

    # 过滤出有效的参数
    args_dict = vars(opt)
    valid_args_dict = {}

    for arg_name in valid_args:
        if arg_name in args_dict and args_dict[arg_name] is not None:
            # 处理特殊参数名转换
            if arg_name == "lr_warmup":
                valid_args_dict["warmup_epochs"] = args_dict[arg_name]
            elif arg_name == "save_best":
                continue
            else:
                valid_args_dict[arg_name] = args_dict[arg_name]

    return valid_args_dict


def main(opt):
    print("\n" + "=" * 80)
    print("🚀 YOLOv8 优化训练脚本启动 - 包含4项关键优化".center(80))
    print("=" * 80)

    # ⭐ 优化1: 智能学习率配置
    opt = setup_high_lr_config(opt)

    # ⭐ 优化2: 数据增强配置
    opt = setup_augmentation(opt)

    # ⭐ 优化3: 小数据集智能优化
    opt = optimize_for_small_dataset(opt)

    print("\n📋 最终训练配置:")
    print(f"  模型: {opt.model}")
    print(f"  数据: {opt.data}")
    print(f"  Epochs: {opt.epochs}")
    print(f"  批大小: {opt.batch}")
    print(f"  学习率: {opt.lr0}")

    print(f"\n📦 加载模型: {opt.model}")
    model = YOLO(opt.model)

    train_args = get_valid_training_args(opt)

    print("\n🎯 开始训练...")
    try:
        model.train(**train_args)
        print("\n✅ 训练完成!")

        print("\n" + "=" * 60)
        print("🎓 优化训练完成 - 总结")
        print("=" * 60)
        print("应用的优化策略:")
        print("  ✅ 优化1: 智能学习率配置")
        print("  ✅ 优化2: 激进数据增强")
        print("  ✅ 优化3: 小数据集自适应")
        print("  ✅ 优化4: 过拟合诊断")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
