import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    parser.add_argument("--plots", action="store_true", help="保存训练过程图表")
    parser.add_argument(
        "--plot-style",
        default="curve",
        choices=["table", "curve", "both"],
        help="图表样式: table(表格), curve(曲线), both(两者都生成)",
    )

    return parser.parse_args()


def setup_high_lr_config(opt):
    """设置高学习率训练配置."""
    if opt.high_lr:
        print("启用高学习率训练模式...")

        # 根据模型大小调整学习率
        if "yolov8n" in opt.model:
            opt.lr0 = 0.01  # 小模型可以用较高学习率
            opt.lrf = 0.1  # 最终学习率 lr0 * lrf
        elif "yolov8s" in opt.model:
            opt.lr0 = 0.01
            opt.lrf = 0.0001
        elif "yolov8m" in opt.model:
            opt.lr0 = 0.01
            opt.lrf = 0.0001
        elif "yolov8l" in opt.model:
            opt.lr0 = 0.04
            opt.lrf = 0.001
        elif "yolov8x" in opt.model:
            opt.lr0 = 0.03
            opt.lrf = 0.001
        else:
            opt.lr0 = 0.05
            opt.lrf = 0.001

        # 高学习率时需要更强的正则化
        opt.weight_decay = 0.001

        # 高学习率时建议使用余弦退火
        opt.cos_lr = True

        # 高学习率时需要更长的预热
        opt.lr_warmup = 5

        print(f"高学习率配置: lr0={opt.lr0}, lrf={opt.lrf}, warmup={opt.lr_warmup}")

    return opt


def setup_augmentation(opt):
    """设置数据增强配置."""
    if opt.augment:
        # 启用完整的数据增强
        opt.mosaic = 1.0
        opt.mixup = 0.1
        opt.cutmix = 0.1
        opt.fliplr = 0.5
    return opt


def get_valid_training_args(opt):
    """获取YOLOv8支持的有效训练参数."""
    # YOLOv8支持的所有训练参数列表（基于ultralytics源码）
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
                # YOLOv8默认保存最佳模型，无需额外参数
                continue
            else:
                valid_args_dict[arg_name] = args_dict[arg_name]

    return valid_args_dict


def create_metric_table_plot(df, save_dir):
    """创建类似示例图片的表格形式指标图."""
    try:
        # 设置中文字体（如果需要显示中文）
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 提取需要显示的指标
        # 训练指标
        train_metrics = []
        val_metrics = []
        all_metrics = []

        # 可能的指标名称列表
        possible_metrics = [
            ("train/box_loss", "box_loss"),
            ("train/obj_loss", "obj_loss"),
            ("train/cls_loss", "cls_loss"),
            ("val/box_loss", "box_loss"),
            ("val/obj_loss", "obj_loss"),
            ("val/cls_loss", "cls_loss"),
            ("metrics/precision", "precision"),
            ("metrics/recall", "recall"),
            ("metrics/mAP50", "mAP50"),
            ("metrics/mAP50-95", "mAP50-95"),
        ]

        # 检查哪些指标存在
        for metric_col, short_name in possible_metrics:
            if metric_col in df.columns:
                all_metrics.append((metric_col, short_name))
                if metric_col.startswith("train/"):
                    train_metrics.append((metric_col, short_name))
                elif metric_col.startswith("val/"):
                    val_metrics.append((metric_col, short_name))

        if not all_metrics:
            print("没有找到可显示的指标")
            return

        # 创建表格数据
        epochs = df["epoch"].values if "epoch" in df.columns else range(len(df))
        num_epochs = len(epochs)

        # 创建图形 - 使用更大的图形来容纳表格
        fig_width = max(12, len(all_metrics) * 1.5)
        fig_height = max(8, num_epochs * 0.4 + 2)

        _fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("tight")
        ax.axis("off")

        # 准备表格数据
        table_data = []

        # 添加表头
        header = ["Epoch"] + [short_name for _, short_name in all_metrics]
        table_data.append(header)

        # 添加每一行数据
        for i, epoch in enumerate(epochs):
            row = [f"{int(epoch)}"]
            for metric_col, _ in all_metrics:
                value = df[metric_col].iloc[i]
                # 格式化数值
                if isinstance(value, (int, np.integer)):
                    formatted_value = f"{value}"
                else:
                    formatted_value = f"{value:.4f}".rstrip("0").rstrip(".")
                    if formatted_value == "":
                        formatted_value = "0.000"
                row.append(formatted_value)
            table_data.append(row)

        # 创建表格
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # 调整行高

        # 设置表头样式
        for j in range(len(header)):
            table[(0, j)].set_facecolor("#40466e")
            table[(0, j)].set_text_props(weight="bold", color="white")

        # 设置交替行颜色
        for i in range(1, len(table_data)):
            row_color = "#f5f5f5" if i % 2 == 1 else "#ffffff"
            for j in range(len(header)):
                table[(i, j)].set_facecolor(row_color)

        # 设置标题
        plt.title("Training Metrics Table", fontsize=16, fontweight="bold", pad=20)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, "metrics_table.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"指标表格图已保存到: {save_path}")

        plt.close()

        # 额外保存为CSV文件以便查看
        csv_path = os.path.join(save_dir, "metrics_summary.csv")
        pd.DataFrame(table_data[1:], columns=table_data[0]).to_csv(csv_path, index=False)
        print(f"指标汇总CSV已保存到: {csv_path}")

        return table_data

    except Exception as e:
        print(f"创建指标表格图时出错: {e}")
        import traceback

        traceback.print_exc()


def create_individual_metric_plots(df, save_dir):
    """为每个指标单独创建图表 - 直接保存在expxx文件夹中."""
    try:
        epochs = df["epoch"].values if "epoch" in df.columns else range(len(df))

        # 定义要绘制的指标及其配置
        metrics_config = [
            # (列名, 标题, y轴标签, 颜色, 是否为损失类型, 文件名)
            ("train/box_loss", "Box Loss (Train)", "Loss", "blue", True, "train_box_loss.png"),
            ("val/box_loss", "Box Loss (Validation)", "Loss", "red", True, "val_box_loss.png"),
            ("train/cls_loss", "Class Loss (Train)", "Loss", "green", True, "train_cls_loss.png"),
            ("val/cls_loss", "Class Loss (Validation)", "Loss", "orange", True, "val_cls_loss.png"),
            ("train/obj_loss", "Object Loss (Train)", "Loss", "purple", True, "train_obj_loss.png"),
            ("val/obj_loss", "Object Loss (Validation)", "Loss", "brown", True, "val_obj_loss.png"),
            ("train/dfl_loss", "DFL Loss (Train)", "Loss", "cyan", True, "train_dfl_loss.png"),
            ("val/dfl_loss", "DFL Loss (Validation)", "Loss", "magenta", True, "val_dfl_loss.png"),
            ("metrics/precision", "Precision", "Value", "darkblue", False, "precision.png"),
            ("metrics/recall", "Recall", "Value", "darkgreen", False, "recall.png"),
            ("metrics/mAP50", "mAP@0.5", "Value", "darkred", False, "mAP50.png"),
            ("metrics/mAP50-95", "mAP@0.5:0.95", "Value", "darkorange", False, "mAP50-95.png"),
        ]

        # 列名映射，处理可能的别名
        column_mapping = {
            "train/box_loss": "train/box_loss",
            "val/box_loss": "val/box_loss",
            "train/cls_loss": "train/cls_loss",
            "val/cls_loss": "val/cls_loss",
            "train/obj_loss": "train/obj_loss",
            "val/obj_loss": "val/obj_loss",
            "train/dfl_loss": "train/dfl_loss",
            "val/dfl_loss": "val/dfl_loss",
            "metrics/precision(B)": "metrics/precision",
            "metrics/recall(B)": "metrics/recall",
            "metrics/mAP50(B)": "metrics/mAP50",
            "metrics/mAP50-95(B)": "metrics/mAP50-95",
        }

        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # 如果没有dfl_loss列，尝试使用obj_loss
        if "train/dfl_loss" not in df.columns and "train/obj_loss" in df.columns:
            df["train/dfl_loss"] = df["train/obj_loss"]
            if "val/obj_loss" in df.columns:
                df["val/dfl_loss"] = df["val/obj_loss"]

        # 为每个指标创建单独的图表
        plots_created = 0
        for metric_col, title, ylabel, color, is_loss, filename in metrics_config:
            if metric_col in df.columns:
                plt.figure(figsize=(10, 6))

                # 绘制指标曲线
                plt.plot(epochs, df[metric_col], color=color, linewidth=2.5, marker="o", markersize=4)

                # 设置图表属性
                plt.title(title, fontsize=14, fontweight="bold", pad=12)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel(ylabel, fontsize=12)

                # 添加网格
                plt.grid(True, alpha=0.3, linestyle="--")

                # 设置坐标轴范围
                plt.xlim([0, max(epochs)])

                # 对于损失指标，自动调整y轴范围
                if is_loss:
                    data_min = df[metric_col].min()
                    data_max = df[metric_col].max()
                    data_range = data_max - data_min
                    plt.ylim([max(0, data_min - 0.1 * data_range), data_max + 0.1 * data_range])
                else:
                    # 对于评估指标，设置y轴范围为0-1
                    plt.ylim([0, 1])

                # 添加背景色和边框
                ax = plt.gca()
                ax.set_facecolor("#f8f9fa")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#6c757d")
                    spine.set_linewidth(1.5)

                # 在右上角显示最终值
                final_value = df[metric_col].iloc[-1]
                plt.annotate(
                    f"Final: {final_value:.4f}",
                    xy=(0.95, 0.95),
                    xycoords="axes fraction",
                    fontsize=10,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

                # 调整布局
                plt.tight_layout()

                # 保存图表 - 直接保存在save_dir中
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots_created += 1
                print(f"  已创建: {title} -> {filename}")

        print(f"\n共创建了 {plots_created} 个单独的指标图表，保存在: {save_dir}")

    except Exception as e:
        print(f"创建单个指标图表时出错: {e}")
        import traceback

        traceback.print_exc()


def create_comparison_plots(df, save_dir, epochs):
    """创建训练和验证的对比图表 - 直接保存在expxx文件夹中."""
    try:
        # 定义对比指标对
        comparison_pairs = [
            ("train/box_loss", "val/box_loss", "Box Loss Comparison", "box_loss_comparison.png"),
            ("train/cls_loss", "val/cls_loss", "Class Loss Comparison", "cls_loss_comparison.png"),
            ("train/obj_loss", "val/obj_loss", "Object Loss Comparison", "obj_loss_comparison.png"),
            ("train/dfl_loss", "val/dfl_loss", "DFL Loss Comparison", "dfl_loss_comparison.png"),
        ]

        for train_metric, val_metric, title, filename in comparison_pairs:
            if train_metric in df.columns and val_metric in df.columns:
                plt.figure(figsize=(10, 6))

                # 绘制训练和验证曲线
                plt.plot(epochs, df[train_metric], color="blue", linewidth=2.5, marker="o", markersize=4, label="Train")
                plt.plot(
                    epochs,
                    df[val_metric],
                    color="red",
                    linewidth=2.5,
                    marker="s",
                    markersize=4,
                    label="Validation",
                    linestyle="--",
                )

                # 设置图表属性
                plt.title(title, fontsize=14, fontweight="bold", pad=12)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Loss", fontsize=12)
                plt.legend(fontsize=11, loc="upper right")
                plt.grid(True, alpha=0.3, linestyle="--")
                plt.xlim([0, max(epochs)])

                # 设置y轴范围
                data_min = min(df[train_metric].min(), df[val_metric].min())
                data_max = max(df[train_metric].max(), df[val_metric].max())
                data_range = data_max - data_min
                plt.ylim([max(0, data_min - 0.1 * data_range), data_max + 0.1 * data_range])

                # 调整布局
                plt.tight_layout()

                # 保存图表 - 直接保存在save_dir中
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

        print(f"对比图表已保存到: {save_dir}")

    except Exception as e:
        print(f"创建对比图表时出错: {e}")


def create_metric_curves_plot(df, save_dir):
    """创建指标变化曲线图 - 直接保存在expxx文件夹中."""
    try:
        # 设置图形大小和子图布局
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.98)

        epochs = df["epoch"].values if "epoch" in df.columns else range(len(df))

        # 定义要绘制的指标及其位置
        metrics_to_plot = [
            # (行, 列, 指标列名, 标题, 颜色, 是否绘制验证集)
            (0, 0, "train/box_loss", "Box Loss", "blue", True),
            (0, 1, "train/cls_loss", "Class Loss", "green", True),
            (0, 2, "train/dfl_loss", "DFL Loss", "red", True),
            (0, 3, "metrics/precision", "Precision", "purple", False),
            (1, 0, "metrics/recall", "Recall", "orange", False),
            (1, 1, "metrics/mAP50", "mAP@0.5", "brown", False),
            (1, 2, "metrics/mAP50-95", "mAP@0.5:0.95", "pink", False),
        ]

        # 替换可能的列名
        column_mapping = {
            "train/box_loss": "train/box_loss",
            "val/box_loss": "val/box_loss",
            "train/cls_loss": "train/cls_loss",
            "val/cls_loss": "val/cls_loss",
            "train/dfl_loss": "train/dfl_loss",
            "val/dfl_loss": "val/dfl_loss",
            "metrics/precision(B)": "metrics/precision",
            "metrics/recall(B)": "metrics/recall",
            "metrics/mAP50(B)": "metrics/mAP50",
            "metrics/mAP50-95(B)": "metrics/mAP50-95",
            "metrics/precision": "metrics/precision",
            "metrics/recall": "metrics/recall",
            "metrics/mAP50": "metrics/mAP50",
            "metrics/mAP50-95": "metrics/mAP50-95",
        }

        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # 如果没有dfl_loss列，尝试使用其他损失
        if "train/dfl_loss" not in df.columns and "train/obj_loss" in df.columns:
            df["train/dfl_loss"] = df["train/obj_loss"]
            if "val/obj_loss" in df.columns:
                df["val/dfl_loss"] = df["val/obj_loss"]

        # 绘制每个指标
        for row, col, metric, title, color, plot_val in metrics_to_plot:
            ax = axes[row, col]

            # 绘制训练指标
            if metric in df.columns:
                ax.plot(epochs, df[metric], color=color, linewidth=2, label="Train", marker="o", markersize=4)

            # 绘制验证指标
            if plot_val:
                val_metric = metric.replace("train/", "val/")
                if val_metric in df.columns:
                    ax.plot(
                        epochs,
                        df[val_metric],
                        color="red",
                        linewidth=2,
                        label="Val",
                        linestyle="--",
                        marker="s",
                        markersize=4,
                    )
                    ax.legend(loc="upper right")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, max(epochs)])

            # 对于精度/召回率/mAP指标，设置y轴范围为0-1
            if metric.startswith("metrics/"):
                ax.set_ylim([0, 1])

        # 最后一个子图显示学习率（如果有）
        ax = axes[1, 3]
        if "lr/pg0" in df.columns:
            ax.plot(epochs, df["lr/pg0"], color="teal", linewidth=2)
            ax.set_title("Learning Rate", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
        else:
            # 如果没有学习率数据，显示训练信息
            ax.text(
                0.5,
                0.5,
                "Training Summary\n\n"
                + f"Epochs: {len(df)}\n"
                + f"Best mAP50: {df['metrics/mAP50'].max():.3f}\n"
                + f"Final mAP50: {df['metrics/mAP50'].iloc[-1]:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"训练指标曲线图已保存到: {save_path}")

        plt.close()

        # 创建损失曲线图
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle("Training and Validation Losses", fontsize=14, fontweight="bold")

        loss_pairs = [
            ("train/box_loss", "val/box_loss", "Box Loss"),
            ("train/cls_loss", "val/cls_loss", "Class Loss"),
            ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
        ]

        for i, (train_loss, val_loss, title) in enumerate(loss_pairs):
            ax = axes2[i]
            if train_loss in df.columns:
                ax.plot(epochs, df[train_loss], "b-", linewidth=2, label="Train")
            if val_loss in df.columns:
                ax.plot(epochs, df[val_loss], "r-", linewidth=2, label="Val")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim([0, max(epochs)])

        plt.tight_layout()
        save_path2 = os.path.join(save_dir, "training_losses.png")
        plt.savefig(save_path2, dpi=300, bbox_inches="tight")
        print(f"训练损失图已保存到: {save_path2}")

        plt.close()

    except Exception as e:
        print(f"创建指标曲线图时出错: {e}")
        import traceback

        traceback.print_exc()


def get_training_save_dir(results):
    """从训练结果中获取保存目录."""
    try:
        # 尝试从results对象中获取保存目录
        if hasattr(results, "save_dir"):
            save_dir = results.save_dir
            print(f"从训练结果获取保存目录: {save_dir}")
            return save_dir

        # 如果results是Results对象，尝试获取路径
        if hasattr(results, "path"):
            save_dir = results.path
            print(f"从训练结果获取路径: {save_dir}")
            return save_dir

        # 如果results是字典，尝试获取'save_dir'键
        if isinstance(results, dict) and "save_dir" in results:
            save_dir = results["save_dir"]
            print(f"从字典结果获取保存目录: {save_dir}")
            return save_dir

    except Exception as e:
        print(f"从训练结果获取保存目录失败: {e}")

    return None


def plot_training_results(opt, results):
    """根据plot-style参数绘制训练结果."""
    try:
        # 获取保存目录
        save_dir = get_training_save_dir(results)

        # 如果无法从结果中获取保存目录，尝试使用默认路径
        if not save_dir or not os.path.exists(save_dir):
            print("无法从训练结果获取有效保存目录，使用默认查找方法...")

            # 尝试从训练参数中构建路径
            save_dir = os.path.join(opt.project, opt.name)
            print(f"尝试使用默认保存目录: {save_dir}")

            # 检查目录是否存在
            if not os.path.exists(save_dir):
                # 尝试查找最新的exp文件夹
                possible_dirs = [
                    "runs/detect/train",
                    "runs/train",
                    "E:/yolov8/runs/detect/train",
                    "E:/yolov8/runs/train",
                ]

                for base_dir in possible_dirs:
                    if os.path.exists(base_dir):
                        # 查找最新的exp文件夹
                        exp_folders = [
                            f
                            for f in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("exp")
                        ]

                        if exp_folders:
                            # 按数字排序获取最新的
                            def extract_num(folder):
                                try:
                                    return int(folder[3:]) if folder[3:].isdigit() else 0
                                except:
                                    return 0

                            exp_folders.sort(key=extract_num, reverse=True)
                            latest_exp = exp_folders[0]
                            save_dir = os.path.join(base_dir, latest_exp)
                            print(f"找到最新训练文件夹: {save_dir}")
                            break

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"创建保存目录: {save_dir}")

        print(f"使用保存目录: {save_dir}")

        # 查找结果文件
        results_csv_path = os.path.join(save_dir, "results.csv")

        if not os.path.exists(results_csv_path):
            print(f"未找到results.csv文件: {results_csv_path}")
            print("尝试在目录中查找results.csv文件...")

            # 在目录中查找
            csv_files = [f for f in os.listdir(save_dir) if f.endswith(".csv")]
            if csv_files:
                results_csv_path = os.path.join(save_dir, csv_files[0])
                print(f"找到CSV文件: {results_csv_path}")
            else:
                print("无法找到results.csv文件")
                return

        # 读取训练结果
        print(f"\n正在读取训练结果文件: {results_csv_path}")
        df = pd.read_csv(results_csv_path)

        # 确保有epoch列
        if "epoch" not in df.columns:
            if len(df) > 0:
                df["epoch"] = range(1, len(df) + 1)
            else:
                df["epoch"] = [0]

        print(f"成功读取训练结果，共有 {len(df)} 个epoch的数据")

        # 列出所有可用的指标
        available_metrics = [col for col in df.columns if col != "epoch"]
        print(f"可用的指标: {available_metrics}")

        # 总是为每个指标创建单独的图表 - 直接保存在expxx文件夹中
        print("\n开始为每个指标创建单独的图表...")
        create_individual_metric_plots(df, save_dir)

        # 创建对比图表
        epochs = df["epoch"].values if "epoch" in df.columns else range(len(df))
        create_comparison_plots(df, save_dir, epochs)

        # 根据plot-style参数选择绘制方式
        if opt.plot_style == "table":
            create_metric_table_plot(df, save_dir)
        elif opt.plot_style == "curve":
            create_metric_curves_plot(df, save_dir)
        elif opt.plot_style == "both":
            create_metric_table_plot(df, save_dir)
            create_metric_curves_plot(df, save_dir)

        # 打印最终指标摘要
        print("\n" + "=" * 60)
        print("训练结果摘要:")
        print("=" * 60)

        if len(df) > 0:
            last_epoch = df.iloc[-1]
            print(f"最终Epoch: {int(last_epoch['epoch'])}")

            # 打印训练损失
            train_metrics = [col for col in df.columns if col.startswith("train/")]
            if train_metrics:
                print("\n训练损失:")
                for metric in train_metrics:
                    if metric in last_epoch:
                        metric_name = metric.replace("train/", "")
                        print(f"  {metric_name}: {last_epoch[metric]:.4f}")

            # 打印验证损失
            val_metrics = [col for col in df.columns if col.startswith("val/")]
            if val_metrics:
                print("\n验证损失:")
                for metric in val_metrics:
                    if metric in last_epoch:
                        metric_name = metric.replace("val/", "")
                        print(f"  {metric_name}: {last_epoch[metric]:.4f}")

            # 打印评估指标
            eval_metrics = [col for col in df.columns if col.startswith("metrics/")]
            if eval_metrics:
                print("\n评估指标:")
                for metric in eval_metrics:
                    if metric in last_epoch:
                        metric_name = metric.replace("metrics/", "")
                        print(f"  {metric_name}: {last_epoch[metric]:.4f}")

            # 找出最佳mAP50的epoch
            if "metrics/mAP50" in df.columns:
                best_mAP50_idx = df["metrics/mAP50"].idxmax()
                best_mAP50 = df.loc[best_mAP50_idx]
                print(f"\n最佳mAP50: {best_mAP50['metrics/mAP50']:.4f} (Epoch {int(best_mAP50['epoch'])})")

        print("=" * 60)

    except Exception as e:
        print(f"绘制训练结果时出错: {e}")
        import traceback

        traceback.print_exc()


def main(opt):
    # 配置高学习率训练
    opt = setup_high_lr_config(opt)

    # 配置数据增强
    opt = setup_augmentation(opt)

    print("训练配置:")
    print(f"  模型: {opt.model}")
    print(f"  数据: {opt.data}")
    print(f"  Epochs: {opt.epochs}")
    print(f"  批大小: {opt.batch}")
    print(f"  学习率: {opt.lr0}")
    print(f"  最终学习率因子: {opt.lrf}")
    print(f"  优化器: {opt.optimizer}")
    print(f"  设备: {opt.device}")

    if opt.high_lr:
        print("  ⚠️ 警告: 高学习率模式已启用，请确保数据已正确标注且充足")
        print("          如果训练出现不稳定，可降低学习率或增加批大小")

    # 加载模型
    model = YOLO(opt.model)

    # 获取有效的训练参数
    train_args = get_valid_training_args(opt)

    print("\n使用的训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")

    # 开始训练
    try:
        results = model.train(**train_args)

        print("\n训练完成!")

        # 绘制训练结果 - 传递results对象
        plot_training_results(opt, results)

        # 训练完成后输出建议
        if opt.high_lr:
            print("\n" + "=" * 50)
            print("高学习率训练完成!")
            print("建议:")
            print("1. 检查训练曲线，确保loss平稳下降")
            print("2. 验证集mAP应在合理范围内")
            print("3. 如果出现过拟合，可考虑:")
            print("   - 降低学习率 (--lr0 0.05)")
            print("   - 增加数据增强 (--augment)")
            print("   - 增加权重衰减 (--weight-decay 0.001)")
            print("4. 如果需要，可以进行微调训练:")
            # 查找最新模型路径
            possible_model_paths = []
            # 尝试获取保存目录
            save_dir = get_training_save_dir(results)
            if save_dir and os.path.exists(save_dir):
                model_path = os.path.join(save_dir, "weights", "best.pt")
                if os.path.exists(model_path):
                    possible_model_paths.append(model_path)

            # 如果没找到，尝试其他路径
            if not possible_model_paths:
                for base_dir in ["runs/detect/train", "runs/train"]:
                    if os.path.exists(base_dir):
                        exp_folders = [
                            f
                            for f in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("exp")
                        ]
                        if exp_folders:

                            def extract_num(folder):
                                try:
                                    return int(folder[3:]) if folder[3:].isdigit() else 0
                                except:
                                    return 0

                            exp_folders.sort(key=extract_num, reverse=True)
                            latest_exp = exp_folders[0]
                            model_path = os.path.join(base_dir, latest_exp, "weights", "best.pt")
                            if os.path.exists(model_path):
                                possible_model_paths.append(model_path)

            if possible_model_paths:
                model_path = possible_model_paths[0]
                print(f"   python train.py --data {opt.data} --model {model_path} --epochs 50 --lr0 0.001")
            else:
                print("   请手动指定最佳模型路径")
            print("=" * 50)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

        if opt.high_lr and ("nan" in str(e).lower() or "explode" in str(e).lower()):
            print("\n可能的学习率过高导致的问题，尝试:")
            print("1. 降低学习率: --lr0 0.05")
            print("2. 减小批大小: --batch 4")
            print("3. 使用混合精度: --amp")
            print("4. 增加预热轮数: --lr-warmup 10")
            print("5. 尝试使用Adam优化器: --optimizer Adam")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
