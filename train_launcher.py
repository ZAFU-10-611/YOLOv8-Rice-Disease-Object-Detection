#!/usr/bin/env python3
"""
🎯 YOLOv8 水稻病害检测 - 优化训练启动器
根据用户选择的方案启动不同的训练配置.
"""

import os
import subprocess
import sys


def print_header():
    print("\n" + "=" * 80)
    print("🚀 YOLOv8 优化训练启动器 - 针对中等数据集优化".center(80))
    print("=" * 80)
    print("\n📊 当前配置分析:")
    print("  • 数据集大小: 3907张图像 (中等数据集)")
    print("  • 模型: yolov8m.pt")
    print("  • 前次最佳成绩: mAP50 = 0.81")
    print("  • 前次问题: 后期波动 (±0.06)")
    print("\n✨ 本次优化:")
    print("  ✅ 学习率: 0.01 → 0.003 (减少波动)")
    print("  ✅ 冻结层: 5 → 8 (减少参数)")
    print("  ✅ 正则化: 0.0005 → 0.0008 (防止过拟合)")
    print("  ✅ 预热轮数: 5 → 10 (更稳定)")
    print("\n📈 预期目标: mAP50 = 0.82-0.85 (波动 ±0.02-0.03)")
    print("=" * 80 + "\n")


def show_options():
    print("请选择训练方案:\n")
    print("  [1] 方案A: 从0开始重新训练 (推荐用于对比)")
    print("      • 完整训练 400 epochs")
    print("      • 应用所有优化参数")
    print("      • 预期时间: ~20小时 (RTX显卡)")
    print()
    print("  [2] 方案B: 继续之前的训练 (恢复模式)")
    print("      • 从Epoch 137继续")
    print("      • 应用改进的超参数")
    print("      • 预期时间: ~10小时")
    print()
    print("  [3] 方案C: 激进优化 (如果继续波动)")
    print("      • 更低的学习率 (0.002)")
    print("      • 更多冻结层 (10)")
    print("      • 更强的正则化")
    print("      • 预期时间: ~20小时")
    print()
    print("  [0] 退出")
    print()
    choice = input("请输入选择 (0-3): ").strip()
    return choice


def run_training(plan):
    """执行训练."""
    base_cmd = ["python", "train_full_optimized.py"]

    if plan == "1":
        # 方案A
        cmd = [
            *base_cmd,
            "--data",
            "rice_disease.yaml",
            "--model",
            "yolov8m.pt",
            "--epochs",
            "400",
            "--batch",
            "4",
            "--high-lr",
            "--augment",
            "--workers",
            "0",
            "--plot-style",
            "both",
        ]
        print("\n" + "=" * 80)
        print("🚀 启动方案A: 从0开始重新训练".center(80))
        print("=" * 80)

    elif plan == "2":
        # 方案B
        cmd = [
            *base_cmd,
            "--data",
            "rice_disease.yaml",
            "--model",
            "runs/train/exp4/weights/last.pt",
            "--epochs",
            "300",
            "--batch",
            "4",
            "--high-lr",
            "--augment",
            "--resume",
            "--workers",
            "0",
            "--plot-style",
            "both",
        ]
        print("\n" + "=" * 80)
        print("🚀 启动方案B: 继续之前的训练 (恢复模式)".center(80))
        print("=" * 80)

    elif plan == "3":
        # 方案C
        cmd = [
            *base_cmd,
            "--data",
            "rice_disease.yaml",
            "--model",
            "yolov8m.pt",
            "--epochs",
            "350",
            "--batch",
            "8",
            "--lr0",
            "0.002",
            "--lrf",
            "0.00001",
            "--weight-decay",
            "0.001",
            "--freeze",
            "10",
            "--augment",
            "--workers",
            "0",
            "--plot-style",
            "both",
        ]
        print("\n" + "=" * 80)
        print("🚀 启动方案C: 激进优化".center(80))
        print("=" * 80)
    else:
        print("❌ 无效选择")
        return False

    print("\n📋 执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 80 + "\n")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⏸️ 训练中断")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return False


def print_footer(success):
    print("\n" + "=" * 80)
    if success:
        print("✅ 训练完成！".center(80))
        print("\n📂 结果位置:")
        print("  • 模型权重: runs/train/expX/weights/best.pt")
        print("  • 指标图表: runs/train/expX/metrics_table.png")
        print("  • 损失曲线: runs/train/expX/train_box_loss.png")
        print("  • mAP曲线: runs/train/expX/mAP50.png")
        print("\n💡 下一步:")
        print("  1. 查看 TRAINING_RECOMMENDATIONS.md 了解详细分析")
        print("  2. 对比前后训练结果的关键指标")
        print("  3. 根据结果决定是否继续优化")
    else:
        print("⚠️ 训练未正常完成".center(80))
    print("=" * 80 + "\n")


def main():
    print_header()

    # 检查必要文件
    if not os.path.exists("train_full_optimized.py"):
        print("❌ 错误: 找不到 train_full_optimized.py")
        sys.exit(1)

    if not os.path.exists("rice_disease.yaml"):
        print("❌ 错误: 找不到 rice_disease.yaml")
        sys.exit(1)

    while True:
        choice = show_options()

        if choice == "0":
            print("👋 退出")
            break
        elif choice in ["1", "2", "3"]:
            success = run_training(choice)
            print_footer(success)
            break
        else:
            print("❌ 无效选择，请重新输入\n")


if __name__ == "__main__":
    main()
