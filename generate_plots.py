#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 YOLOv8 训练结果绘图工具
用于从results.csv生成训练过程图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path


def load_results(results_path):
    """加载训练结果"""
    if not os.path.exists(results_path):
        print(f"❌ 错误: 找不到results.csv: {results_path}")
        return None
    
    try:
        df = pd.read_csv(results_path)
        if 'epoch' not in df.columns:
            df['epoch'] = range(1, len(df) + 1)
        print(f"✅ 成功加载结果: {len(df)} 个 epoch")
        return df
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def create_individual_metric_plots(df, save_dir):
    """为每个指标单独创建图表"""
    try:
        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
        metrics_config = [
            ('train/box_loss', 'Box Loss (Train)', 'Loss', 'blue', True, 'train_box_loss.png'),
            ('val/box_loss', 'Box Loss (Val)', 'Loss', 'red', True, 'val_box_loss.png'),
            ('train/cls_loss', 'Class Loss (Train)', 'Loss', 'green', True, 'train_cls_loss.png'),
            ('val/cls_loss', 'Class Loss (Val)', 'Loss', 'orange', True, 'val_cls_loss.png'),
            ('metrics/precision', 'Precision', 'Value', 'darkblue', False, 'precision.png'),
            ('metrics/recall', 'Recall', 'Value', 'darkgreen', False, 'recall.png'),
            ('metrics/mAP50', 'mAP@0.5', 'Value', 'darkred', False, 'mAP50.png'),
            ('metrics/mAP50-95', 'mAP@0.5:0.95', 'Value', 'darkorange', False, 'mAP50-95.png'),
        ]

        plots_created = 0
        for metric_col, title, ylabel, color, is_loss, filename in metrics_config:
            # 处理列名，支持带(B)后缀的格式
            actual_col = metric_col
            if metric_col not in df.columns and metric_col + '(B)' in df.columns:
                actual_col = metric_col + '(B)'
            
            if actual_col in df.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(epochs, df[actual_col], color=color, linewidth=2.5, marker='o', markersize=5)
                plt.title(title, fontsize=16, fontweight='bold', pad=15)
                plt.xlabel('Epoch', fontsize=13)
                plt.ylabel(ylabel, fontsize=13)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.xlim([0, max(epochs)])

                if is_loss:
                    data_min = df[actual_col].min()
                    data_max = df[actual_col].max()
                    data_range = data_max - data_min
                    plt.ylim([max(0, data_min - 0.1 * data_range), data_max + 0.1 * data_range])
                else:
                    plt.ylim([0, 1])

                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')

                final_value = df[actual_col].iloc[-1]
                plt.annotate(f'Final: {final_value:.4f}',
                             xy=(0.95, 0.95), xycoords='axes fraction',
                             fontsize=11, ha='right', va='top',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

                plt.tight_layout()
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created += 1
                print(f"  ✅ 已创建: {title}")

        print(f"\n📊 共创建了 {plots_created} 个指标图表")
        return True

    except Exception as e:
        print(f"❌ 创建图表时出错: {e}")
        return False


def create_comparison_plots(df, save_dir, epochs):
    """创建训练和验证的对比图表"""
    try:
        comparison_pairs = [
            ('train/box_loss', 'val/box_loss', 'Box Loss Comparison', 'box_loss_comparison.png'),
            ('train/cls_loss', 'val/cls_loss', 'Class Loss Comparison', 'cls_loss_comparison.png'),
        ]

        for train_metric, val_metric, title, filename in comparison_pairs:
            # 处理带(B)后缀的列名
            actual_train = train_metric if train_metric in df.columns else (train_metric + '(B)' if train_metric + '(B)' in df.columns else None)
            actual_val = val_metric if val_metric in df.columns else (val_metric + '(B)' if val_metric + '(B)' in df.columns else None)
            
            if actual_train and actual_val:
                plt.figure(figsize=(12, 6))
                plt.plot(epochs, df[actual_train], color='blue', linewidth=2.5, marker='o', markersize=5, label='Train')
                plt.plot(epochs, df[actual_val], color='red', linewidth=2.5, marker='s', markersize=5, label='Val', linestyle='--')
                plt.title(title, fontsize=16, fontweight='bold', pad=15)
                plt.xlabel('Epoch', fontsize=13)
                plt.ylabel('Loss', fontsize=13)
                plt.legend(fontsize=12, loc='upper right')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.xlim([0, max(epochs)])
                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')
                plt.tight_layout()
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✅ 已创建: {title}")

    except Exception as e:
        print(f"❌ 创建对比图表时出错: {e}")


def create_metric_table_plot(df, save_dir):
    """创建指标表格图"""
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        all_metrics = []
        possible_metrics = [
            ('train/box_loss', 'Box Loss'),
            ('train/cls_loss', 'Cls Loss'),
            ('val/box_loss', 'Val Box'),
            ('val/cls_loss', 'Val Cls'),
            ('metrics/precision', 'Precision'),
            ('metrics/recall', 'Recall'),
            ('metrics/mAP50', 'mAP50'),
            ('metrics/mAP50-95', 'mAP50-95')
        ]

        for metric_col, short_name in possible_metrics:
            # 处理列名，支持带(B)后缀的格式
            actual_col = metric_col
            if metric_col not in df.columns and metric_col + '(B)' in df.columns:
                actual_col = metric_col + '(B)'
            
            if actual_col in df.columns:
                all_metrics.append((actual_col, short_name))

        if not all_metrics:
            print("⚠️ 没有找到可显示的指标")
            return

        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
        num_epochs = len(epochs)

        fig_width = max(14, len(all_metrics) * 1.8)
        fig_height = max(10, num_epochs * 0.3)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')

        table_data = [['Epoch'] + [short_name for _, short_name in all_metrics]]

        # 显示所有epoch（或者按步长显示）
        for i, epoch in enumerate(epochs):
            row = [f"{int(epoch)}"]
            for metric_col, _ in all_metrics:
                value = df[metric_col].iloc[i]
                if isinstance(value, (int, np.integer)):
                    formatted_value = f"{value}"
                else:
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
                    if formatted_value == '':
                        formatted_value = '0.000'
                row.append(formatted_value)
            table_data.append(row)

        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#40466e')
            table[(0, j)].set_text_props(weight='bold', color='white', fontsize=10)

        for i in range(1, len(table_data)):
            row_color = '#f5f5f5' if i % 2 == 1 else '#ffffff'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(row_color)

        plt.title('Training Metrics Table', fontsize=17, fontweight='bold', pad=20)
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'metrics_table.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已创建: Metrics Table")

    except Exception as e:
        print(f"❌ 创建指标表时出错: {e}")


def analyze_overfitting(df):
    """过拟合分析"""
    print("\n" + "="*60)
    print("📊 过拟合分析")
    print("="*60)
    
    try:
        loss_pairs = []
        column_pairs = [
            ('train/box_loss', 'val/box_loss'),
            ('train/cls_loss', 'val/cls_loss'),
        ]
        
        for train_col, val_col in column_pairs:
            # 处理带(B)后缀的列名
            actual_train = train_col if train_col in df.columns else (train_col + '(B)' if train_col + '(B)' in df.columns else None)
            actual_val = val_col if val_col in df.columns else (val_col + '(B)' if val_col + '(B)' in df.columns else None)
            
            if actual_train and actual_val:
                loss_pairs.append((actual_train, actual_val))
        
        if not loss_pairs:
            print("⚠️ 无法进行过拟合分析")
            return
        
        last_n_epochs = min(50, len(df))
        analysis_df = df.tail(last_n_epochs)
        
        print(f"\n📈 分析最后 {last_n_epochs} 个epoch的过拟合程度:\n")
        
        avg_gaps = []
        for train_col, val_col in loss_pairs:
            loss_name = train_col.split('/')[-1]
            gaps = analysis_df[val_col] - analysis_df[train_col]
            avg_gap = gaps.mean()
            avg_gaps.append(avg_gap)
            
            if avg_gap > 0.5:
                severity = "🔴 严重过拟合"
            elif avg_gap > 0.2:
                severity = "🟡 轻度过拟合"
            else:
                severity = "🟢 过拟合轻微"
            
            print(f"   {loss_name:12} - 平均差距: {avg_gap:.4f} {severity}")
        
        overall_avg_gap = np.mean(avg_gaps)
        print(f"\n📋 整体评估: 平均差距 = {overall_avg_gap:.4f}")
        
        # 显示最佳指标
        mAP50_col = None
        if 'metrics/mAP50' in df.columns:
            mAP50_col = 'metrics/mAP50'
        elif 'metrics/mAP50(B)' in df.columns:
            mAP50_col = 'metrics/mAP50(B)'
        
        if mAP50_col:
            best_idx = df[mAP50_col].idxmax()
            best_mAP50 = df.loc[best_idx, mAP50_col]
            best_epoch = int(df.loc[best_idx, 'epoch'])
            final_mAP50 = df.iloc[-1][mAP50_col]
            
            print(f"\n🎯 性能指标:")
            print(f"   最佳mAP50: {best_mAP50:.4f} (Epoch {best_epoch})")
            print(f"   最终mAP50: {final_mAP50:.4f}")
            
            improvement = ((final_mAP50 - df.iloc[0][mAP50_col]) / df.iloc[0][mAP50_col] * 100) if df.iloc[0][mAP50_col] > 0 else 0
            print(f"   📈 整体提升: {improvement:.1f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")


def main():
    print("\n" + "="*80)
    print("🎨 YOLOv8 训练结果绘图工具".center(80))
    print("="*80 + "\n")
    
    # 查找results.csv
    possible_paths = [
        "E:/yolov8/runs/detect/runs/train/exp5/results.csv",
        "runs/detect/runs/train/exp5/results.csv",
        "runs/train/exp5/results.csv",
    ]
    
    results_path = None
    for path in possible_paths:
        if os.path.exists(path):
            results_path = path
            break
    
    if results_path is None:
        print("❌ 未找到results.csv文件")
        print("\n📂 请检查以下路径:")
        for path in possible_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    print(f"📂 使用结果文件: {results_path}\n")
    
    # 加载数据
    df = load_results(results_path)
    if df is None:
        sys.exit(1)
    
    # 确定输出目录
    save_dir = os.path.dirname(results_path)
    print(f"📁 输出目录: {save_dir}\n")
    
    # 生成图表
    print("🎨 正在生成图表...\n")
    print("📊 单个指标图表:")
    create_individual_metric_plots(df, save_dir)
    
    print("\n📊 对比图表:")
    epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
    create_comparison_plots(df, save_dir, epochs)
    
    print("\n📊 指标表格:")
    create_metric_table_plot(df, save_dir)
    
    # 分析过拟合
    analyze_overfitting(df)
    
    print("\n" + "="*80)
    print("✅ 完成！所有图表已保存到:".center(80))
    print(save_dir.center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
