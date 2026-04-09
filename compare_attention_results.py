"""
YOLOv8 注意力机制对比分析工具
用于比较默认模型和注意力模型的训练结果
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_results_csv(search_path='runs/train'):
    """查找所有results.csv文件"""
    results = {}
    runs_path = Path(search_path)
    
    if runs_path.exists():
        for exp_dir in runs_path.glob('*/'):
            for results_file in exp_dir.glob('results.csv'):
                exp_name = exp_dir.name
                results[exp_name] = str(results_file)
    
    return results


def load_results(results_csv):
    """加载训练结果"""
    try:
        df = pd.read_csv(results_csv)
        # 清理列名（删除前后空格）
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def plot_comparison(results_dict):
    """绘制对比图表"""
    if not results_dict or len(results_dict) < 2:
        print("⚠️ 需要至少2个训练结果进行对比")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv8 注意力机制训练对比', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, (name, df) in enumerate(results_dict.items()):
        if df is None:
            continue
        
        color = colors[idx % len(colors)]
        
        # 绘制mAP50
        if 'metrics/mAP50' in df.columns:
            axes[0, 0].plot(df.index, df['metrics/mAP50'], 
                           label=name, marker='o', color=color, linewidth=2)
        
        # 绘制损失
        if 'train/loss' in df.columns and 'val/loss' in df.columns:
            axes[0, 1].plot(df.index, df['train/loss'], 
                           label=f'{name} (train)', linestyle='-', color=color)
            axes[0, 1].plot(df.index, df['val/loss'], 
                           label=f'{name} (val)', linestyle='--', color=color)
        
        # 绘制Box Loss
        if 'train/box_loss' in df.columns:
            axes[1, 0].plot(df.index, df['train/box_loss'], 
                           label=name, marker='s', color=color, linewidth=2)
        
        # 绘制Cls Loss
        if 'train/cls_loss' in df.columns:
            axes[1, 1].plot(df.index, df['train/cls_loss'], 
                           label=name, marker='^', color=color, linewidth=2)
    
    # 设置子图标签
    axes[0, 0].set_title('mAP50 (越高越好)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP50')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('总损失 (越低越好)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Box Loss (越低越好)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Box Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Cls Loss (越低越好)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cls Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 对比图表已保存: attention_comparison.png")
    plt.show()


def print_statistics(results_dict):
    """打印统计信息"""
    print("\n" + "="*80)
    print("📊 训练结果统计".center(80))
    print("="*80)
    
    stats = {}
    
    for name, df in results_dict.items():
        if df is None:
            continue
        
        print(f"\n📌 {name}")
        
        # 获取最佳mAP50
        if 'metrics/mAP50' in df.columns:
            max_mAP50 = df['metrics/mAP50'].max()
            max_epoch = df['metrics/mAP50'].idxmax()
            final_mAP50 = df['metrics/mAP50'].iloc[-1]
            
            print(f"   最佳mAP50: {max_mAP50:.4f} (Epoch {int(max_epoch)})")
            print(f"   最终mAP50: {final_mAP50:.4f}")
            
            stats[name] = {
                'best_mAP50': max_mAP50,
                'final_mAP50': final_mAP50,
                'best_epoch': max_epoch
            }
        
        # 损失信息
        if 'train/loss' in df.columns:
            final_train_loss = df['train/loss'].iloc[-1]
            final_val_loss = df['val/loss'].iloc[-1] if 'val/loss' in df.columns else None
            
            print(f"   最终训练损失: {final_train_loss:.4f}")
            if final_val_loss:
                print(f"   最终验证损失: {final_val_loss:.4f}")
                print(f"   过拟合程度: {final_val_loss - final_train_loss:.4f}")
    
    # 对比分析
    if len(stats) >= 2:
        print("\n" + "="*80)
        print("🏆 性能排名".center(80))
        print("="*80)
        
        sorted_stats = sorted(stats.items(), 
                             key=lambda x: x[1]['final_mAP50'], 
                             reverse=True)
        
        for rank, (name, stat) in enumerate(sorted_stats, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            print(f"{medal} {rank}. {name}: {stat['final_mAP50']:.4f} mAP50")


def main():
    print("\n" + "="*80)
    print("🔍 YOLOv8 注意力机制对比分析".center(80))
    print("="*80)
    
    # 查找所有结果
    print("\n📂 搜索训练结果...")
    results_dirs = find_results_csv()
    
    if not results_dirs:
        print("❌ 未找到任何训练结果")
        print("   请先运行训练: python train_with_attention.py")
        return
    
    print(f"✅ 找到 {len(results_dirs)} 个训练结果\n")
    
    # 让用户选择要对比的模型
    print("可用的训练结果:")
    for idx, (name, path) in enumerate(results_dirs.items(), 1):
        print(f"  {idx}. {name}")
    
    print("\n选择对比模型:")
    print("  - 输入'all'对比所有模型")
    print("  - 输入数字编号选择单个模型")
    print("  - 输入多个数字用逗号分隔，例如: 1,2,3")
    
    user_input = input("\n请选择 (all/1/1,2,3): ").strip().lower()
    
    selected = {}
    
    if user_input == 'all':
        selected = results_dirs
    else:
        try:
            indices = [int(x.strip())-1 for x in user_input.split(',')]
            items = list(results_dirs.items())
            for idx in indices:
                if 0 <= idx < len(items):
                    name, path = items[idx]
                    selected[name] = path
        except:
            print("❌ 无效输入")
            return
    
    # 加载选中的结果
    print("\n📊 加载结果...")
    results_data = {}
    for name, path in selected.items():
        df = load_results(path)
        if df is not None:
            results_data[name] = df
            print(f"✅ {name}: {len(df)} epochs")
        else:
            print(f"❌ {name}: 加载失败")
    
    if not results_data:
        print("❌ 未能加载任何结果")
        return
    
    # 打印统计
    print_statistics(results_data)
    
    # 绘制对比图
    try:
        plot_comparison(results_data)
    except Exception as e:
        print(f"⚠️ 绘制图表失败: {e}")
    
    print("\n" + "="*80)
    print("✅ 分析完成！".center(80))
    print("="*80)
    
    print("\n💡 建议:")
    print("  1. 查看mAP50曲线，选择性能最好的模型")
    print("  2. 检查过拟合情况，确保验证损失正常下降")
    print("  3. 使用best.pt权重进行推理")
    print("\n📁 训练结果位置: runs/train_attn/")


if __name__ == '__main__':
    main()
