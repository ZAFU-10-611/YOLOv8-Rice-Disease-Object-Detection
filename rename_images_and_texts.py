#!/usr/bin/env python3
"""
脚本功能：
1. 手动选择图片文件夹，按自定义规则重命名所有图片
2. 手动选择TXT文件夹，找到与原图片名称相同的TXT，重命名为新的图片名称
3. 所有结果保留在原文件夹.
"""

import os
import sys
from pathlib import Path
from tkinter import Tk, filedialog, simpledialog

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def select_folder(title="选择文件夹"):
    """打开文件夹选择对话框."""
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


def input_string(prompt="请输入", title="输入"):
    """弹出输入对话框."""
    root = Tk()
    root.withdraw()
    result = simpledialog.askstring(title, prompt)
    root.destroy()
    return result


def input_yes_no(prompt="是否继续?", title="确认"):
    """弹出是/否对话框."""
    root = Tk()
    root.withdraw()

    def on_yes():
        root.result = True
        root.destroy()

    def on_no():
        root.result = False
        root.destroy()

    from tkinter import messagebox

    root.withdraw()
    root.destroy()

    root = Tk()
    root.withdraw()
    result = messagebox.askyesno(title, prompt)
    root.destroy()
    return result


def rename_images(image_folder, prefix):
    """重命名图片文件."""
    image_path = Path(image_folder)

    if not image_path.exists():
        print(f"✗ 错误：图片文件夹不存在 - {image_folder}")
        return {}

    # 获取所有图片文件
    image_files = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print("✗ 错误：找不到任何图片文件")
        return {}

    # 按文件名排序
    image_files.sort(key=lambda x: x.name)

    rename_map = {}  # 存储旧名称 -> 新名称的映射

    for index, image_file in enumerate(image_files, 1):
        # 生成新文件名：前缀_数字.扩展名
        new_filename = f"{prefix}_{index}{image_file.suffix}"
        new_path = image_path / new_filename

        try:
            # 处理文件名冲突
            if new_path.exists():
                print(f"  ⚠ 警告：目标文件已存在，跳过 {image_file.name}")
                continue

            os.rename(image_file, new_path)
            print(f"  ✓ {image_file.name} → {new_filename}")

            # 记录映射（不含扩展名）
            rename_map[image_file.stem] = new_filename.rsplit(".", 1)[0]

        except Exception as e:
            print(f"  ✗ 重命名失败：{image_file.name} - {e!s}")

    return rename_map


def rename_txt_files(txt_folder, rename_map):
    """根据映射重命名TXT文件."""
    txt_path = Path(txt_folder)

    if not txt_path.exists():
        print(f"✗ 错误：TXT文件夹不存在 - {txt_folder}")
        return 0

    renamed_count = 0

    for txt_file in txt_path.glob("*.txt"):
        old_name_without_ext = txt_file.stem

        # 检查是否在映射中
        if old_name_without_ext in rename_map:
            new_name_without_ext = rename_map[old_name_without_ext]
            new_filename = f"{new_name_without_ext}.txt"
            new_path = txt_path / new_filename

            try:
                # 处理文件名冲突
                if new_path.exists():
                    print(f"  ⚠ 警告：目标文件已存在，跳过 {txt_file.name}")
                    continue

                os.rename(txt_file, new_path)
                print(f"  ✓ {txt_file.name} → {new_filename}")
                renamed_count += 1

            except Exception as e:
                print(f"  ✗ 重命名失败：{txt_file.name} - {e!s}")

    return renamed_count


def main():
    """主程序."""
    print("=" * 60)
    print("  图片和TXT文件批量重命名工具")
    print("=" * 60)
    print()

    # 第一步：选择图片文件夹
    print("第一步：请选择包含图片的文件夹...")
    image_folder = select_folder("选择图片文件夹")

    if not image_folder:
        print("✗ 未选择图片文件夹，程序退出")
        sys.exit(1)

    print(f"✓ 选择的图片文件夹：{image_folder}\n")

    # 第二步：获取自定义前缀
    print("第二步：请输入自定义前缀...")
    prefix = input_string("请输入文件名前缀\n例如：image、photo、data 等", "输入前缀")

    if not prefix:
        print("✗ 未输入前缀，程序退出")
        sys.exit(1)

    print(f"✓ 自定义前缀：{prefix}")
    print(f"  文件名格式：{prefix}_1, {prefix}_2, {prefix}_3...\n")

    # 第三步：重命名图片
    print("第三步：开始重命名图片...\n")
    rename_map = rename_images(image_folder, prefix)

    if not rename_map:
        print("✗ 没有成功重命名任何图片，程序退出")
        sys.exit(1)

    print(f"\n✓ 成功重命名 {len(rename_map)} 张图片\n")

    # 第四步：选择TXT文件夹
    print("第四步：请选择包含TXT文件的文件夹...")
    txt_folder = select_folder("选择TXT文件夹")

    if not txt_folder:
        print("✗ 未选择TXT文件夹")
        print("✓ 仅重命名了图片，程序结束")
        sys.exit(0)

    print(f"✓ 选择的TXT文件夹：{txt_folder}\n")

    # 第五步：重命名TXT
    print("第五步：开始重命名匹配的TXT文件...\n")
    txt_count = rename_txt_files(txt_folder, rename_map)

    # 显示最终结果
    print()
    print("=" * 60)
    print("处理完成！")
    print(f"  • 重命名图片数：{len(rename_map)} 张")
    print(f"  • 重命名TXT数：{txt_count} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
