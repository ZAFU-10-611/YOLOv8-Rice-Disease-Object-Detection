#!/usr/bin/env python3
"""
脚本功能：
1. 手动选择包含TXT文件的文件夹
2. 手动选择包含图片的文件夹
3. 找出所有与TXT文件同名的图片，复制到脚本执行路径下.
"""

import shutil
import sys
from pathlib import Path
from tkinter import Tk, filedialog

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def select_folder(title="选择文件夹"):
    """打开文件夹选择对话框."""
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


def get_txt_filenames(txt_folder):
    """获取TXT文件的名称（不含扩展名）."""
    txt_files = {}
    txt_path = Path(txt_folder)

    if not txt_path.exists():
        print(f"错误：TXT文件夹不存在 - {txt_folder}")
        return {}

    for file in txt_path.glob("*.txt"):
        # 获取不含扩展名的文件名
        filename_without_ext = file.stem
        txt_files[filename_without_ext] = file.name

    print(f"✓ 找到 {len(txt_files)} 个TXT文件")
    return txt_files


def find_and_copy_images(image_folder, txt_filenames, output_folder):
    """查找并复制匹配的图片."""
    image_path = Path(image_folder)
    output_path = Path(output_folder)

    if not image_path.exists():
        print(f"错误：图片文件夹不存在 - {image_folder}")
        return 0

    # 创建输出文件夹（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    # 遍历图片文件夹
    for image_file in image_path.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
            filename_without_ext = image_file.stem

            # 检查是否存在对应的TXT文件
            if filename_without_ext in txt_filenames:
                try:
                    dest_path = output_path / image_file.name
                    shutil.copy2(image_file, dest_path)
                    print(f"  ✓ 复制：{image_file.name}")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ 复制失败：{image_file.name} - {e!s}")

    return copied_count


def main():
    """主程序."""
    print("=" * 50)
    print("  图片匹配复制工具")
    print("=" * 50)
    print()

    # 第一步：选择TXT文件夹
    print("第一步：请选择包含TXT文件的文件夹...")
    txt_folder = select_folder("选择TXT文件夹")

    if not txt_folder:
        print("✗ 未选择TXT文件夹，程序退出")
        sys.exit(1)

    print(f"✓ 选择的TXT文件夹：{txt_folder}\n")

    # 获取TXT文件名列表
    txt_filenames = get_txt_filenames(txt_folder)

    if not txt_filenames:
        print("✗ 没有找到TXT文件，程序退出")
        sys.exit(1)

    # 第二步：选择图片文件夹
    print("第二步：请选择包含图片的文件夹...")
    image_folder = select_folder("选择图片文件夹")

    if not image_folder:
        print("✗ 未选择图片文件夹，程序退出")
        sys.exit(1)

    print(f"✓ 选择的图片文件夹：{image_folder}\n")

    # 第三步：选择输出路径
    print("第三步：请选择输出文件夹（保存复制的图片）...")
    output_folder = select_folder("选择输出文件夹")

    if not output_folder:
        print("✗ 未选择输出文件夹，程序退出")
        sys.exit(1)

    print(f"✓ 选择的输出文件夹：{output_folder}\n")

    # 第四步：复制匹配的图片
    print("第四步：开始复制匹配的图片...\n")

    copied_count = find_and_copy_images(image_folder, txt_filenames, output_folder)

    # 显示结果
    print()
    print("=" * 50)
    print(f"处理完成！共复制了 {copied_count} 张图片")
    print("=" * 50)


if __name__ == "__main__":
    main()
