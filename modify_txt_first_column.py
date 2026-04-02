#!/usr/bin/env python3
"""
脚本功能：
1. 手动选择文件夹
2. 输入自定义字符前缀
3. 找到文件夹里以该前缀开头的所有TXT文件
4. 把这些TXT文件里的第一列数字改成自定义输入的数字.
"""

import sys
from pathlib import Path
from tkinter import Tk, filedialog, simpledialog


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


def modify_txt_first_column(folder_path, prefix, new_value):
    """修改TXT文件第一列数字."""
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"✗ 错误：文件夹不存在 - {folder_path}")
        return 0

    # 找所有以前缀开头的TXT文件
    txt_files = [f for f in folder_path.glob(f"{prefix}*.txt")]

    if not txt_files:
        print(f"✗ 没有找到以 '{prefix}' 开头的TXT文件")
        return 0

    txt_files.sort(key=lambda x: x.name)

    modified_count = 0

    for txt_file in txt_files:
        try:
            # 读取文件内容
            with open(txt_file, encoding="utf-8") as f:
                lines = f.readlines()

            # 修改每一行的第一列
            modified_lines = []
            for line in lines:
                line = line.rstrip("\n")

                # 使用空格或Tab分隔
                if line.strip():  # 非空行
                    parts = line.split()
                    if len(parts) > 0:
                        # 替换第一列
                        parts[0] = new_value
                        modified_line = " ".join(parts)
                        modified_lines.append(modified_line + "\n")
                    else:
                        modified_lines.append(line + "\n")
                else:
                    # 空行保持原样
                    modified_lines.append(line + "\n")

            # 写入文件
            with open(txt_file, "w", encoding="utf-8") as f:
                f.writelines(modified_lines)

            print(f"  ✓ 修改完成：{txt_file.name}")
            modified_count += 1

        except Exception as e:
            print(f"  ✗ 修改失败：{txt_file.name} - {e!s}")

    return modified_count


def main():
    """主程序."""
    print("=" * 60)
    print("  TXT文件第一列数字批量修改工具")
    print("=" * 60)
    print()

    # 第一步：选择文件夹
    print("第一步：请选择包含TXT文件的文件夹...")
    folder_path = select_folder("选择文件夹")

    if not folder_path:
        print("✗ 未选择文件夹，程序退出")
        sys.exit(1)

    print(f"✓ 选择的文件夹：{folder_path}\n")

    # 第二步：输入前缀
    print("第二步：请输入文件名前缀...")
    prefix = input_string("输入要查找的文件名前缀\n例如：l 或 image", "输入前缀")

    if not prefix:
        print("✗ 未输入前缀，程序退出")
        sys.exit(1)

    print(f"✓ 输入的前缀：{prefix}\n")

    # 第三步：输入新的第一列数字
    print("第三步：请输入新的第一列数字...")
    new_value = input_string("输入要替换成的数字\n例如：0、1、2等", "输入新数字")

    if not new_value:
        print("✗ 未输入数字，程序退出")
        sys.exit(1)

    print(f"✓ 新的第一列数字：{new_value}\n")

    # 第四步：执行修改
    print("第四步：开始修改TXT文件...\n")
    modified_count = modify_txt_first_column(folder_path, prefix, new_value)

    # 显示结果
    print()
    print("=" * 60)
    if modified_count > 0:
        print(f"✓ 修改完成！共修改了 {modified_count} 个文件")
    else:
        print("✗ 没有修改任何文件")
    print("=" * 60)


if __name__ == "__main__":
    main()
