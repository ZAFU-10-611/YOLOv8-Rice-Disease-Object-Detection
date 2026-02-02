# -*- coding: utf-8 -*-
"""
功能：
1. 运行程序后弹出文件夹选择窗口
2. 批量处理该文件夹下所有 TXT 文件
3. 若每行第一列为 0，则修改为 1
4. 其余内容保持不变
5. 直接覆盖原 TXT 文件
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox


def modify_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped_line = line.strip()

        # 空行直接保留
        if not stripped_line:
            new_lines.append(line)
            continue

        parts = stripped_line.split()

        if parts[0] == "1":
            parts[0] = "0"
        elif parts[0] == "2":
            parts[0] = "1"
        elif parts[0] == "3":
            parts[0] = "2"
        elif parts[0] == "4":
            parts[0] = "3"
        elif parts[0] == "5":
            parts[0] = "4"

        new_lines.append(" ".join(parts) + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def batch_process(folder_path):
    txt_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".txt")
    ]

    if not txt_files:
        messagebox.showinfo("提示", "该文件夹下未找到 TXT 文件。")
        return

    count = 0
    for txt_file in txt_files:
        modify_txt(txt_file)
        count += 1

    messagebox.showinfo(
        "完成",
        f"批量处理完成！\n\n处理文件数：{count}"
    )


def main():
    # 初始化 Tkinter（不显示主窗口）
    root = tk.Tk()
    root.withdraw()

    # 选择文件夹
    folder_path = filedialog.askdirectory(
        title="请选择包含 TXT 文件的文件夹"
    )

    if not folder_path:
        messagebox.showinfo("提示", "未选择文件夹，程序已退出。")
        return

    try:
        batch_process(folder_path)
    except Exception as e:
        messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    main()

import os
import tkinter as tk
from tkinter import filedialog


# 打开文件夹选择对话框
def select_folder():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择文件夹")  # 打开文件夹选择框
    return folder_path


# 检查文件是否为空
def check_empty_txt_files(folder_path):
    # 获取文件夹下所有的TXT文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # 遍历所有TXT文件
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        # 判断文件大小是否为0
        if os.path.getsize(file_path) == 0:
            print(f"文件为空: {txt_file}")


# 主函数
def main():
    folder_path = select_folder()  # 选择文件夹路径
    if folder_path:
        check_empty_txt_files(folder_path)  # 检查文件夹中的TXT文件
    else:
        print("未选择文件夹路径")


# 调用主函数
if __name__ == "__main__":
    main()