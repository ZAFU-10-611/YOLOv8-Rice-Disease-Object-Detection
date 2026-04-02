import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import threading
import os
from datetime import datetime


class YOLODetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("水稻叶片病害智能检测系统")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.model = None
        self.model_path = None
        self.image_folder = None
        self.image_files = []
        self.current_image_idx = 0
        self.current_image = None
        self.original_image = None
        self.current_results = None
        self.save_path = None
        self.detected_images = []  # 存储检测结果图像
        
        # 创建GUI
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 使用网格布局管理主窗口
        self.root.grid_rowconfigure(1, weight=1)  # 中间部分自动扩展
        self.root.grid_columnconfigure(0, weight=1)
        
        # ========== 顶部配置面板 ==========
        config_frame = ttk.LabelFrame(self.root, text="配置信息", padding=10)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 第一行：模型选择
        row1 = ttk.Frame(config_frame)
        row1.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(row1, text="模型路径：", width=12).pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(row1, text="未加载", foreground="red", width=50)
        self.model_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="选择模型", command=self.load_model, width=12).pack(side=tk.LEFT, padx=5)
        
        # 第二行：图像源选择
        row2 = ttk.Frame(config_frame)
        row2.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(row2, text="图像源：", width=12).pack(side=tk.LEFT, padx=5)
        self.image_source_label = ttk.Label(row2, text="未选择", foreground="red", width=50)
        self.image_source_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(row2, text="单张图像", command=self.select_single_image, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="文件夹", command=self.select_folder, width=12).pack(side=tk.LEFT, padx=2)
        
        # 第三行：保存设置和置信度
        row3 = ttk.Frame(config_frame)
        row3.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(row3, text="保存路径：", width=12).pack(side=tk.LEFT, padx=5)
        self.save_path_label = ttk.Label(row3, text="默认同目录", foreground="blue", width=50)
        self.save_path_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="选择保存路径", command=self.select_save_path, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="使用默认路径", command=self.reset_save_path, width=12).pack(side=tk.LEFT, padx=2)
        
        # 第四行：置信度和操作按钮
        row4 = ttk.Frame(config_frame)
        row4.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(row4, text="置信度：", width=12).pack(side=tk.LEFT, padx=5)
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_slider = ttk.Scale(row4, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                     variable=self.conf_var, length=200)
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        self.conf_label = ttk.Label(row4, text="0.50", width=6)
        self.conf_label.pack(side=tk.LEFT, padx=2)
        self.conf_slider.configure(command=self.update_conf_label)
        
        ttk.Button(row4, text="开始检测", command=self.detect_all, width=12).pack(side=tk.LEFT, padx=10)
        
        # 进度条
        ttk.Label(row4, text="进度：", width=6).pack(side=tk.LEFT, padx=(20, 5))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(row4, variable=self.progress_var, 
                                           maximum=100, length=400, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_label = ttk.Label(row4, text="0/0", width=8)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # ========== 中间主体区域 ==========
        middle_frame = ttk.Frame(self.root)
        middle_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)
        
        # 左侧：图像显示区域
        left_frame = ttk.LabelFrame(middle_frame, text="检测结果", padding=5)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(left_frame, bg="gray20", width=700, height=600)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        
        # 右侧：检测结果详情
        right_frame = ttk.LabelFrame(middle_frame, text="检测详情", padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # 创建滚动文本框显示检测结果
        scrollbar = ttk.Scrollbar(right_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.result_text = tk.Text(right_frame, height=30, width=40, 
                                   yscrollcommand=scrollbar.set, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.result_text.yview)
        
        # 默认显示信息
        self.result_text.insert(tk.END, "检测详情：\n" + "="*40 + "\n\n")
        self.result_text.insert(tk.END, "等待检测...\n")
        self.result_text.config(state=tk.DISABLED)
        
        # ========== 底部导航和进度栏 ==========
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        # 导航栏
        nav_frame = ttk.Frame(bottom_frame)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="⬅ 上一张", command=self.prev_image, width=15).pack(side=tk.LEFT, padx=5)
        
        self.nav_info_label = ttk.Label(nav_frame, text="未加载图像")
        self.nav_info_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        ttk.Button(nav_frame, text="下一张 ➜", command=self.next_image, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(nav_frame, text="保存当前结果", command=self.save_current_result, width=15).pack(side=tk.LEFT, padx=5)
        
    def update_conf_label(self, value):
        """更新置信度标签"""  

        self.conf_label.config(text=f"{float(value):.2f}")
    
    def load_model(self):
        """加载YOLO模型"""
        file_path = filedialog.askopenfilename(
            title="选择模型",
            filetypes=[("模型文件", "*.pt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                model_name = os.path.basename(file_path)
                self.model_label.config(text=model_name, foreground="green")
                messagebox.showinfo("成功", f"模型已加载: {model_name}")
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {str(e)}")
                self.model_label.config(text="加载失败", foreground="red")
    
    def select_single_image(self):
        """选择单张图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                      ("所有文件", "*.*")]
        )
        
        if file_path:
            self.image_files = [Path(file_path)]
            self.image_folder = os.path.dirname(file_path)
            self.detected_images = []  # 清空之前的检测结果
            self.current_image_idx = 0
            self.current_results = None
            
            file_name = os.path.basename(file_path)
            self.image_source_label.config(text=file_name, foreground="green")
            self.update_result_text(f"已加载: {file_name}\n\n请点击「批量检测」进行检测")
            self.nav_info_label.config(text=f"已加载: {file_name}")
    
    def select_folder(self):
        """选择图像文件夹"""
        folder = filedialog.askdirectory(title="选择包含图像的文件夹")
        if folder:
            # 查找所有图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            self.image_files = []
            
            for ext in image_extensions:
                self.image_files.extend(Path(folder).glob(f"*{ext}"))
                self.image_files.extend(Path(folder).glob(f"*{ext.upper()}"))
            
            self.image_files = sorted(list(set(self.image_files)))
            
            if self.image_files:
                self.image_folder = folder
                folder_name = os.path.basename(folder)
                self.image_source_label.config(
                    text=f"{folder_name} ({len(self.image_files)}张)", 
                    foreground="green"
                )
                self.detected_images = []  # 清空之前的检测结果
                self.current_image_idx = 0
                self.current_results = None
                messagebox.showinfo("成功", f"找到 {len(self.image_files)} 张图像")
                self.update_result_text(f"已加载文件夹，共 {len(self.image_files)} 张图像\n\n请点击「批量检测」进行检测")
                self.nav_info_label.config(text=f"共 {len(self.image_files)} 张图像，请开始检测")
            else:
                messagebox.showwarning("提示", f"文件夹中未找到图像文件")
                self.image_source_label.config(text="未找到图像", foreground="orange")
    
    def select_save_path(self):
        """选择保存路径"""
        folder = filedialog.askdirectory(title="选择保存检测结果的文件夹")
        if folder:
            self.save_path = folder
            folder_name = os.path.basename(folder)
            self.save_path_label.config(text=f"{folder_name}", foreground="green")
            messagebox.showinfo("成功", f"保存路径已设置")
    
    def reset_save_path(self):
        """重置保存路径为默认（同图像目录）"""
        self.save_path = None
        self.save_path_label.config(text="默认同目录", foreground="blue")
    
    def update_result_text(self, content):
        """更新检测结果文本框"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "检测详情：\n" + "="*40 + "\n\n")
        self.result_text.insert(tk.END, content)
        self.result_text.config(state=tk.DISABLED)
    
    def display_image(self):
        """在canvas上显示图像"""
        if self.current_image is None:
            return
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小以适应canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 700
            canvas_height = 600
        
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 显示在canvas
        self.canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        self.canvas.create_image(x, y, image=photo, anchor=tk.NW)
        self.canvas.image = photo
    
    def display_detection_results(self, result):
        """显示检测结果详情"""
        result_info = "检测结果信息：\n" + "="*40 + "\n\n"
        
        if result.boxes is None or len(result.boxes) == 0:
            result_info += "未检测到任何目标\n"
        else:
            result_info += f"✓ 检测到 {len(result.boxes)} 个目标\n\n"
            
            # 获取类别名称
            class_names = result.names if result.names else {}
            
            # 逐个显示检测结果
            for idx, box in enumerate(result.boxes):
                result_info += f"──────────────────\n"
                result_info += f"目标 #{idx + 1}\n"
                result_info += f"──────────────────\n"
                
                # 获取类别ID
                cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                
                # 获取置信度
                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                
                # 获取坐标
                if hasattr(box.xyxy, 'cpu'):
                    coords = box.xyxy[0].cpu().numpy()
                else:
                    coords = box.xyxy[0]
                
                # 获取类别名称
                if class_names and cls_id in class_names:
                    cls_name = class_names[cls_id]
                else:
                    cls_name = f"Class {cls_id}"
                
                result_info += f"类别: {cls_name}\n"
                result_info += f"置信度: {conf:.4f} ({conf*100:.2f}%)\n"
                result_info += f"左上角: ({coords[0]:.1f}, {coords[1]:.1f})\n"
                result_info += f"右下角: ({coords[2]:.1f}, {coords[3]:.1f})\n"
                result_info += f"宽×高: {coords[2]-coords[0]:.1f}×{coords[3]-coords[1]:.1f}\n\n"
        
        self.update_result_text(result_info)
    
    def save_current_result(self):
        """保存当前检测结果"""
        if not self.detected_images or self.current_image_idx >= len(self.detected_images):
            messagebox.showwarning("警告", "请先进行批量检测")
            return
        
        try:
            # 确定保存路径
            if self.save_path:
                save_dir = Path(self.save_path)
            else:
                save_dir = Path(self.image_folder)
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取原始图像文件名
            original_path = self.detected_images[self.current_image_idx]['path']
            original_filename = os.path.basename(str(original_path))
            name, ext = os.path.splitext(original_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{name}_detected_{timestamp}{ext}"
            
            save_path = save_dir / save_filename
            
            # 保存检测结果图像
            cv2.imwrite(str(save_path), self.detected_images[self.current_image_idx]['image'])
            messagebox.showinfo("成功", f"检测结果已保存:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def detect_all(self):
        """批量检测所有图像"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if not self.image_files:
            messagebox.showwarning("警告", "请先选择图像文件夹")
            return
        
        # 在后台线程执行检测
        thread = threading.Thread(target=self._detect_all_thread)
        thread.daemon = True
        thread.start()
    
    def _detect_all_thread(self):
        """后台线程：批量检测"""
        try:
            conf = self.conf_var.get()
            total = len(self.image_files)
            detected_count = 0
            self.detected_images = []  # 清空之前的检测结果
            
            for idx, image_path in enumerate(self.image_files):
                # 读取图像
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                
                # 进行推理
                results = self.model(img, conf=conf, verbose=False)
                
                # 绘制检测结果
                result_img = results[0].plot()
                
                # 保存检测结果图像和结果对象
                self.detected_images.append({
                    'image': result_img,
                    'result': results[0],
                    'path': image_path
                })
                
                # 统计检测到的目标数
                if results[0].boxes:
                    detected_count += len(results[0].boxes)
                
                # 更新进度条
                progress_value = ((idx + 1) / total) * 100
                self.progress_var.set(progress_value)
                self.progress_label.config(text=f"{idx + 1}/{total}")
                self.nav_info_label.config(text=f"检测进度: {idx + 1}/{total}")
                self.root.update()
            
            # 检测完成，重置进度条
            self.progress_var.set(0)
            self.progress_label.config(text="0/0")
            
            # 加载第一张检测结果并显示
            if self.detected_images:
                self.current_image_idx = 0
                self.current_image = self.detected_images[0]['image']
                self.current_results = self.detected_images[0]['result']
                
                # 显示结果图像
                self.display_image()
                
                # 显示检测信息
                self.display_detection_results(self.detected_images[0]['result'])
                
                # 更新导航标签
                file_name = os.path.basename(str(self.detected_images[0]['path']))
                self.nav_info_label.config(
                    text=f"检测结果: {file_name} (1/{len(self.detected_images)})"
                )
            
            result_text = f"✓ 批量检测完成！\n\n"
            result_text += f"共扫描: {total} 张图像\n"
            result_text += f"检测到: {detected_count} 个目标\n\n"
            result_text += "提示：\n"
            result_text += "• 检测结果已显示\n"
            result_text += "• 使用「上一张」/「下一张」浏览\n"
            result_text += "• 点击「保存当前结果」保存图像"
            
            self.update_result_text(result_text)
            messagebox.showinfo("成功", f"批量检测完成！\n共 {total} 张，检测到 {detected_count} 个目标")
            
        except Exception as e:
            self.progress_var.set(0)
            self.progress_label.config(text="0/0")
            messagebox.showerror("错误", f"批量检测失败: {str(e)}")
            self.update_result_text(f"✗ 批量检测失败: {str(e)}")
    
    def prev_image(self):
        """显示上一张检测结果"""
        if not self.detected_images:
            messagebox.showwarning("警告", "请先进行批量检测")
            return
        
        new_idx = (self.current_image_idx - 1) % len(self.detected_images)
        self.current_image_idx = new_idx
        self.current_image = self.detected_images[new_idx]['image']
        self.current_results = self.detected_images[new_idx]['result']
        
        # 显示结果
        self.display_image()
        self.display_detection_results(self.detected_images[new_idx]['result'])
        
        # 更新导航标签
        file_name = os.path.basename(str(self.detected_images[new_idx]['path']))
        self.nav_info_label.config(
            text=f"检测结果: {file_name} ({new_idx + 1}/{len(self.detected_images)})"
        )
    
    def next_image(self):
        """显示下一张检测结果"""
        if not self.detected_images:
            messagebox.showwarning("警告", "请先进行批量检测")
            return
        
        new_idx = (self.current_image_idx + 1) % len(self.detected_images)
        self.current_image_idx = new_idx
        self.current_image = self.detected_images[new_idx]['image']
        self.current_results = self.detected_images[new_idx]['result']
        
        # 显示结果
        self.display_image()
        self.display_detection_results(self.detected_images[new_idx]['result'])
        
        # 更新导航标签
        file_name = os.path.basename(str(self.detected_images[new_idx]['path']))
        self.nav_info_label.config(
            text=f"检测结果: {file_name} ({new_idx + 1}/{len(self.detected_images)})"
        )


def main():
    root = tk.Tk()
    app = YOLODetectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
