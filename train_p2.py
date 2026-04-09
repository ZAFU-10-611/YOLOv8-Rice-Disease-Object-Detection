from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8l.pt')

# 使用 P2 配置训练
results = model.train(
    data='yolov8-p2.yaml',
    epochs=300,
    imgsz=640,
    batch=4,
    device=0,
    workers=8,
    # 优化参数（因为P2结构更复杂）
    lr0=0.01,           # 初始学习率
    lrf=0.0001,         # 最终学习率因子
    weight_decay=0.0008, # 更强的正则化
    momentum=0.937,
    optimizer='SGD',
    patience=50,        # 早停
    amp=True,           # 混合精度
    close_mosaic=15,    # 最后15个epoch关闭mosaic
    augment=True,
    mosaic=1.0,
    mixup=0.1,          # 加一点mixup
    fliplr=0.5,
    flipud=0.1,
    paience=50,
    save_period=-1,     # 只保存最好和最后
    plots=True
)