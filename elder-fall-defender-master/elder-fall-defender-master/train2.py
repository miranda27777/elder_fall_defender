from ultralytics import YOLO

def main():
    # 不需要重新初始化模型，直接从检查点恢复
    model = YOLO("runs/train/fallDetect1/weights/last.pt")  # 直接加载中断的模型

    # 继续训练（关键修改resume参数）
    results = model.train(
        resume=True,  # 必须设为True
    )

if __name__ == '__main__':
    main()