from ultralytics import YOLO



def main():
    # 加载模型
    model = YOLO("yolov8s.pt")

    # 训练模型
    results = model.train(

        # 设置训练参数
        data='./fallDownPic/mydata.yaml',  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        batch=16,  # 批量大小
        imgsz=640,  # 输入图像尺寸
        device='0',  # 使用GPU (0 或 0,1,2,3 或 'cpu')
        workers=8,  # 数据加载线程数
        image_weights=True,  # 解决类别不平衡问题
        project='runs/train/',  # 保存结果的目录
        name='fallDetect1',  # 实验名称
        exist_ok=True,  # 是否允许覆盖现有实验

        # 设置优化器参数
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率(lr0*lrf)
        momentum=0.937,  # SGD动量/Adam beta1
        weight_decay=0.0005,  # 优化器权重衰减
        warmup_epochs=3.0,  # 热身轮数
        warmup_momentum=0.8,  # 热身初始动量
        warmup_bias_lr=0.1,  # 热身初始偏置学习率

        # 数据增强
        hsv_h=0.10,  # 为适应弱光线或者穿着衣服颜色差异而增大 ori=0.015
        hsv_s=0.7,
        hsv_v=0.6,  # 为适应多光线强度而增大 ori=0.4
        fliplr=0.5,
        mosaic=0.5,  # 认为不需要进行小目标检测调小 ori=1.0

        # 损失权重
        box=7.5,  # 边框精确度
        cls=0.8,  # 分类损失权重 ori=0.5
        dfl=1.5,  # 边界框分布

        # 其他
        val=True,  # 训练期间验证
        plots=True,  # 保存训练曲线图
        resume=False  # 从最新检查点恢复训练
    )

    results = model.val()  # evaluate model performance on the validation set

if __name__ == '__main__':
    main()



# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
#results = model.train(data="coco128.yaml", epochs=3)  # train the model
#results = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#success = model.export(format="onnx")  # export the model to ONNX format