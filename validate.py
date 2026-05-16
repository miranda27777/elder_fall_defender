from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO("runs/train/fallDetect1/weights/best.pt")
    results = model.val(
        data='./fallDownPic/mydata.yaml'
    )

if __name__ == '__main__':
    main()



