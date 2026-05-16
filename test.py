import os
import cv2
import numpy as np
from enum import IntEnum
from ultralytics import YOLO

# 定义行为类别枚举
class ActionClass(IntEnum):
    FALL = 0  # 跌倒
    SIT = 1  # 坐姿
    STAND = 2  # 站姿
    LIE = 3  # 躺卧


def get_class_name(class_id: int) -> str:
    """
    根据类别ID获取对应的行为类别名称
    参数: class_id (int): 类别ID
    返回: str: 对应的类别名称，如果ID无效则返回"Unknown"
    """
    try:
        return ActionClass(class_id).name.lower()
    except ValueError:
        return "Unknown"

def image_detect(model, path, save_dir=None):
    """
    使用YOLO模型检测路径下的图片或者视频
    参数:
        model: 已加载的YOLO模型
        path (str): 检测路径，可以是文件或目录（不含递归）
        save_dir (str, optional): 指定保存路径；默认为None，保存在runs/detect
    """
    supported_extensions = ('.png', '.jpg', '.jpeg',
                            '.gif', '.mp4', '.avi', '.mov')

    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return

    print("\n开始检测...")

    # 处理保存路径
    yolo_args = {'save': True}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # 创建保存路径
        yolo_args['project'] = save_dir

    # 如果path是文件
    if os.path.isfile(path):
        if path.lower().endswith(supported_extensions):
            print(f"正在检测文件: {path}")
            try:
                # 使用传入的model进行检测
                model(source=path, **yolo_args)
                save_location = save_dir if save_dir else os.path.join("runs", "detect")
                print(f"检测完成，结果已保存到: {save_location}")
            except Exception as e:
                print(f"检测失败: {e}")
        else:
            print(f"不支持的文件格式: {path}")

    # 如果path是目录
    elif os.path.isdir(path):
        # 获取目录下所有支持的媒体文件
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f)) and
                 f.lower().endswith(supported_extensions)]

        if not files:
            print(f"目录中没有支持的图片或视频文件: {path}")
            return

        print(f"在目录中发现 {len(files)} 个可检测文件")

        for file_path in files:
            print(f"\n正在检测: {file_path}")
            try:
                model(source=file_path, **yolo_args)
                save_location = save_dir if save_dir else os.path.join("runs", "detect")
                print(f"检测完成，结果已保存到: {save_location}")
            except Exception as e:
                print(f"检测失败: {e}")

def print_details(result):
    """
    打印检测结果的详细信息
    """
    boxes = result.boxes
    print("\n=== 检测详情 ===")
    for i, box in enumerate(boxes, 1):
        #class_id = int(box.cls[0].item())
        #class_name = get_class_name(class_id)
        confidence = round(box.conf[0].item(), 2)
        xyxy = np.round(box.xyxy[0].tolist(), 2)  # 像素坐标 [x1,y1,x2,y2]
        xywh = np.round(box.xywh[0].tolist(), 2)  # 中心点坐标 [x_center,y_center,width,height]
        xyxyn = np.round(box.xyxyn[0].tolist(), 4)  # 归一化坐标 [0-1]

        print(
            #f"对象 {i}: {class_name}({class_id})\n"
            f"  置信度: {confidence:.2f}\n"
            f"  像素坐标 (xyxy): {xyxy}\n"
            f"  中心坐标 (xywh): {xywh}\n"
            f"  归一化坐标 (xyxyn): {xyxyn}"
        )

def print_detection_warnings(result):
    """
    子函数：根据检测到的类别打印警告信息
    """
    boxes = result.boxes
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id == 0:  # 假设0是'fall'类别
            print("警告！检测到跌倒！")
            # 可以添加声音报警或其他警示方式


def real_time_detection(model, show_details=False):
    """
    实时视频检测函数

    参数:
        model: 加载好的YOLO模型
        show_details: 是否显示检测详细信息
    """
    print("\n正在尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return
    else:
        print("摄像头已成功开启！")
        print("实时检测中 (按Q键退出)...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("警告：摄像头帧读取失败！")
            break

        results = model.predict(frame, stream=True, verbose=False)

        for result in results:
            # 打印警告信息
            print_detection_warnings(result)

            # 根据参数决定是否打印详细信息
            if show_details:
                print_details(result)

            # 手动绘制检测框,因为默认的show=True窗口有问题，打不开
                # 绘制所有检测框
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)  # BGR格式
                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                # 绘制标签背景
                class_id = int(box.cls[0].item())
                class_name = get_class_name(class_id)
                # 绘制文本 字体：FONT_HERSHEY_SIMPLEX, 字体大小：1.0 颜色：(B,G,R), 粗细：2
                cv2.putText(frame, class_name, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # 显示画面
            cv2.imshow("Detection", frame)

            # 检测按键
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

def main():
    # 初始化模型
    print("正在加载YOLO模型...")
    try:
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(__file__)

        # 构建相对路径
        yolo = os.path.join(base_dir, "runs", "train", "fallDetect1", "weights", "best.pt")
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # 应用示例
    image_detect(yolo,'tests/fallDownTest/1.jpg','runs/detect/predict2')
    # image_detect(yolo,'tests/fallDownTest/')
    # real_time_detection(yolo)

if __name__ == "__main__":
    main()