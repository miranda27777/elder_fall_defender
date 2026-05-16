import os
import sys
import ssl

# 禁用 SSL 验证（解决 ultralytics 联网检查的证书问题）
ssl._create_default_https_context = ssl._create_unverified_context

# 禁止 ultralytics 联网检查更新
os.environ['ULTRALYTICS_NO_CHECK_UPDATE'] = '1'
os.environ['YOLO_NO_AUTO_UPDATE'] = '1'
os.environ['YOLO_VERBOSE'] = 'False'

from PySide6 import QtWidgets, QtCore, QtGui
import cv2
from threading import Lock
from ultralytics import YOLO
import numpy as np


class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        
        # 视频相关属性
        self.cap = None
        self.is_camera = False
        self.frame_lock = Lock()
        self.current_frame = None

        # 获取当前脚本所在目录（src/）
        base_dir = os.path.dirname(__file__)

        # 向上回到项目根目录，然后找 model/best.pt
        self.model_path = os.path.join(base_dir, "..", "model", "best.pt")
        self.model_path = os.path.normpath(self.model_path)
        
        # 初始化模型
        self.init_model()
        
        # 定时器设置
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 状态变量
        self.running = False

    def init_model(self):
        """初始化YOLO模型"""
        self.textLog.append(f"⏳ 正在加载YOLO模型: {self.model_path}")
        try:
            if not os.path.exists(self.model_path):
                self.textLog.append(f"❌ 模型文件不存在: {self.model_path}")
                self.model = None
                return
                
            self.model = YOLO(self.model_path, verbose=False)
            self.textLog.append("✅ 模型加载成功")
            self.textLog.append(f"模型类别: {self.model.names}")
        except Exception as e:
            self.textLog.append(f"❌ 模型加载失败: {str(e)}")
            self.model = None

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('YOLOv8 视频/摄像头检测系统')

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 视频显示区域
        video_layout = QtWidgets.QHBoxLayout()
        self.label_ori = QtWidgets.QLabel()
        self.label_result = QtWidgets.QLabel()
        for label in [self.label_ori, self.label_result]:
            label.setMinimumSize(640, 480)
            label.setStyleSheet('border: 1px solid #D7E2F9; background: black;')
            video_layout.addWidget(label)
        main_layout.addLayout(video_layout)

        # 控制面板
        control_panel = QtWidgets.QHBoxLayout()
        
        btn_group = QtWidgets.QGroupBox("操作控制")
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_camera = QtWidgets.QPushButton('📷 摄像头')
        self.btn_video = QtWidgets.QPushButton('🎞️ 视频文件')
        self.btn_stop = QtWidgets.QPushButton('🛑 停止')
        
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop.clicked.connect(self.stop)
        
        for btn in [self.btn_camera, self.btn_video, self.btn_stop]:
            btn.setFixedHeight(40)
            btn_layout.addWidget(btn)
        
        btn_group.setLayout(btn_layout)
        control_panel.addWidget(btn_group)
        
        main_layout.addLayout(control_panel)

        # 日志区域
        self.textLog = QtWidgets.QTextBrowser()
        main_layout.addWidget(self.textLog)

    def start_camera(self):
        """启动摄像头检测"""
        if self.running:
            self.stop()
        
        self.textLog.append("📷 正在尝试打开摄像头...")
        
        # 尝试不同的后端
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, name in backends:
            for i in range(3):
                self.textLog.append(f"   尝试 {name} 后端，索引 {i}...")
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # 尝试读取一帧验证
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.textLog.append(f"✅ 成功打开摄像头 ({name}, 索引 {i})")
                        self.is_camera = True
                        self.running = True
                        self.timer.start(30)
                        self.textLog.append("📷 摄像头模式 - 开始实时检测...")
                        return
                    else:
                        cap.release()
                else:
                    cap.release()
        
        self.textLog.append("❌ 无法打开摄像头，请检查：")
        self.textLog.append("   1. 摄像头是否已连接")
        self.textLog.append("   2. 摄像头驱动是否正常")
        self.textLog.append("   3. Windows 隐私设置 > 相机 > 允许应用访问相机")
        self.textLog.append("   4. 尝试用其他软件（如相机应用）测试摄像头是否正常")

    def open_video(self):
        """打开视频文件检测"""
        if self.running:
            self.stop()
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
            
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.textLog.append(f"❌ 无法打开视频文件: {path}")
            return
            
        self.is_camera = False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        interval = max(10, int(1000 / fps)) if fps > 0 else 30
        
        self.running = True
        self.timer.start(interval)
        self.textLog.append(f"🎞️ 视频模式 - 开始检测: {os.path.basename(path)}")

    def update_frame(self):
        """更新视频帧并进行检测"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            if self.is_camera:
                self.textLog.append("⚠️ 摄像头读取失败")
            else:
                self.textLog.append("✅ 视频播放完毕")
                self.stop()
            return
        
        # 显示原始帧
        self.display_frame(frame, self.label_ori)
        
        # 使用模型检测
        if self.model:
            try:
                with self.frame_lock:
                    results = self.model.predict(frame, verbose=False)
                    
                    annotated_frame = frame.copy()
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            color = (0, 255, 0)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{self.model.names[class_id]} {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                            self.textLog.append(f"检测到: {self.model.names[class_id]} - 置信度: {conf:.2f}")
                    
                    self.display_frame(annotated_frame, self.label_result)
                    
            except Exception as e:
                self.textLog.append(f"❌ 检测出错: {str(e)}")

    def display_frame(self, frame, label):
        """显示帧到指定QLabel"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            label.width(), label.height(), QtCore.Qt.KeepAspectRatio
        ))

    def stop(self):
        """停止检测"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.timer.stop()
        self.running = False
        self.textLog.append("⏹️ 已停止检测")
        
        self.label_ori.clear()
        self.label_result.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MWindow()
    window.show()
    sys.exit(app.exec())