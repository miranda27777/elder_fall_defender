import os
import sys
import traceback #打印错误堆栈，调试异常时显示详细信息。
# 图像处理模块
import cv2
import numpy as np
from PIL import Image #Python Imaging Library，用于图像读取和保存（与 OpenCV 互补）
import matplotlib.pyplot as plt # 用于绘制图像直方图、可视化图像分析结果。
# PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, qRed, qGreen, qBlue
from PyQt5.QtCore import Qt
# 自定义模块
from ultralytics import YOLO


# 定义opencv图像转QImage图像的函数
def cvImgtoQtImg(cvImg):  
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    QtImg = QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QImage.Format_RGB32)
    return QtImg

# 定义QImage(能改像素)图像转opencv图像的函数
def QImage2CV(qimg):
    tmp = qimg

    # 使用numpy创建空的图象
    cv_image = np.zeros((tmp.height(), tmp.width(), 3), dtype=np.uint8)

    for row in range(0, tmp.height()):
        for col in range(0, tmp.width()):
            r = qRed(tmp.pixel(col, row))
            g = qGreen(tmp.pixel(col, row))
            b = qBlue(tmp.pixel(col, row))
            cv_image[row, col, 0] = r
            cv_image[row, col, 1] = g
            cv_image[row, col, 2] = b

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    return cv_image

# 定义QPixmap图像(不能改像素，但适合做展示)转opencv图像的函数
def QPixmap2cv(qtpixmap):
    try:
        qimg = qtpixmap.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    except Exception as e:
        traceback.print_exc()

    return result

# 傅里叶变换
def FFT2(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = 20 * np.log(np.abs(fshift))
    res = res - res.min()
    res = res / res.max() * 255
    res = np.array(res, np.uint8)

    plt.imshow(res)
    plt.axis('off')
    plt.savefig('Img.png', bbox_inches='tight', pad_inches=0.0)
    plt.close()
    result = cv2.imread('Img.png', 0)
    os.remove('Img.png')

    return result


#MyWindow类
class MyWindow(QMainWindow):
    def __init__(self, Ui_MainWindow):
        super().__init__()
        self.ui = Ui_MainWindow
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.ui.setupUi(MainWindow)
        self.picpath = ''
        self.openfile_name = ''
        self.pixmapBefore = QPixmap()
        self.pixmapAfter = QPixmap()

        self.ui.PicBefore.setScaledContents(False)
        self.ui.PicBefore.setAlignment(Qt.AlignCenter)
        self.ui.PicAfter.setScaledContents(False)
        self.ui.PicAfter.setAlignment(Qt.AlignCenter)

        self.ui.Import.clicked.connect(lambda: self.Import())
        self.ui.Grayscale.clicked.connect(lambda: self.Grayscale())
        self.ui.Binarization.clicked.connect(lambda: self.Binarization())
        self.ui.Histogram.clicked.connect(lambda: self.Histogram())
        self.ui.Equalize.clicked.connect(lambda: self.Equalize())
        self.ui.Mean.clicked.connect(lambda: self.Mean())
        self.ui.Median.clicked.connect(lambda: self.Median())
        self.ui.LowPass.clicked.connect(lambda: self.Lowpass())
        self.ui.HighPass.clicked.connect(lambda: self.Highpass())
        self.ui.Roberts.clicked.connect(lambda: self.Roberts())
        self.ui.Prewitt.clicked.connect(lambda: self.Prewitt())
        self.ui.Sobel.clicked.connect(lambda: self.Sobel())
        self.ui.Corrosion.clicked.connect(lambda: self.Corrosion())
        self.ui.Expansion.clicked.connect(lambda: self.Expansion())
        self.ui.Open.clicked.connect(lambda: self.Open())
        self.ui.Close.clicked.connect(lambda: self.Close())
        self.ui.LoG.clicked.connect(lambda: self.LOG())
        self.ui.Laplacian.clicked.connect(lambda: self.Laplacian())
        self.ui.Canny.clicked.connect(lambda: self.Canny())
        self.ui.Save.clicked.connect(lambda: self.Save())
        self.ui.pushButton_4.clicked.connect(self.DeepLearningProcess)

        MainWindow.show()
        sys.exit(app.exec_())


    def Import(self):
    # 导入图片
     self.openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', "Image Files (*.png *.jpg *.bmp)")[0]
     if not self.openfile_name:
        # 如果没有选择文件，直接返回
        return

     try:
        # Use QPixmap to load the image first
        self.pixmapBefore = QPixmap(self.openfile_name)
        if self.pixmapBefore.isNull():
            raise ValueError("无法加载图片文件")

        self.picpath = self.openfile_name

        # Convert the path to a format that OpenCV can handle
        # On Windows, use the native path with backslashes
        cv_path = self.convert_path(self.picpath)
        
        # Read the image using OpenCV with proper encoding handling
        image = cv2.imdecode(np.fromfile(cv_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图片文件")

        self.ui.Label_H.setText(str(image.shape[0]))
        self.ui.Label_W.setText(str(image.shape[1]))
        self.ui.Label_T.setText(str(image.shape[2]))
        self.ui.Label_Type.setText(str(image.dtype))

        self.resizeImage(self.ui.PicBefore, self.pixmapBefore)

     except Exception as e:
        traceback.print_exc()
        QMessageBox.critical(self, "错误", f"加载图片文件时出错: {e}")

    def Save(self):
        if self.pixmapAfter.isNull():
            QMessageBox.about(self, '保存失败', '没有已经处理完成的图片')
            return
        # 保存
        SaveName = QFileDialog.getSaveFileName(self, '选择文件', '', "Image Files (*.png *.jpg *.bmp)")[0]

        if not SaveName:
            return

        try:
            if type(self.pixmapAfter) == QImage:
                print("1")
                result = cv2.imwrite(SaveName, QImage2CV(self.pixmapAfter))
            else:
                print("2")
                result = cv2.imwrite(SaveName, QPixmap2cv(self.pixmapAfter))
            if result:
                QMessageBox.about(self, '保存成功', '保存成功')
            else:
                QMessageBox.about(self, '保存失败', '路径中不能含有中文和空格')
        except Exception as e:
            traceback.print_exc()


    def resizeImage(self, label, pixmap):
        if pixmap:
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.resizeImage(self.ui.PicBefore,self.pixmapBefore)
        self.resizeImage(self.ui.PicAfter,self.pixmapAfter)
        super(MyWindow, self).resizeEvent(event)

    def check(self):
        if self.pixmapBefore.isNull():
            QMessageBox.about(self, '操作失败', '请先导入图片')
            return True
        img = cv2.imread(self.picpath)
        if img is None:
            QMessageBox.about(self, '操作失败', '无法读取图片')
            return True
        return False

    def Grayscale(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        qt_img = cvImgtoQtImg(grayImg)
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter,self.pixmapAfter)

    def Binarization(self):
        if self.check():
            return
        try:
            img = cv2.imread(self.picpath)
            grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            qt_img = cvImgtoQtImg(cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)[1])
            self.pixmapAfter = QPixmap.fromImage(qt_img)
            self.resizeImage(self.ui.PicAfter, self.pixmapAfter)
        except Exception as e:
            traceback.print_exc()

    def Histogram(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)], [0], None, [256], [0, 255])
        plt.plot(img)
        plt.savefig('img.jpg')
        plt.close()
        self.pixmapAfter = QPixmap('img.jpg')
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)
        os.remove('img.jpg')

    def Equalize(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        qt_img = cvImgtoQtImg(cv2.equalizeHist(img))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Mean(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        qt_img = cvImgtoQtImg(cv2.blur(img, (3, 5)))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Median(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        qt_img = cvImgtoQtImg(cv2.medianBlur(img, 5))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Roberts(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernelx_Robert = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely_Robert = np.array([[0, -1], [1, 0]], dtype=int)
        x_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernelx_Robert)
        y_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernely_Robert)
        absX_Robert = cv2.convertScaleAbs(x_Robert)
        absY_Robert = cv2.convertScaleAbs(y_Robert)
        qt_img = cvImgtoQtImg(cv2.addWeighted(absX_Robert, 0.5, absY_Robert, 0.5, 0))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Prewitt(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernelx_Prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely_Prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x_Prewitt = cv2.filter2D(img_binary, -1, kernelx_Prewitt)
        y_Prewitt = cv2.filter2D(img_binary, -1, kernely_Prewitt)
        absX_Prewitt = cv2.convertScaleAbs(x_Prewitt)
        absY_Prewitt = cv2.convertScaleAbs(y_Prewitt)
        qt_img = cvImgtoQtImg(cv2.addWeighted(absX_Prewitt, 0.5, absY_Prewitt, 0.5, 0))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Sobel(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        x_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 1, 0)
        y_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 0, 1)
        absX_Sobel = cv2.convertScaleAbs(x_Sobel)
        absY_Sobel = cv2.convertScaleAbs(y_Sobel)
        qt_img = cvImgtoQtImg(cv2.addWeighted(absX_Sobel, 0.5, absY_Sobel, 0.5, 0))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Lowpass(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_dft = np.fft.fft2(img)
        dft_shift_low = np.fft.fftshift(img_dft)
        h, w = dft_shift_low.shape[0:2]
        h_center, w_center = int(h / 2), int(w / 2)
        img_black = np.zeros((h, w), np.uint8)
        img_black[h_center - int(100 / 2):h_center + int(100 / 2), w_center - int(100 / 2):w_center + int(100 / 2)] = 1
        dft_shift_low = dft_shift_low * img_black
        idft_shift = np.fft.ifftshift(dft_shift_low)
        ifimg = np.fft.ifft2(idft_shift)
        ifimg = np.abs(ifimg)
        ifimg = np.int8(ifimg)
        cv2.imwrite('img.jpg', ifimg)
        self.pixmapAfter = QPixmap('img.jpg')
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)
        os.remove('img.jpg')

    def Highpass(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(img_dft)
        h, w = dft_shift.shape[0:2]
        h_center, w_center = int(h / 2), int(w / 2)
        dft_shift[h_center - int(50 / 2):h_center + int(50 / 2),
        w_center - int(50 / 2):w_center + int(50 / 2)] = 0
        idft_shift = np.fft.ifftshift(dft_shift)
        img_idft = np.fft.ifft2(idft_shift)
        img_idft = np.abs(img_idft)
        img_idft = np.int8(img_idft)
        cv2.imwrite('img.jpg', img_idft)
        self.pixmapAfter = QPixmap('img.jpg')
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)
        os.remove('img.jpg')

    def Corrosion(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        qt_img = cvImgtoQtImg(cv2.erode(img_binary, kernel))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Expansion(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        qt_img = cvImgtoQtImg(cv2.dilate(img_binary, kernel, iterations=1))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Open(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        qt_img = cvImgtoQtImg(cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Close(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        qt_img = cvImgtoQtImg(cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def LOG(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1, 1)
        LOG_result = cv2.Laplacian(img_blur, cv2.CV_16S, ksize=1)
        qt_img = cvImgtoQtImg(cv2.convertScaleAbs(LOG_result))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def Laplacian(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        qt_img = cvImgtoQtImg(abs_laplacian)
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)


    def Canny(self):
        if self.check():
            return
        img = cv2.imread(self.picpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur_canny = cv2.GaussianBlur(img_gray, (7, 7), 1, 1)
        qt_img = cvImgtoQtImg(cv2.Canny(img_blur_canny, 50, 150))
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)

    def DeepLearningProcess(self):
     try:
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(__file__)

        # 拼接相对路径
        model_path = os.path.join(base_dir, "runs", "train", "fallDetect1", "weights", "best.pt")

        # 加载模型
        model = YOLO(model_path)
        results = model(self.picpath)
        
        # 获取类别名称（通过model.names而不是results.names）
        class_names = model.names
        
        # 加载原始图像
        img = cv2.imread(self.picpath)
        
        # 绘制检测结果
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            label = f"{class_names[cls_id]} {conf:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        qt_img = cvImgtoQtImg(img)
        self.pixmapAfter = QPixmap.fromImage(qt_img)
        self.resizeImage(self.ui.PicAfter, self.pixmapAfter)
        
        QMessageBox.information(self, "检测结果",
                              f"检测到: {class_names[cls_id]}\n置信度: {conf:.2f}")
    
     except Exception as e:
        QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
        traceback.print_exc()




    def convert_path(self,path):
        return path.replace("/", "\\")