import sys
from PyQt5.QtWidgets import *
from MainWindow import Ui_MainWindow
from MyUi import MyWindow

if __name__ == "__main__":
    #QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyleSheet("""
   /* 全局样式 */
* {
    font-family: 'Microsoft YaHei';
}

/* 透明背景的特定标签样式 */
QLabel#text_1,QLabel#text_2,QLabel#text_3,QLabel#text_4,QLabel#text_5,QLabel#text_6,QLabel#text_7,QLabel#text_8,QLabel#PicBefore,QLabel#PicAfter{
    background-color: transparent;
}

QLabel#label_1,QLabel#label_2,QLabel#label_3,QLabel#label_4,QLabel#Label_H,QLabel#Label_T,QLabel#Label_W,QLabel#Label_Type{
    background-color: transparent;
}

QLabel#text_1,QLabel#text_2,QLabel#text_3,QLabel#text_4,QLabel#text_5,QLabel#text_6,QLabel#text_7,QLabel#text_8 {
    font-size: 18px; /* 可以根据需要调整字体大小 */
}


/* 窗口背景样式 */
QWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop: 0.1 #dceeff,   /* 雾蓝 */
                                stop: 0.3 #e6f1f8,   /* 浅蓝白 */
                                stop: 0.75 #e6f1f8,  /* 浅蓝白 */
                                stop: 0.9 #dceeff);  /* 雾蓝 */
}



/* 按钮样式 */
QPushButton {
    background-color: #f0f7fb;
    color: #333333;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #CCCCCC;
}

/* 按钮悬停样式 */
QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

/* 按钮按下样式 */
QPushButton:pressed {
    background-color: rgba(255, 255, 255, 0.4);
}

/* 按钮禁用样式 */
QPushButton:disabled {
    background-color: transparent;
    color: #A0A0A0;
    border: 1px solid #E0E0E0;
}


    """)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    My_Ui = MyWindow(ui)
    sys.exit(app.exec_())


