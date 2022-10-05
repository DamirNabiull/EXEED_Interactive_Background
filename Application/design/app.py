from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import cv2

import Menu
import End
import Instruction
import Video

from ctypes import windll

widget: QStackedWidget


def go_next_screen():
    global widget
    widget.setCurrentIndex(widget.currentIndex() + 1)


def go_to_menu():
    global widget
    widget.setCurrentIndex(0)


class MenuWin(QMainWindow, Menu.Ui_MenuMainWindow):
    def __init__(self, parent=None):
        super(MenuWin, self).__init__(parent)
        self.setupUi(self)
        self.menuStartButton.clicked.connect(go_next_screen)


class InstructionWin(QMainWindow, Instruction.Ui_InstructionMainWindow):
    def __init__(self, parent=None):
        super(InstructionWin, self).__init__(parent)
        self.setupUi(self)
        self.instructionButton.clicked.connect(go_next_screen)


class VideoWin(QMainWindow, Video.Ui_VideoMainWindow):
    def __init__(self, parent=None):
        super(VideoWin, self).__init__(parent)
        self.setupUi(self)
        self.videoUpdater = VideoWorker()
        self.videoUpdater.start()
        self.videoUpdater.ImageUpdate.connect(self.ImageUpdateSlot)

    def ImageUpdateSlot(self, Image):
        self.videoLabel.setPixmap(QPixmap.fromImage(Image))


class VideoWorker(QThread):
    ImageUpdate = pyqtSignal(QImage)
    height: int
    width: int

    def run(self):
        cap = cv2.VideoCapture('../video/background.mp4')
        success = True

        self.height = 1920
        self.width = 1080

        while success:
            success, frame = cap.read()
            self.update_pic(frame)

    def update_pic(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.ImageUpdate.emit(Pic)


class EndWin(QMainWindow, End.Ui_EndMainWindow):
    def __init__(self, parent=None):
        super(EndWin, self).__init__(parent)
        self.setupUi(self)
        self.endButton.clicked.connect(go_to_menu)


def main():
    global widget
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont('fonts/TacticSans-Bld.otf')
    stylesheet = open('style.qss').read()
    app.setStyleSheet(stylesheet)
    widget = QStackedWidget()

    menuWin = MenuWin()
    instructionWin = InstructionWin()
    videoWin = VideoWin()
    endWin = EndWin()

    widget.addWidget(menuWin)
    widget.addWidget(instructionWin)
    widget.addWidget(videoWin)
    widget.addWidget(endWin)

    widget.setWindowFlag(Qt.FramelessWindowHint)
    widget.showFullScreen()
    widget.show()
    print(type(widget))
    app.exec_()


if __name__ == '__main__':
    h = windll.user32.FindWindowA(b'Shell_TrayWnd', None)
    # hide the taskbar
    windll.user32.ShowWindow(h, 9)
    main()
