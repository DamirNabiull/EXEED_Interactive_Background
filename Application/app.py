import time

from BackgroundChanger import Camera, Displayer, VideoDataset, cv2_frame_to_cuda
from CaptureBackground import capture_background
import json
import cv2
from os import path
import torch
from torch import nn
from torchvision.transforms import ToTensor
from model import MattingBase, MattingRefine

import sys
import uuid
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from design import Menu, End, Instruction, Video
from ctypes import windll

widget: QStackedWidget
cam: Camera
countdown_time: int
record_time: int
config: dict
model = None
tb_video: VideoDataset
bgr = None
current_id: str
WIDTH: int
HEIGHT: int


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
        self.instructionButton.clicked.connect(self.start_video)

    def start_video(self):
        global widget, current_id
        current_id = str(uuid.uuid4())
        print(current_id)
        widget.setCurrentIndex(2)


class VideoWin(QMainWindow, Video.Ui_VideoMainWindow):
    def __init__(self, parent=None):
        global start_timer
        super(VideoWin, self).__init__(parent)
        self.setupUi(self)

        self.videoUpdater = VideoWorker()
        self.videoUpdater.start()
        self.videoUpdater.update_image.connect(self.update_image)
        self.videoUpdater.start_timer.connect(self.update_gui)
        self.videoUpdater.start_prepare.connect(self.start_prepare)
        self.videoUpdater.start_record.connect(self.start_recording)

    def update_image(self, Image):
        self.videoLabel.setPixmap(QPixmap.fromImage(Image))

    def update_gui(self, complete_time, past_time):
        left_time = complete_time - past_time
        self.secondsLabel.setText(f'{left_time} sec')

    def start_prepare(self):
        self.comandLabel.setText('Приготовься...')
        self.prepareLabel.show()

    def start_recording(self):
        self.comandLabel.setText('Идет запись')
        self.prepareLabel.hide()


class VideoWorker(QThread):
    update_image = pyqtSignal(QImage)
    start_timer = pyqtSignal(int, int)
    start_prepare = pyqtSignal()
    start_record = pyqtSignal()
    height: int
    width: int
    frame_num: int
    rotated: bool
    frame_array: list

    def run(self):
        global WIDTH, HEIGHT, config, countdown_time, record_time, current_id
        self.rotated = config['rotated']
        self.height = HEIGHT
        self.width = WIDTH
        self.frame_array = []
        self.frame_num = 0
        if self.rotated:
            self.height = WIDTH
            self.width = HEIGHT

        while True:
            if widget.currentIndex() == 2:
                self.frame_array = []
                self.frame_num = 0

                self.start_prepare.emit()
                prev = time.time()
                delta = time.time() - prev

                while delta < countdown_time:
                    frame = self.get_frame()
                    self.update_pic(frame)
                    self.start_timer.emit(countdown_time, int(delta))
                    delta = time.time() - prev

                self.start_record.emit()
                prev = time.time()
                delta = time.time() - prev
                while delta < record_time:
                    frame = self.get_frame()
                    res = self.add_background(frame)
                    self.update_pic(res)
                    self.frame_array.append(res)
                    self.start_timer.emit(record_time, int(delta))
                    delta = time.time() - prev

                self.save_video()
                go_next_screen()

    def update_pic(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(1080, 1920, Qt.KeepAspectRatio)
        self.update_image.emit(Pic)

    def get_frame(self):
        frame = cam.read()
        if self.rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def add_background(self, frame):
        global model, tb_video, bgr

        src = cv2_frame_to_cuda(frame)
        pha, fgr = model(src, bgr)[:2]

        vidframe = tb_video[self.frame_num].unsqueeze_(0).cuda()
        tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
        self.frame_num += 1
        if self.frame_num >= tb_video.__len__():
            self.frame_num = 0

        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        return res

    def save_video(self):
        global current_id

        print('Saving')
        name = fr'user_video\{current_id}.mp4'
        print(name)
        print('Started saving')
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'),
                              len(self.frame_array) // config['record_time'],
                              (self.width, self.height))

        for i in range(len(self.frame_array)):
            out.write(self.frame_array[i])

        out.release()
        print('Out is released')


class EndWin(QMainWindow, End.Ui_EndMainWindow):
    def __init__(self, parent=None):
        super(EndWin, self).__init__(parent)
        self.setupUi(self)
        self.endButton.clicked.connect(go_to_menu)


def start(width=1080, height=1920, cam_=None, bgr_=None, model_=None, tb_video_=None, config_=None, c_t=10, r_t=15):
    global widget, cam, countdown_time, record_time, config, model, tb_video, bgr, WIDTH, HEIGHT

    WIDTH = width
    HEIGHT = height
    cam = cam_
    countdown_time = c_t
    record_time = r_t
    config = config_
    model = model_
    tb_video = tb_video_
    bgr = bgr_

    app = QApplication(sys.argv)

    QFontDatabase.addApplicationFont('design/fonts/TacticSans-Bld.otf')
    stylesheet = open('design/style.qss').read()
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
    app.exec_()


if __name__ == '__main__':
    h = windll.user32.FindWindowA(b'Shell_TrayWnd', None)
    # hide the taskbar
    windll.user32.ShowWindow(h, 9)
    start()
