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

import threading

# --------------- Main ---------------

config = json.load(open('config.json'))

# Load model
model = None
if config['model_type'] == 'mattingbase':
    model = MattingBase(config['model_backbone'])
if config['model_type'] == 'mattingrefine':
    model = MattingRefine(
        config['model_backbone'],
        config['model_backbone_scale'],
        config['model_refine_mode'],
        config['model_refine_sample_pixels'],
        config['model_refine_threshold'])

model = model.cuda().eval()
model.load_state_dict(torch.load(config['model_checkpoint']), strict=False)


WIDTH, HEIGHT = map(int, config['resolution'].split())
cam = Camera(width=WIDTH, height=HEIGHT)
dsp = None

# if config['rotated']:
#     # Rotated
#     dsp = Displayer('MattingV2', cam.height, cam.width, show_info=False)
# else:
#     # Normal
#     dsp = Displayer('MattingV2', cam.width, cam.height, show_info=False)


tb_video = VideoDataset(config['target_video'], transforms=ToTensor(), rotated=config['rotated'])
# tb_video.preprocessing()
frame_num = 0
bgr = None

if not path.exists(config["video_bgr"]):
    print('Not exist')
    bgr = capture_background(cam, config["video_bgr"])
    exit()
else:
    print('Exist')
    bgr = cv2.imread(config['video_bgr'], 0)
    if config['rotated']:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    bgr = cv2_frame_to_cuda(bgr)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CaptureBTN = QPushButton("Capture")
        self.CaptureBTN.clicked.connect(self.CaptureFeed)
        self.VBL.addWidget(self.CaptureBTN)

        self.SaveBTN = QPushButton("Save")
        self.SaveBTN.clicked.connect(self.SaveFeed)
        self.VBL.addWidget(self.SaveBTN)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CaptureFeed(self):
        self.Worker1.set_capture()

    def SaveFeed(self):
        self.Worker1.save_video()

    def CancelFeed(self):
        self.Worker1.stop()
        exit()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    frame_array: list
    ThreadActive: bool
    frame_num: int
    is_capture: bool
    height: int
    width: int

    def run(self):
        self.frame_num = 0
        self.ThreadActive = True
        self.frame_array = []
        self.is_capture = False
        if config['rotated']:
            self.height = WIDTH
            self.width = HEIGHT
        else:
            self.height = HEIGHT
            self.width = WIDTH

        while self.ThreadActive:
            frame = self.get_frame()
            self.update_pic(frame)

            if self.is_capture:
                prev = time.time()
                while time.time() - prev < 15:
                    frame = self.get_frame()
                    print('*')
                    res = self.__add_background(frame)
                    self.update_pic(res)
                    self.frame_array.append(res)

                time.sleep(5)

                self.save_video()

    def get_frame(self):
        frame = cam.read()
        if config['rotated']:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def update_pic(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(self.width // 2, self.height // 2, Qt.KeepAspectRatio)
        self.ImageUpdate.emit(Pic)

    def set_capture(self):
        self.frame_num = 0
        self.is_capture = True
        print('Start capture')

    def __add_background(self, frame):
        print(r'--')
        src = cv2_frame_to_cuda(frame)
        pha, fgr = model(src, bgr)[:2]

        print(r'++')
        vidframe = tb_video[self.frame_num].unsqueeze_(0).cuda()
        print(r'..')
        print(self.frame_num)
        tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
        print(r'][')
        self.frame_num += 1
        if self.frame_num >= tb_video.__len__():
            self.frame_num = 0

        print(r'()')
        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        print(r'\/')
        return res

    def save_video(self):
        print('Saving')
        id = str(uuid.uuid4())
        name = f'{id}.mp4'
        print(len(self.frame_array))
        print('Started saving')

        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'),
                              len(self.frame_array) // config['record_time'], (self.width, self.height))

        for i in range(len(self.frame_array)):
            out.write(self.frame_array[i])

        out.release()
        print('Out is released')
        self.frame_array = []
        self.frame_num = 0
        self.is_capture = False

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
# with torch.no_grad():
#     while True:
#         # Normal
#         frame = cam.read()
#
#         # Rotated
#         if config['rotated']:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#
#         src = cv2_frame_to_cuda(frame)
#         pha, fgr = model(src, bgr)[:2]
#
#         vidframe = tb_video[frame_num].unsqueeze_(0).cuda()
#         tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
#         frame_num += 1
#         if frame_num >= tb_video.__len__():
#             frame_num = 0
#
#         # res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
#         res = pha * fgr + (1 - pha) * tgt_bgr
#         res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
#         res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
#         key = dsp.step(res)
#         if key == ord('q'):
#             exit()
