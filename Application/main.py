from BackgorunChanger import Camera, Displayer, VideoDataset, cv2_frame_to_cuda
from CaptureBackground import capture_background
import json
import cv2
from os import path
import torch
from torch import nn
from torchvision.transforms import ToTensor
from model import MattingBase, MattingRefine

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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


width, height = map(int, config['resolution'].split())
cam = Camera(width=width, height=height)
dsp = None

if config['rotated']:
    # Rotated
    dsp = Displayer('MattingV2', cam.height, cam.width, show_info=False)
else:
    # Normal
    dsp = Displayer('MattingV2', cam.width, cam.height, show_info=False)


tb_video = VideoDataset(config['target_video'], transforms=ToTensor(), rotated=config['rotated'])
frame_num = 0
bgr = None

if not path.exists(config["video_bgr"]):
    bgr = capture_background(cam, config["video_bgr"])
    exit()
else:
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

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            frame = cam.read()
            if frame:
                if config['rotated']:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                src = cv2_frame_to_cuda(frame)
                pha, fgr = model(src, bgr)[:2]

                vidframe = tb_video[frame_num].unsqueeze_(0).cuda()
                tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
                frame_num += 1
                if frame_num >= tb_video.__len__():
                    frame_num = 0

                # res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
                res = pha * fgr + (1 - pha) * tgt_bgr
                res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(res, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(720, 1280, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

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
