import time

from BackgroundChanger import Camera, VideoDataset, cv2_frame_to_cuda
import cv2
from torch import nn

import sys
import uuid
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from design import Menu, End, Instruction, Video, Email
from ctypes import windll
import vlc
import moviepy.editor as mpe
import smtplib as smtp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yadisk
import re

widget: QStackedWidget
cam: Camera
countdown_time: int
record_time: int
config: dict
model = None
tb_video: VideoDataset
bgr = None
current_id: str
video_path: str
final_video: str
video_url: str
email: str
smtp_pass: str
token: str
WIDTH: int
HEIGHT: int
to_send_email: str
to_send_email = ''
MENU_INDEX = 0
VIDEO_INDEX = 2
PLAYER_INDEX = 3
PIC_WIDTH = 1080
PIC_HEIGHT = 1920
audio_play_path = r'file:///sound\sound.mp3'
audio_final_path = r'sound\sound.mp3'
sound_player = vlc.MediaPlayer(audio_play_path)
audio_background = mpe.AudioFileClip(audio_final_path)
y_disk: yadisk.YaDisk


def go_next_screen():
    global widget
    widget.setCurrentIndex(widget.currentIndex() + 1)


def go_to_menu():
    global widget
    widget.setCurrentIndex(MENU_INDEX)


def go_record():
    global widget
    widget.setCurrentIndex(VIDEO_INDEX)


def prepare_video():
    global video_path, final_video, audio_background
    my_clip = mpe.VideoFileClip(video_path)
    final_clip = my_clip.set_audio(audio_background)
    final_video = fr'final\{current_id}.mp4'
    final_clip.write_videofile(final_video, codec='mpeg4', audio_codec='libvorbis')
    upload_video()


def upload_video():
    global video_url, y_disk
    dir = config['disk_dir']
    dest_path = f'/{dir}/{current_id}.mp4'
    y_disk.upload(final_video, dest_path)
    video_url = y_disk.get_download_link(dest_path)
    print(video_url)


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
        widget.setCurrentIndex(VIDEO_INDEX)


class VideoWorker(QThread):
    update_image = pyqtSignal(QImage)
    close_window = pyqtSignal()
    start_timer = pyqtSignal(int, int)
    start_prepare = pyqtSignal()
    start_record = pyqtSignal()
    start_player = pyqtSignal()
    enable_play_butt = pyqtSignal()
    disable_play_butt = pyqtSignal()
    prepare_video_show = pyqtSignal()
    stop_recording_sig = pyqtSignal()
    height: int
    width: int
    frame_num: int
    rotated: bool
    frame_array: list

    stop_rec: bool
    stoped: bool
    in_player: bool
    is_play: bool
    video_uploaded: bool
    cap: cv2.VideoCapture
    fps: int
    sec_to_frame: float

    def run(self):
        global WIDTH, HEIGHT, config, countdown_time, record_time, current_id, sound_player
        self.rotated = config['rotated']
        self.height = HEIGHT
        self.width = WIDTH
        self.frame_array = []
        self.frame_num = 0
        self.video_uploaded = False
        self.stoped = False
        self.in_player = False
        self.is_play = False
        self.stop_rec = False
        self.fps = 0
        if self.rotated:
            self.height = WIDTH
            self.width = HEIGHT

        while True:
            if self.stop_rec and widget.currentIndex() == VIDEO_INDEX:
                prev = time.time()
                while time.time() - prev < 1:
                    pass
                self.stop_rec = False
                self.stoped = True
                self.stop_recording_sig.emit()

            if widget.currentIndex() == VIDEO_INDEX and (not self.in_player) and (not self.stoped):
                self.video_uploaded = False
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
                sound_player.play()
                while delta < record_time:
                    frame = self.get_frame()
                    res = self.add_background(frame)
                    self.update_pic(res)
                    self.frame_array.append(res)
                    self.start_timer.emit(record_time, int(delta))
                    delta = time.time() - prev

                sound_player.stop()
                self.save_video()
                # go_next_screen()
                self.in_player = True
                self.is_play = True

            if self.in_player:
                if not self.video_uploaded:
                    self.video_uploaded = True
                    self.cap = cv2.VideoCapture(video_path)
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.sec_to_frame = 1 / self.fps
                    print(video_path)
                    print(self.fps)

                self.start_player.emit()
                self.disable_play_butt.emit()

                if self.is_play:
                    self.is_play = False
                    ended = False
                    prev = time.time()
                    sound_player.play()
                    while self.cap.isOpened():
                        if widget.currentIndex() != VIDEO_INDEX:
                            self.unset_player()
                            break
                        if ended and (not self.is_play):
                            self.enable_play_butt.emit()
                            continue

                        delta = time.time() - prev

                        if delta < self.sec_to_frame:
                            continue

                        ret, frame = self.cap.read()
                        ended = False
                        prev = time.time()
                        if self.is_play:
                            self.is_play = False
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        if ret:
                            self.update_pic(frame)
                        else:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ended = True

    def set_play(self):
        self.disable_play_butt.emit()
        sound_player.stop()
        sound_player.play()
        self.is_play = True
        print('Set Play')

    def unset_player(self):
        sound_player.stop()
        print('Unset player')
        self.cap.release()
        self.is_play = False
        self.in_player = False

    def stop_recording(self):
        self.prepare_video_show.emit()
        self.stop_rec = True
        self.unset_player()

    def update_pic(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(PIC_WIDTH, PIC_HEIGHT, Qt.KeepAspectRatio)
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
        global current_id, video_path

        print('Saving')
        video_path = fr'user_video\{current_id}.mp4'
        print(video_path)
        print('Started saving')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                              len(self.frame_array) // config['record_time'],
                              (self.width, self.height))

        for i in range(len(self.frame_array)):
            out.write(self.frame_array[i])

        out.release()
        print(len(self.frame_array))
        print('Out is released')


class VideoWin(QMainWindow, Video.Ui_VideoMainWindow):
    def __init__(self, parent=None):
        super(VideoWin, self).__init__(parent)
        self.setupUi(self)
        self.hide_player()

        self.videoUpdater = VideoWorker()
        self.videoUpdater.start()
        self.videoUpdater.update_image.connect(self.update_image)
        self.videoUpdater.start_timer.connect(self.update_gui)
        self.videoUpdater.start_prepare.connect(self.start_prepare)
        self.videoUpdater.start_record.connect(self.start_recording)
        self.videoUpdater.start_player.connect(self.start_player)
        self.videoUpdater.stop_recording_sig.connect(self.stop_recording)

        self.videoUpdater.prepare_video_show.connect(self.prepare_video_show)

        self.videoUpdater.enable_play_butt.connect(self.enable_play_butt)
        self.videoUpdater.disable_play_butt.connect(self.disable_play_butt)

        self.recordButton.clicked.connect(self.videoUpdater.unset_player)
        self.playButton.clicked.connect(self.videoUpdater.set_play)
        self.sendButton.clicked.connect(self.videoUpdater.stop_recording)

    def stop_recording(self):
        go_next_screen()
        print('Prepare')
        prepare_video()
        print('Prepare end')
        self.videoUpdater.stoped = False

    def prepare_video_show(self):
        self.videoPrepareLabel.show()

    def disable_play_butt(self):
        self.playButton.setDisabled(True)

    def enable_play_butt(self):
        self.playButton.setDisabled(False)

    def hide_record(self):
        self.prepareLabel.hide()
        self.footerLabel.hide()
        self.comandLabel.hide()
        self.secondsLabel.hide()

    def hide_player(self):
        self.videoPrepareLabel.hide()
        self.playerFooterLabel.hide()
        self.playButton.hide()
        self.recordButton.hide()
        self.sendButton.hide()

    def show_record(self):
        self.prepareLabel.show()
        self.footerLabel.show()
        self.comandLabel.show()
        self.secondsLabel.show()

    def show_player(self):
        self.playerFooterLabel.show()
        self.playButton.show()
        self.recordButton.show()
        self.sendButton.show()

    def start_player(self):
        self.hide_record()
        self.show_player()

    def update_image(self, Image):
        self.videoLabel.setPixmap(QPixmap.fromImage(Image))

    def update_gui(self, complete_time, past_time):
        left_time = complete_time - past_time
        self.secondsLabel.setText(f'{left_time} sec')

    def start_prepare(self):
        self.hide_player()
        self.show_record()
        self.comandLabel.setText('Приготовься...')
        self.prepareLabel.show()

    def start_recording(self):
        self.comandLabel.setText('Идет запись')
        self.prepareLabel.hide()

    def closeEvent(self, event):
        self.videoUpdater.quit()
        exit()


class EmailWin(QMainWindow, Email.Ui_emailMainWindow):
    shift: bool

    def __init__(self, parent=None):
        super(EmailWin, self).__init__(parent)
        self.shift = False
        self.setupUi(self)

        self.pushButton.clicked.connect(self.button_press)
        self.pushButton_2.clicked.connect(self.button_press)
        self.pushButton_3.clicked.connect(self.button_press)
        self.pushButton_4.clicked.connect(self.button_press)
        self.pushButton_5.clicked.connect(self.button_press)
        self.pushButton_6.clicked.connect(self.button_press)
        self.pushButton_7.clicked.connect(self.button_press)
        self.pushButton_8.clicked.connect(self.button_press)
        self.pushButton_9.clicked.connect(self.button_press)
        self.pushButton_10.clicked.connect(self.button_press)
        self.pushButton_11.clicked.connect(self.button_press)
        self.pushButton_12.clicked.connect(self.button_press)
        self.pushButton_13.clicked.connect(self.button_press)
        self.pushButton_14.clicked.connect(self.button_press)
        self.pushButton_15.clicked.connect(self.button_press)
        self.pushButton_16.clicked.connect(self.button_press)
        self.pushButton_17.clicked.connect(self.button_press)
        self.pushButton_18.clicked.connect(self.button_press)
        self.pushButton_19.clicked.connect(self.button_press)
        self.pushButton_20.clicked.connect(self.button_press)
        self.pushButton_21.clicked.connect(self.button_press)
        self.pushButton_22.clicked.connect(self.button_press)
        self.pushButton_23.clicked.connect(self.button_press)
        self.pushButton_24.clicked.connect(self.button_press)
        self.pushButton_25.clicked.connect(self.button_press)
        self.pushButton_26.clicked.connect(self.button_press)
        self.pushButton_27.clicked.connect(self.button_press)
        self.pushButton_28.clicked.connect(self.button_press)
        self.pushButton_29.clicked.connect(self.button_press)
        self.pushButton_30.clicked.connect(self.button_press)
        self.pushButton_31.clicked.connect(self.button_press)
        self.pushButton_32.clicked.connect(self.button_press)
        self.pushButton_33.clicked.connect(self.button_press)
        self.pushButton_34.clicked.connect(self.button_press)
        self.pushButton_35.clicked.connect(self.button_press)
        self.pushButton_36.clicked.connect(self.button_press)
        self.pushButton_37.clicked.connect(self.button_press)
        self.pushButton_38.clicked.connect(self.button_press)
        self.pushButton_39.clicked.connect(self.button_press)
        self.pushButton_40.clicked.connect(self.button_press)

        self.backspaceButton.clicked.connect(self.backspace_press)
        self.shiftButton.clicked.connect(self.shift_press)

        self.sendEmailButton.clicked.connect(self.send_press)

    def shift_press(self):
        self.shift = not self.shift

    def backspace_press(self):
        global to_send_email
        if len(to_send_email) > 0:
            to_send_email = to_send_email[:-1]
        self.label.setText(to_send_email)

    def button_press(self):
        global to_send_email
        sending_button = self.sender()
        char = sending_button.text()
        if self.shift and char.isalpha():
            to_send_email += char.upper()
        else:
            to_send_email += char
        self.label.setText(to_send_email)

    def send_press(self):
        global to_send_email, video_url, email
        print('Send started')
        self.label.setText('')
        print(email)
        if not re.match(r"[^@]+@[^@]+\.[^@]+", to_send_email):
            pass
        else:
            server = smtp.SMTP_SSL('smtp.yandex.com', 465)
            server.set_debuglevel(1)
            server.ehlo(email)
            server.login(email, smtp_pass)
            server.auth_plain()
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "Exeed Video"
            msg['From'] = email
            msg['To'] = to_send_email
            part1 = MIMEText(video_url, 'plain')
            msg.attach(part1)

            server.sendmail(email, to_send_email, msg.as_string())
            server.quit()
        to_send_email = ''
        print('Send ended')
        go_next_screen()


class EndWin(QMainWindow, End.Ui_EndMainWindow):
    def __init__(self, parent=None):
        super(EndWin, self).__init__(parent)
        self.setupUi(self)
        self.endButton.clicked.connect(go_to_menu)


def start(width=1080, height=1920, cam_=None, bgr_=None, model_=None, tb_video_=None, config_=None, c_t=10, r_t=15,
          smtp_pass_='', token_=''):
    global widget, cam, countdown_time, record_time, config, model, tb_video, bgr, WIDTH, HEIGHT, email, smtp_pass, \
        token, y_disk

    WIDTH = width
    HEIGHT = height
    cam = cam_
    countdown_time = c_t
    record_time = r_t
    config = config_
    model = model_
    tb_video = tb_video_
    bgr = bgr_
    email = config['login']
    print(email)
    smtp_pass = smtp_pass_
    token = token_

    y_disk = yadisk.YaDisk(token=token)
    print(y_disk.check_token())

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(exit)

    QFontDatabase.addApplicationFont('design/fonts/TacticSans-Reg.otf')
    QFontDatabase.addApplicationFont('design/fonts/TacticSans-Bld.otf')
    stylesheet = open('design/style.qss').read()
    app.setStyleSheet(stylesheet)

    widget = QStackedWidget()

    menuWin = MenuWin()
    instructionWin = InstructionWin()
    videoWin = VideoWin()
    emailWin = EmailWin()
    endWin = EndWin()

    widget.addWidget(menuWin)
    widget.addWidget(instructionWin)
    widget.addWidget(videoWin)
    widget.addWidget(emailWin)
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
