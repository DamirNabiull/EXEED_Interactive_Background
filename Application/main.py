from BackgroundChanger import Camera, Displayer, VideoDataset, cv2_frame_to_cuda
from CaptureBackground import capture_background
import json
import cv2
from os import path
import torch
from torchvision.transforms import ToTensor
from model import MattingBase, MattingRefine

import app

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

tb_video = VideoDataset(config['target_video'], transforms=ToTensor(), rotated=config['rotated'])
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


if __name__ == "__main__":
    app.start(WIDTH, HEIGHT, cam, bgr, model, tb_video, config, config['countdown_time'], config['record_time'])
