from BackgorunChanger import Camera, Displayer, VideoDataset, cv2_frame_to_cuda
from CaptureBackground import capture_background
import json
import cv2
from os import path
import torch
from torch import nn
from torchvision.transforms import ToTensor
from model import MattingBase, MattingRefine

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

with torch.no_grad():
    while True:
        # Normal
        frame = cam.read()

        # Rotated
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        src = cv2_frame_to_cuda(frame_rotated)
        pha, fgr = model(src, bgr)[:2]

        vidframe = tb_video[frame_num].unsqueeze_(0).cuda()
        tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
        frame_num += 1
        if frame_num >= tb_video.__len__():
            frame_num = 0

        # res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
        res = pha * fgr + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        key = dsp.step(res)
        if key == ord('q'):
            exit()
