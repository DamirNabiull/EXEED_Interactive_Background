import cv2
import time


def capture_background(cam, path):
    time.sleep(10)
    frame = cam.read()
    cv2.imwrite(path, frame)
    return frame
