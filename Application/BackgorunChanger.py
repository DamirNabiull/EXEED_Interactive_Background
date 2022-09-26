import time
from threading import Thread, Lock

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


# ----------- Utility classes -------------

# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame
    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio
    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        # fps_sample = 1.0 / (t_new - self._last_tick)
        fps_sample = 1.0
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()
    def get(self):
        return self._avg_fps

# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


def cv2_frame_to_cuda(cv2_frame):
    cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(cv2_frame)).unsqueeze_(0).cuda()


class VideoDataset(Dataset):
    def __init__(self, path: str, transforms: any = None, rotated = False):
        self.cap = cv2.VideoCapture(path)
        self.transforms = transforms

        if rotated:
            # For rotated
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            # For normal
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.cap.read()
        if not ret:
            raise IndexError(f'Idx: {idx} out of length: {len(self)}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)
        return img

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()