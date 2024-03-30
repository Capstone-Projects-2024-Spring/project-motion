# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

import cv2
class Webcam:
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """
    def __init__(
        self,
        webcam_id=0,
    ):
        self.webcam_id = webcam_id
        self.start()

    def start(self):
        # OpenCV setup
        self.stream = cv2.VideoCapture(self.webcam_id)
        # motion JPG format
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        (self.grabbed, self.frame) = self.stream.read()
        self.frame = cv2.flip(self.frame, 1)

    def stop(self):
        self.stream.release()
        cv2.destroyAllWindows()

    def read(self, flip=True):
        (self.grabbed, self.frame) = self.stream.read()
        if flip:
            self.frame = cv2.flip(self.frame, 1)
        return (self.grabbed, self.frame)