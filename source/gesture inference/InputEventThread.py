# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

import threading 
import traceback

class InputEventThread(threading.Thread):
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """

    def __init__(
        self,
        method,
        times_per_second = 200
    ):
        threading.Thread.__init__(self)
        self.stopped = False
        self.method = method
        self.event = threading.Event()
        self.tps = times_per_second

    def run(self):
        while not self.stopped:
            try:
                self.method()
                self.event.wait(1/self.tps)

            except Exception as e:
                traceback.print_exc()
                quit()

    def stop(self):
        self.event.set()
        self.join()
