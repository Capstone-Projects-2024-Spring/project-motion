# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

from threading import Thread
import mediapipe as mp
from time import time
from FeedForward import FeedForward
import traceback
import Console
from Webcam import Webcam
from LSTM import LSTM
import threading
from copy import copy

lock = threading.Lock()

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
class GetHands(Thread):
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """
    def __init__(
        self,
        mediapipe_model="hand_landmarker.task",
        flags=None,
    ):
        Thread.__init__(self, daemon=True)
        self.model_path = mediapipe_model
        self.confidence = 0.1
        self.stopped = False

        self.flags = flags

        self.camera = Webcam()
        self.camera.start(self.camera.working_ports[0])
        if "feedforward" in flags["gesture_model_path"]:
            self.set_gesture_model_FF(flags["gesture_model_path"])
        elif "lstm" in flags["gesture_model_path"]:
            self.set_gesture_model_LSTM(flags["gesture_model_path"])
        else:
            print("path: %s", flags["gesture_model_path"])
            raise Exception("invalid model file path")
            
        self.hand_sequences = [[],[],[],[]]
        
        self.gesture_list = self.gesture_model.labels
        self.confidence_vectors = self.gesture_model.confidence_vector
        self.flags["gesture_list"] = self.gesture_list
        self.gestures = ["no gesture"]
        self.delay = 0
        self.result = []

        self.click = ""
        self.location = []
        self.velocity = []
        self.num_hands_detected = 0

        (self.grabbed, self.frame) = self.camera.read()

        self.timer = 0

        self.build_mediapipe_model(flags["number_of_hands"])

    def set_gesture_model_FF(self, path):
        self.gesture_model = FeedForward(path, force_cpu=True)
        Console.print(self.gesture_model.device)
        
    def set_gesture_model_LSTM(self, path):
        self.gesture_model = LSTM(path)
        Console.print(self.gesture_model.device)

    def build_mediapipe_model(self, hands_num):
        """Takes in option parameters for the Mediapipe hands model

        Args:
            hands_num (int): max number of hands for Mediapipe to find
        """
        # mediapipe setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            num_hands=hands_num,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.7,
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.results_callback,
        )

        # build hands model
        self.hands_detector = self.HandLandmarker.create_from_options(self.options)
        
    def get_results(self):
        with lock:
            return copy(self.result)

    def results_callback(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        with lock:
            self.location = []
            self.click = ""
            self.velocity = []
            self.num_hands_detected = len(result.hand_world_landmarks)
            if self.num_hands_detected == 0:
                self.result = []
                self.confidence_vectors = []
                Console.table(self.gesture_list, self.confidence_vectors)
                return

            self.result = result

            location, velocity = self.gesture_model.find_velocity_and_location(result)
            self.location = location
            self.velocity = velocity

            if self.flags["run_model_flag"]:

                # get all the hands and format them
                model_inputs = self.gesture_model.gesture_input(result, velocity)
                #print(model_inputs)

                # serialized input
                hand_confidences = []  # prepare data for console table
                gestures = []  # store gesture output as text
                for index, hand in enumerate(model_inputs):
                    #build a sequence of hands
                    if type(self.gesture_model) == LSTM:
                        self.hand_sequences[index].append(hand)
                        if len(self.hand_sequences[index]) > self.gesture_model.sequence_length:
                            self.hand_sequences[index].pop(0)
                        output = self.gesture_model.get_gesture(self.hand_sequences[index])
                    elif type(self.gesture_model) == FeedForward:
                        output = self.gesture_model.get_gesture([hand])
                    if output != None:
                        confidences, predicted, predicted_confidence = output
                        gestures.append(self.gesture_list[predicted[0]])  # save gesture
                        hand_confidences.append(confidences[0])
                        Console.table(self.gesture_list, confidences)

                self.gestures = gestures
                self.confidence_vectors = hand_confidences
                
            # timestamps are in microseconds so convert to ms

            current_time = time()
            self.delay = (current_time - self.timer) * 1000
            self.timer = current_time

    def run(self):
        """Continuously grabs new frames from the webcam and uses Mediapipe to detect hands"""
        while not self.stopped:
            if not self.grabbed:
                self.camera.stop()
                self.stop()
            else:
                try:
                    (self.grabbed, self.frame) = self.camera.read()
                except:
                    print("camera read fail")
                    quit()
            # Detect hand landmarks
            self.detect_hands(self.frame)

    def detect_hands(self, frame):
        """Wrapper function for Mediapipe's hand detector in livestream mode

        Args:
            frame (cv2.image): OpenCV webcam frame
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.hands_detector.detect_async(
            mp_image, mp.Timestamp.from_seconds(time()).value
        )

    def stop(self):
        self.stopped = True
