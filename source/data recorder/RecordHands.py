# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

from threading import Thread
import mediapipe as mp
import time
import math
import traceback
from Webcam import Webcam

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class RecordHands:
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """

    def __init__(
        self,
        render_hands,
        surface=None,
        show_window=False,
        confidence=0.5,
        model_path="hand_landmarker.task",
        write_csv=None,
        gesture_list=None,
        gesture_confidence=0.50,
        flags=None,
    ):
        """Builds a Mediapipe hand model and a PyTorch gesture recognition model

        Args:
            render_hands (_type_): _description_
            mode (_type_): _description_
            surface (_type_, optional): _description_. Defaults to None.
            show_window (bool, optional): _description_. Defaults to False.
            hands (int, optional): _description_. Defaults to 1.
            confidence (float, optional): _description_. Defaults to 0.5.
            webcam_id (int, optional): _description_. Defaults to 0.
            model_path (str, optional): _description_. Defaults to "hand_landmarker.task".
            control_mouse (_type_, optional): _description_. Defaults to None.
            write_csv (_type_, optional): _description_. Defaults to None.
            gesture_vector (_type_, optional): _description_. Defaults to None.
            gesture_list (_type_, optional): _description_. Defaults to None.
            move_mouse_flag (list, optional): _description_. Defaults to [False].
            gesture_confidence (float, optional): _description_. Defaults to 0.50.
        """
        self.surface = surface
        self.show_window = show_window
        self.model_path = model_path
        self.render_hands = render_hands
        self.confidence = confidence
        self.stopped = False
        self.write_csv = write_csv
        self.gesture_vector = flags["gesture_vector"]
        self.gesture_list = gesture_list
        self.gesture_confidence = gesture_confidence
        self.flags = flags
        self.sensitinity = 0.05
        self.last_origin = [(0, 0)]
        self.camera = Webcam()


        (self.grabbed, self.frame) = self.camera.read()

        self.last_timestamp = mp.Timestamp.from_seconds(time.time()).value
        self.timer1 = 0
        self.timer2 = 0

        self.build_model(1)

    def build_model(self, hands_num):
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
            min_hand_detection_confidence=self.confidence,
            min_hand_presence_confidence=self.confidence,
            min_tracking_confidence=self.confidence,
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.results_callback,
        )

        # build hands model
        self.hands_detector = self.HandLandmarker.create_from_options(self.options)

    def results_callback(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        # this try catch block is for debuggin. this code runs in a different thread and doesn't automatically raise its own exceptions
        try:
            location, velocity = self.find_velocity_and_location(result)

            # write to CSV
            # flag for writing is saved in the last index of this vector
            if self.gesture_vector[len(self.gesture_vector) - 1] == True:
                self.write_csv(
                    result.hand_world_landmarks, velocity, self.gesture_vector
                )

            # timestamps are in microseconds so convert to ms
            self.timer2 = mp.Timestamp.from_seconds(time.time()).value
            hands_delay = (self.timer2 - self.timer1) / 1000
            total_delay = (timestamp_ms - self.last_timestamp) / 1000
            self.last_timestamp = timestamp_ms
            self.render_hands(
                result,
                output_image,
                (total_delay, hands_delay),
                self.surface,
                location,
                velocity,
            )

        except Exception as e:
            traceback.print_exc()
            quit()

    def start(self):
        Thread(target=self.run, args=()).start()
        return self

    def start(self):
        """Generates and starts the Mediapipe thread

        Returns:
            self: new thread
        """
        Thread(target=self.run, args=()).start()
        return self

    def run(self):
        """Continuously grabs new frames from the webcam and uses Mediapipe to detect hands"""
        while not self.stopped:
            if not self.grabbed:
                self.camera.stop()
                self.stop()
            else:
                (self.grabbed, self.frame) = self.camera.read()

            # Detect hand landmarks
            self.detect_hands(self.frame)

    def detect_hands(self, frame):
        """Wrapper function for Mediapipe's hand detector in livestream mode

        Args:
            frame (cv2.image): OpenCV webcam frame
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.hands_detector.detect_async(
            mp_image, mp.Timestamp.from_seconds(time.time()).value
        )
        self.timer1 = mp.Timestamp.from_seconds(time.time()).value

    def stop(self):
        self.stopped = True

    def find_velocity_and_location(self, result):
        """Given a Mediapipe result object, calculates the velocity and origin of hands.

        Args:
            result (Mediapipe.hands.result): Direct output object from Mediapipe hands model

        Returns:
            (origins, velocity): A tuple containing an array of tuples representing hand origins, and an array of tuples containing hand velocitys
        """

        normalized_origin_offset = []
        hands_location_on_screen = []
        velocity = []

        for hand in result.hand_world_landmarks:
            # take middle finger knuckle
            normalized_origin_offset.append(hand[9])

        for index, hand in enumerate(result.hand_landmarks):
            originX = hand[9].x - normalized_origin_offset[index].x
            originY = hand[9].y - normalized_origin_offset[index].y
            originZ = hand[9].z - normalized_origin_offset[index].z
            hands_location_on_screen.append((originX, originY, originZ))
            velocityX = self.last_origin[index][0] - hands_location_on_screen[index][0]
            velocityY = self.last_origin[index][1] - hands_location_on_screen[index][1]
            velocity.append((velocityX, velocityY))
            self.last_origin = hands_location_on_screen

        return hands_location_on_screen, velocity
