# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

from threading import Thread
import mediapipe as mp
import time
import math
from FeedForward import NeuralNet
import traceback
from Console import GestureConsole
from Webcam import Webcam
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
        render_hands,
        mediapipe_model="hand_landmarker.task",
        mouse=None,
        flags=None,
        keyboard=None,
    ):
        Thread.__init__(self)

        self.model_path = mediapipe_model
        self.render_hands = render_hands
        self.confidence = 0.8
        self.stopped = False
        self.mouse = mouse

        self.flags = flags
        self.keyboard = keyboard
        self.console = GestureConsole()

        self.camera = Webcam()

        self.set_gesture_model(flags["gesture_model_path"])
        
        self.gesture_list = self.gesture_model.labels
        self.confidence_vectors = self.gesture_model.confidence_vector
        self.gestures = ["no gesture"]
        self.delay = 0

        self.click = ""
        self.mouse_location = []

        (self.grabbed, self.frame) = self.camera.read()

        self.timer = 0

        self.build_mediapipe_model(flags["number_of_hands"])
        
    def set_gesture_model(self, path):
        self.gesture_model = NeuralNet(path)

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
            min_hand_detection_confidence=self.confidence,
            min_hand_presence_confidence=self.confidence,
            min_tracking_confidence=0.5,
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.results_callback,
        )

        # build hands model
        self.hands_detector = self.HandLandmarker.create_from_options(self.options)

    def move_mouse(self, location, button: str):
        """Wrapper method to control the mouse

        Args:
            hands_location_on_screen (origins): The origins result from find_velocity_and_location()
            mouse_button_text (str): Type of click
        """
        if callable(self.control_mouse):
            if location != []:
                # (0,0) is the top left corner
                self.control_mouse(
                    location[0][0],
                    location[0][1],
                    button,
                )

    def results_callback(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        # this try catch block is for debugging. this code runs in a different thread and doesn't automatically raise its own exceptions
        try:

            self.mouse_location = []
            self.click = ""
            if len(result.hand_world_landmarks) == 0:

                self.render_hands(
                    result,
                    None,
                    None,
                    None,
                )
                return

            location, velocity = self.gesture_model.find_velocity_and_location(result)

            if self.flags["run_model_flag"]:

                # get all the hands and format them
                model_inputs = self.gesture_model.gesture_input(result, velocity)

                # for some reason parrellization with batches makes the model super slow
                # if len(model_inputs) > 0:
                #     self.confidence_vector, indexs = self.gesture_model.get_gesture_confidence(model_inputs)
                #     # only take inputs from the first hand, subsequent hands can't control the keyboard
                #     self.keyboard.gesture_input(self.confidence_vector[0])

                # serialized input
                hand_confidences = []  # prepare data for console table
                gestures = []  # store gesture output as text
                for index, hand in enumerate(model_inputs):
                    confidences, predicted, predicted_confidence = (
                        self.gesture_model.get_gesture([hand], print_table=False)
                    )
                    gestures.append(self.gesture_list[predicted[0]])  # save gesture
                    hand_confidences.append(confidences[0])
                    # only take inputs from the first hand, subsequent hands can't control the keyboard

                self.gestures = gestures
                self.confidence_vectors = hand_confidences

                self.keyboard.gesture_input(self.confidence_vectors[0])

                self.console.table(self.gesture_list, hand_confidences)

            if self.flags["move_mouse_flag"] and location != []:
                mouse_button_text = ""
                hand = result.hand_world_landmarks[0]
                if self.mouse.is_clicking(hand[8], hand[4], self.flags["click_sense"]):
                    mouse_button_text = "left"

                self.click = mouse_button_text
                self.mouse_location = location

            # timestamps are in microseconds so convert to ms

            current_time = time.time()
            self.delay = (current_time - self.timer) * 1000
            self.timer = current_time

            self.render_hands(
                result,
                self.flags["render_hands_mode"],
                location,
                velocity,
            )

        except Exception as e:
            traceback.print_exc()
            quit()

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

    def stop(self):
        self.stopped = True
