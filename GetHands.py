# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

from threading import Thread
import cv2
import mediapipe as mp
import time
import numpy as np
from RNN import LSTM
class GetHands:
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """
    def __init__(
        self,
        render_hands,
        mode,
        surface=None,
        show_window=False,
        hands=1,
        confidence=0.5,
        webcam_id=0,
        model_path="hand_landmarker.task",
        control_mouse=None,
        write_csv=None,
        gesture_vector=None,
        gesture_list=None,
        move_mouse_flag=[False],
        gesture_confidence=0.50,
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
        self.render_hands_mode = mode
        self.confidence = confidence
        self.stopped = False
        self.last_origin = [(0, 0)]
        self.control_mouse = control_mouse
        self.write_csv = write_csv
        self.gesture_vector = gesture_vector
        self.gesture_list = gesture_list
        self.move_mouse_flag = move_mouse_flag
        self.gesture_confidence = gesture_confidence

        self.gesture_model = LSTM("gestureModel.pth")

        self.model_input = np.zeros((1,self.gesture_model.sequence_length, 65), dtype="float32")

        print("self.gesture_model.sequence_length"+str(self.gesture_model.sequence_length))
        # OpenCV setup
        self.stream = cv2.VideoCapture(webcam_id)
        # motion JPG format
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        (self.grabbed, self.frame) = self.stream.read()
        self.frame = cv2.flip(self.frame, 1)
        self.last_timestamp = mp.Timestamp.from_seconds(time.time()).value
        self.timer1 = 0
        self.timer2 = 0

        self.build_model(hands)

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

    def gesture_input(self, result, velocity):
        """Converts Mediapipe landmarks and a velocity into a format usable by the gesture recognition model

        Args:
            result (Mediapipe.hands.result): The result object returned by Mediapipe
            velocity ([(float, float)]): An array of tuples containing the velocity of hands

        Returns:
            array: An array of length 65
        """

        this_frame = []

        if result.handedness == []:
            return False

        for hand in result.hand_world_landmarks:
            for point in hand:
                this_frame.append(point.x)
                this_frame.append(point.y)
                this_frame.append(point.z)
        if velocity != []:
            this_frame.append(velocity[0][0])
            this_frame.append(velocity[0][1])

        this_frame = np.array([this_frame], dtype="float32")

        #pop index zero, which is the oldest frame
        self.model_input[0] = np.roll(self.model_input[0], -1, axis=0)
        self.model_input[0, -1] = this_frame
        #the last index of the input vector is the newest frame
        #append our newest frame to the input

        #keep track of how many frames in the input buffer

        return True

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

    def move_mouse(self, hands_location_on_screen, mouse_button_text):
        """Wrapper method to control the mouse

        Args:
            hands_location_on_screen (origins): The origins result from find_velocity_and_location()
            mouse_button_text (str): Type of click
        """
        if callable(self.control_mouse):
            if hands_location_on_screen != []:
                # (0,0) is the top left corner
                self.control_mouse(
                    hands_location_on_screen[0][0],
                    hands_location_on_screen[0][1],
                    mouse_button_text,
                )

    def results_callback(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Wrapper callback function used by Mediapipe hands model. When called, recives the result of the
        hands model and processes it to find velocity and origins. Then passes data to a gesture recognition model.
        Using all this data will optionally control a mouse. Then will call a render funtion to render the hand.

        Args:
            result (mp.tasks.vision.HandLandmarkerResult): This result object contains handedness, hand world landmarks, and hand normalized landmarks
            output_image (mp.Image): image that hands were detected on
            timestamp_ms (int): time the image was taken
        """

        mouse_button_text = ""

        hands_location_on_screen, velocity = self.find_velocity_and_location(result)

        if self.move_mouse_flag[0]:
            sequence = self.gesture_input(result, velocity)
            if sequence:
                #print(self.model_input)
                confidence, gesture = self.gesture_model.get_gesture(self.model_input)
                print(confidence[0], self.gesture_list[gesture[0]])

        #     if model_input.size != 0:
        #         confidence, gesture = self.gesture_model.get_gesture(model_input)

        #         if confidence[0] > self.gesture_confidence:
        #             print(confidence[0], self.gesture_list[gesture[0]])

        #         if gesture[0] == 0 and confidence[0] > self.gesture_confidence:
        #             mouse_button_text = ""
        #         elif gesture[0] == 1 and confidence[0] > self.gesture_confidence:
        #             mouse_button_text = "left"
        #         elif gesture[0] == 2 and confidence[0] > self.gesture_confidence:
        #             mouse_button_text = "middle"
        #         elif gesture[0] == 3 and confidence[0] > self.gesture_confidence:
        #             mouse_button_text = "right"

        #     self.move_mouse(hands_location_on_screen, mouse_button_text)

        # # write to CSV
        # if self.gesture_vector[len(self.gesture_vector) - 1] == True:
        #     self.write_csv(result.hand_world_landmarks, velocity, self.gesture_vector)

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
            self.render_hands_mode,
            hands_location_on_screen,
            velocity,
            mouse_button_text,
        )

    def start(self):
        """Generates and starts the Mediapipe thread

        Returns:
            self: new thread
        """
        Thread(target=self.run, args=()).start()
        return self

    def run(self):
        """Continuously grabs new frames from the webcam and uses Mediapipe to detect hands
        """
        while not self.stopped:
            if not self.grabbed:
                self.stream.release()
                cv2.destroyAllWindows()
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                self.frame = cv2.flip(self.frame, 1)

            # Detect hand landmarks
            self.detect_hands(self.frame)
            if self.show_window:
                self.show()

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

    def show(self):
        """Displays another window with the raw webcam stream
        """
        cv2.imshow("Video", self.frame)
        if cv2.waitKey(1) == ord("q"):
            self.stopped = True
            cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
