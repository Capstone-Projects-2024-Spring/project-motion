# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

from threading import Thread
import cv2
import mediapipe as mp
import time


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
        hands=2,
        confidence=0.5,
        webcam_id=0,
        model_path="hand_landmarker.task",
        is_pinching=None
    ):
        """
        Class that continuously gets frames and extracts hand data
        with a dedicated thread and

        Start frame capture and hands processing by calling start() on this class

        Args:
            param render_hands (function): Mediapipe callback function
                This is called whenever the current frame has finished proccessing.
                The callback will recieve 3 parameters:
                result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int

            show_window (bool): Whether to display the webcam in a seperate window or not

            hands (int): number hands to detect

            confidence (float): Minimum hand detection confidence level

            webcam_id (int): Camera device id for OpenCV Videocapture function

            model_path (string): Path to Mediapipe hand landmarker model
        """
        self.surface = surface
        self.show_window = show_window
        self.model_path = model_path
        self.render_hands = render_hands
        self.render_hands_mode = mode
        self.confidence = confidence
        self.stopped = False
        self.last_origin = [(0,0)]
        self.is_pinching = is_pinching

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
        """
        Wrapper function which finds the time taken to process the image, the origin of the hand, and velocity
        """
        pinch = ""
        normalized_origin_offset = []

        for hand in result.hand_world_landmarks:
            # take middle finger knuckle
            normalized_origin_offset.append(hand[9])
            if self.is_pinching(hand[8], hand[4]):
                pinch = "Pinching"


        world_hand_origin = []
        velocity = []

        for index, hand in enumerate(result.hand_landmarks):
            originX = hand[9].x - normalized_origin_offset[index].x
            originY = hand[9].y - normalized_origin_offset[index].y
            originZ = hand[9].z - normalized_origin_offset[index].z
            world_hand_origin.append((originX, originY, originZ))
            velocityX = (self.last_origin[index][0] - world_hand_origin[index][0])
            velocityY = (self.last_origin[index][1] - world_hand_origin[index][1])
            velocity.append((velocityX,velocityY))
            self.last_origin = world_hand_origin

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
            world_hand_origin,
            velocity,
            pinch
        )

    def start(self):
        Thread(target=self.run, args=()).start()
        return self

    def run(self):
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.hands_detector.detect_async(
            mp_image, mp.Timestamp.from_seconds(time.time()).value
        )
        self.timer1 = mp.Timestamp.from_seconds(time.time()).value

    def show(self):
        cv2.imshow("Video", self.frame)
        if cv2.waitKey(1) == ord("q"):
            self.stopped = True
            cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
