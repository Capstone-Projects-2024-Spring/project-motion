import pydirectinput
import time
from Console import GestureConsole
import numpy as np
import threading


class Keyboard(threading.Thread):
    def __init__(
        self,
        threshold=0.0,
        toggle_threshold=0.3,
        threshold_off_time=0.05,
        flags=None,
        tps=60,
    ) -> None:

        threading.Thread.__init__(self, daemon=True)
        """ the used for toggling the mouse is saved in flags

        Args:
            threshold (float, optional): don't press key if press will be less than this time. Defaults to 0.0.
            toggle_threshold (float, optional): if same key is receieved twice within this time, but not continously, key will be toggled
        """
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0
        self.tps = tps
        self.threshold = threshold
        self.toggle_threshold = toggle_threshold
        self.last_key = "none"
        self.toggle_keys_pressed = {}
        self.key_pressed = ("none", time.time())
        self.console = GestureConsole()
        self.flags = flags
        self.threshold_off_time = threshold_off_time

    def run(self):
        event = threading.Event()
        while True:
            if (
                self.flags["run_model_flag"]
                and len(self.flags["hands"].confidence_vectors) > 0
            ):
                # send only the first hand confidence vector the gesture model output
                confidence_vectors = self.flags["hands"].confidence_vectors
                if len(confidence_vectors) > self.flags["keyboard_hand_num"]:
                    self.gesture_input(
                        confidence_vectors[self.flags["keyboard_hand_num"]]
                    )
            else:
                self.release()
            event.wait(timeout=1 / self.tps)

    def release(self):
        if self.key_pressed[0] != "none":
            self.console.print(f"releasing key: {self.key_pressed[0]}")
            pydirectinput.keyUp(self.key_pressed[0])
        for key in self.toggle_keys_pressed:
            self.console.print(f"releasing key: {key}")
            pydirectinput.keyUp(key)
        self.toggle_keys_pressed = {}
        self.key_pressed = ("none", time.time())

    def gesture_input(self, confidences):
        max_value = np.max(confidences)
        # gesture_list = self.flags["gesture_list"]

        if max_value > self.flags["min_confidence"]:
            max_index = np.argmax(confidences)
            if max_index == 0:
                self.press("none")
            elif max_index == 1:
                self.press("space")
            elif max_index == 2:
                self.press(self.flags["toggle_mouse_key"])
            elif max_index == 3:
                self.press("p")

    def press(self, key: str):
        current_time = time.time()

        # if key is same update time and return
        if self.last_key == key:
            self.key_pressed = (key, current_time)
            return

        # self.key_pressed = (key, current_time)

        # disable toggle on this key if it is toggled
        if key in self.toggle_keys_pressed and self.toggle_keys_pressed[key]:
            self.console.print(f"releasing key (was toggled): {key}")
            self.toggle_keys_pressed[key] = False
            pydirectinput.keyUp(key)  # Release the last key

        # enable toggle on this key if it has already been pressed, its being recieved twiced in under the threshold time, and wasn't recieved for at least the off time
        if (
            key == self.key_pressed[0]
            and current_time - self.key_pressed[1] < self.toggle_threshold
            and current_time - self.key_pressed[1] > self.threshold_off_time
        ):
            if key != "none":
                self.toggle_keys_pressed[key] = True
                self.console.print(f"pressing key (toggled): {key}")
                pydirectinput.keyDown(key)

        # if key is not toggled
        if (
            not key in self.toggle_keys_pressed
            or key in self.toggle_keys_pressed
            and self.toggle_keys_pressed[key] == False
        ):
            # if there is a key pressed
            if self.key_pressed[0] != "none":
                self.console.print(f"releasing key: {self.key_pressed[0]}")
                pydirectinput.keyUp(self.key_pressed[0])
                self.key_pressed = (key, current_time)
            # press
            elif key != "none":
                self.console.print(f"pressing key: {key}")
                self.key_pressed = (key, current_time)
                pydirectinput.keyDown(key)

        self.last_key = key
