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
        self.toggle_instances = []
        self.key_pressed = [("none", 0.0), ("none", 0.0), ("none", 0.0)]
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
                self.release_all()
            event.wait(timeout=1 / self.tps)

    def release_all(self):
        if self.key_pressed[2][0] != "none":
            self.console.print(f"releasing key: {self.key_pressed[2][0]}")
            pydirectinput.keyUp(self.key_pressed[2][0])
        for key in self.toggle_keys_pressed:
            self.console.print(f"releasing key: {key}")
            pydirectinput.keyUp(key)
        self.toggle_keys_pressed = {}
        self.key_pressed = [("none", 0.0), ("none", 0.0), ("none", 0.0)]

    def gesture_input(self, confidences, hand_num=0):
        max_value = np.max(confidences)
        max_index = np.argmax(confidences)
        bindings = self.flags["key_bindings"]

        if len(bindings) == len(confidences):
            if max_value > self.flags["min_confidence"]:
                self.press(bindings[max_index])

    def press(self, key: str):
        current_time = time.time()

        # do nothing with repeated inputs
        if self.last_key == key:
            return
        """
            up up up     |  release and press
            up up down   |  release
            up down up   |  toggle
            down up up   |  release and press
            down up down |  un-toggle
        """
        # when there is an input change, remove oldest edge and insert newest edge
        self.key_pressed.append((key, current_time))
        self.key_pressed.pop(0)

        if key == "none":
            # down up down
            if self.key_pressed[0][0] == "none":
                # self.console.print("none key none")
                # self.console.print(self.key_pressed)
                self.untoggle_or_release(self.key_pressed[1])
            # up up down
            else:
                # self.console.print("key key none")
                # self.console.print(self.key_pressed)
                self.release(key)
        else:
            # up down up
            if self.key_pressed[1][0] == "none":
                # self.console.print("key none key")
                # self.console.print(self.key_pressed)
                self.toggle_or_press(current_time, key)
            # down up up
            elif self.key_pressed[0][0] == "none":
                # self.console.print("none key key")
                # self.console.print(self.key_pressed)
                self.release_and_press(key)
            else:
                # self.console.print("key key key")
                # self.console.print(self.key_pressed)
                self.release_and_press(key)

        self.last_key = key

    def untoggle_or_release(self, instance):
        key = instance[0]
        if instance in self.toggle_instances:
            return
        if key in self.toggle_keys_pressed:
            if self.toggle_keys_pressed[key] == True:
                self.toggle_keys_pressed[key] = False
        self.console.print(f"releasing key: {key}")
        pydirectinput.keyUp(key)

    def release(self, key):
        self.console.print(f"releasing key: {self.last_key}")
        pydirectinput.keyUp(self.last_key)  # Release the last key

    def toggle_or_press(self, current_time, key):
        # if the same key was pressed during the first edge and this edge
        if self.key_pressed[0][0] == self.key_pressed[2][0]:
            # if the first key press time wasn't longer ago than the threshold time
            if current_time - self.key_pressed[0][1] < self.toggle_threshold:
                # if other key was pressed for longer than off time
                if (
                    self.key_pressed[1][1] - self.key_pressed[0][1]
                    > self.threshold_off_time
                ):
                    self.toggle_keys_pressed[key] = True
                    self.toggle_instances.append(self.key_pressed[2])
                    self.console.print(f"pressing key (toggled on): {key}")
                    pydirectinput.keyDown(key)
            else:
                self.console.print(f"pressing key: {key}")
                pydirectinput.keyDown(key)
        else:
            self.console.print(f"pressing key: {key}")
            pydirectinput.keyDown(key)

    def release_and_press(self, key):
        if (
            not key in self.toggle_keys_pressed
            or self.toggle_keys_pressed[key] == False
        ):
            self.console.print(f"releasing key: {self.last_key}")
            pydirectinput.keyUp(self.last_key)  # Release the last key
            self.console.print(f"pressing key: {key}")
            pydirectinput.keyDown(key)
