import pydirectinput
import time
from Console import GestureConsole
import numpy as np


class Keyboard:
    def __init__(
        self,
        threshold=0.0,
        toggle_key_threshold=0.15,
        toggle_key_toggle_time=1,
        toggle_mouse_func=None,
        flags=None,
        toggle_key="",
    ) -> None:
        """_summary_

        Args:
            threshold (float, optional): don't press key if press will be less than this time. Defaults to 0.0.
            toggle_key_threshold (float, optional): don't press the toggle key if press will be less than this time. Defaults to 0.15.
            mouse_toggle_key (str, optional): the key to use as theh mosue ontrol toggle. . Defaults to 'm'.

            Key press options:
            ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace', 'browserback', 'browserfavorites', 'browserforward', 'browserhome', 'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear', 'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete', 'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20', 'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja', 'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail', 'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack', 'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn', 'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn', 'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator', 'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab', 'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen', 'command', 'option', 'optionleft', 'optionright']
        """
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0
        self.threshold = threshold
        self.toggle_key_threshold = toggle_key_threshold
        self.last_time = time.time()
        self.last_time_toggle_key = time.time()
        self.last_key = ""
        self.toggle_key_pressed = False
        self.key_pressed = False
        self.toggle_mouse_func = toggle_mouse_func
        self.toggle_key_toggle_time = toggle_key_toggle_time
        self.console = GestureConsole()
        self.flags = flags
        self.toggle_key = toggle_key

    def release(self):
        if self.key_pressed:
            pydirectinput.keyUp(self.last_key)
            self.key_pressed = False
        if self.toggle_key_pressed:
            self.toggle_key_pressed = False

    def gesture_input(self, confidences):
        max_value = np.max(confidences)
        max_index = np.argmax(confidences)
        if max_index == 0:
            self.press("space")
        elif max_index == 1:
            self.press("none")
        elif max_index == 2:
            self.press(self.toggle_key)

    def press(self, key: str):
        
        current_time = time.time() 
        if key == "none":
            self.key_pressed = False
            pydirectinput.keyUp(self.last_key)  # Release the last key
            return
        if key != self.last_key:
            self.key_pressed = False
            self.toggle_key_pressed = False
            self.last_key = key

            if key == self.toggle_key:
                self.last_time_toggle_key = current_time
            else:
                self.last_time = current_time

        if key == self.toggle_key:
            self.handle_toggle_key(current_time)
        elif current_time - self.last_time > self.threshold and not self.key_pressed:
            self.key_pressed = True
            #self.console.print(f"pressing key: {key}")
            pydirectinput.keyDown(key)
        # else:
        #     self.console.print(f"key {key} already pressed")

    def handle_toggle_key(self, current_time):
        if (
            not self.toggle_key_pressed
            or current_time - self.last_time_toggle_key > self.toggle_key_toggle_time
        ):
            if current_time - self.last_time_toggle_key > self.toggle_key_threshold:
                if self.toggle_mouse_func is not None:
                    self.toggle_mouse_func()
                self.toggle_key_pressed = True
                self.last_time_toggle_key = current_time
