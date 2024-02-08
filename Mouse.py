import pyautogui
import time


class Mouse:
    def __init__(
        self, mouse_scale, click_threshold_time=0.2, drag_threshold_time=0.15
    ) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.mouse_scale = mouse_scale
        self.click_threshold_time = click_threshold_time
        self.drag_threshold_time = drag_threshold_time
        self.left_down = False
        self.middle_down = False
        self.right_down = False

    def control(self, x, y, mouse_button, last_time):

        x = int(
            ((self.mouse_scale) * x - (self.mouse_scale - 1) / 2)
            * pyautogui.size().width
        )
        y = int(
            ((self.mouse_scale) * y - (self.mouse_scale - 1) / 2)
            * pyautogui.size().height
        )

        if mouse_button == "":
            # un-click
            if self.left_down:
                pyautogui.mouseUp(button="left", _pause=False)
                self.left_down = False
            if self.middle_down:
                pyautogui.mouseUp(button="middle", _pause=False)
                self.middle_down = False
            if self.right_down:
                pyautogui.mouseUp(button="right", _pause=False)
                self.right_down = False
            self.move(x, y)
        else:
            # click or click and drag
            self.click(last_time, x, y, mouse_button)

    def move(self, x, y):
        pyautogui.moveTo(
            x,
            y,
            duration=0,
            _pause=False,
        )

    def click(self, last_time, x, y, mouse_button):

        # if it has been longer than threshold time
        current_time = time.time()
        if current_time - last_time[0] > self.click_threshold_time:
            last_time[0] = current_time
            print("click")
            pyautogui.click(button=mouse_button, _pause=False)
            # pyautogui.mouseDown(button=mouse_button)
            # pyautogui.mouseUp(button=mouse_button)
        elif (
            (current_time - last_time[0] > self.drag_threshold_time)
            or self.left_down
            or self.middle_down
            or self.right_down
        ):

            if mouse_button == "left":
                last_time[0] = current_time
                if not self.left_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.left_down = True

            if mouse_button == "middle":
                last_time[0] = current_time
                if not self.middle_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.middle_down = True

            if mouse_button == "right":
                last_time[0] = current_time
                if not self.right_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.right_down = True

            self.move(x, y)
