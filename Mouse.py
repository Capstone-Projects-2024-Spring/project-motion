import pyautogui
import time


class Mouse:
    def __init__(self, mouse_scale, threshold_time=0.6) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.mouse_scale = mouse_scale
        self.threshold_time = threshold_time
        self.left_down = False
        self.middle_down = False
        self.right_down = False

    def control(self, x, y, mouse_button, last_time):

        x = (
            (self.mouse_scale) * x - (self.mouse_scale - 1) / 2
        ) * pyautogui.size().width
        y = (
            (self.mouse_scale) * y - (self.mouse_scale - 1) / 2
        ) * pyautogui.size().height

        if mouse_button == "":
            if self.left_down:
                pyautogui.mouseUp(button="left")
                self.left_down = False
            if self.middle_down:
                pyautogui.mouseUp(button="middle")
                self.middle_down = False
            if self.right_down:
                pyautogui.mouseUp(button="right")
                self.right_down = False
            self.move(x, y)
        else:
            self.click(last_time, x, y, mouse_button)

    def mouse_rel(self, x, y):
        pyautogui.moveRel(
            1 * x * pyautogui.size().width,
            1 * y * pyautogui.size().height,
            duration=0,
            _pause=False,
        )

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
        if current_time - last_time[0] > self.threshold_time:
            last_time[0] = current_time
            print(mouse_button)
            pyautogui.click(button=mouse_button)
        else:
            if mouse_button == "left":
                last_time[0] = current_time
                if not self.left_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.left_down = True

            if mouse_button == "middle":
                last_time[0] = current_time
                if not self.left_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.left_down = True

            if mouse_button == "right":
                last_time[0] = current_time
                if not self.left_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.left_down = True

            self.move(x, y)
