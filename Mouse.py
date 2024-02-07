import pyautogui
import time
class Mouse:
    def __init__(self, mouse_scale) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.mouse_scale = mouse_scale

    def mouse_rel(self,x, y):
        pyautogui.moveRel(
        1 * x * pyautogui.size().width,
        1 * y * pyautogui.size().height,
        duration=0,
        _pause=False,
    )

    def move(self,x, y):
        pyautogui.moveTo(
            ((self.mouse_scale)*x - (self.mouse_scale-1)/2) * pyautogui.size().width,
            ((self.mouse_scale)*y - (self.mouse_scale-1)/2) * pyautogui.size().height,
            duration=0,
            _pause=False,
        )

    def click(self,last_time):
        
        #if it has been longer than 0.5 seconds
        current_time = time.time()

        if current_time - last_time[0] > 0.5:
            last_time[0] = current_time
            pyautogui.click()
