import pyautogui
import time


class Mouse:
    def __init__(
        self, mouse_scale, click_threshold_time=0.2, drag_threshold_time=0.15
    ) -> None:
        """Initialization of Mouse class.

        Args:
            mouse_scale (float): Scale factor for mouse movement.
            click_threshold_time (float, optional): Threshold time for registering a click. Defaults to 0.2.
            drag_threshold_time (float, optional): Threshold time for registering a drag. Defaults to 0.15.
        """
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.mouse_scale = mouse_scale
        self.click_threshold_time = click_threshold_time
        self.drag_threshold_time = drag_threshold_time
        self.left_down = False
        self.middle_down = False
        self.right_down = False
        self.last_time = time.time()

    def control(self, x: float, y: float, mouse_button: str):
        """Moves the mouse to XY coordinates and can perform single clicks, or click and drags when called repeatelly

        Args:
            x (float): x coordinate between 0 and 1
            y (float): y coordinate between 0 and 1
            mouse_button (string): can be "", "left", "middle", or "right"
        """
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
            self.click(x, y, mouse_button)

    def move(self, x, y):
        """Move the mouse to the specified coordinates.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
        """
        pyautogui.moveTo(
            x,
            y,
            duration=0,
            _pause=False,
        )

    def click(self, x, y, mouse_button):
        """Handle mouse clicking.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            mouse_button (str): Mouse button to click.
        """
        current_time = time.time()  # if it has been longer than threshold time
        if current_time - self.last_time > self.click_threshold_time:
            self.last_time = current_time
            print("click")
            pyautogui.click(button=mouse_button, _pause=False)
        elif (
            (current_time - self.last_time > self.drag_threshold_time)
            or self.left_down
            or self.middle_down
            or self.right_down
        ):

            if mouse_button == "left":
                self.last_time = current_time
                if not self.left_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.left_down = True

            if mouse_button == "middle":
                self.last_time = current_time
                if not self.middle_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.middle_down = True

            if mouse_button == "right":
                self.last_time = current_time
                if not self.right_down:
                    pyautogui.mouseDown(button=mouse_button, _pause=False)
                    self.right_down = True

            self.move(x, y)
