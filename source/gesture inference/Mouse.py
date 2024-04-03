import pyautogui
import time
from Console import GestureConsole
import math


class Mouse:
    def __init__(
        self,
        mouse_sensitivity=1,
        click_threshold_time=0.22,
        drag_threshold_time=0.2,
        x_scale=1.3,
        y_scale=1.5,
        alpha=0.15,
        deadzone=15,
    ) -> None:
        """Initialization of Mouse class.

        Args:
            mouse_scale (float): Scale factor for mouse movement.
            click_threshold_time (float, optional): The click_threshold_time is the minimum time interval between two consecutive clicks to register them as separate clicks.
                                                    If you increase click_threshold_time, you will need to wait longer between two clicks for them to be considered separate clicks.
                                                    If you decrease click_threshold_time, you can click faster and still register them as separate clicks.
            drag_threshold_time (float, optional): The drag_threshold_time is the maximum time interval after which a mouse movement after a click is considered a drag rather than a separate click.
                                                    If you increase drag_threshold_time, you will have more time to move the mouse after clicking without triggering a drag.
                                                    If you decrease drag_threshold_time, even a slight movement of the mouse shortly after clicking can be considered a drag rather than a separate click.
        """
        if click_threshold_time <= drag_threshold_time:
            raise Exception(
                "drag_threshold_time must be less than click_threshold_time"
            )

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.mouse_sensitivity = float(mouse_sensitivity)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.deadzone = deadzone

        self.click_threshold_time = click_threshold_time
        self.drag_threshold_time = drag_threshold_time
        self.left_down = False
        self.middle_down = False
        self.right_down = False
        self.last_time = time.time()
        self.console = GestureConsole()

        # expontial moving average stuff
        self.x_window = []
        self.y_window = []
        self.window_size = 12
        self.alpha = alpha

    def control(self, x: float, y: float, mouse_button: str):
        """Moves the mouse to XY coordinates and can perform single clicks, or click and drags when called repeatelly

        Args:
            x (float): x coordinate between 0 and 1
            y (float): y coordinate between 0 and 1
            mouse_button (string): can be "", "left", "middle", or "right"
        """
        x = int(
            (
                (self.x_scale * self.mouse_sensitivity) * x
                - (self.x_scale * self.mouse_sensitivity - 1) / 2
            )
            * pyautogui.size().width
        )
        y = int(
            (
                (self.y_scale * self.mouse_sensitivity) * y
                - (self.y_scale * self.mouse_sensitivity - 1) / 2
            )
            * pyautogui.size().height
        )

        # Check if the movement is smaller than the specified radius
        last_x, last_y = pyautogui.position()
        distance = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

        # Specify the radius distance (you can adjust this value)
        ignore_small_movement = distance <= self.deadzone

        self.x_window.append(x)
        self.y_window.append(y)

        x = self.exponential_moving_average(self.x_window)
        y = self.exponential_moving_average(self.y_window)

        if len(self.x_window) > self.window_size:
            self.x_window.pop(0)
            self.y_window.pop(0)
            if mouse_button == "":
                # un-click
                @self.console.console_flag
                def print():
                    self.console.print(f"releasing mouse {mouse_button}")
                if self.left_down:
                    pyautogui.mouseUp(button="left", _pause=False)
                    self.left_down = False
                if self.middle_down:
                    pyautogui.mouseUp(button="middle", _pause=False)
                    self.middle_down = False
                if self.right_down:
                    pyautogui.mouseUp(button="right", _pause=False)
                    self.right_down = False
                if not ignore_small_movement:
                    self.move(x, y)
            else:
                # click or click and drag
                self.click(
                    x,
                    y,
                    mouse_button,
                )

    def is_clicking(self, tip1, tip2, click_sensitinity):
        distance = math.sqrt(
            (tip1.x - tip2.x) ** 2 + (tip1.y - tip2.y) ** 2 + (tip1.z - tip2.z) ** 2
        )
        if distance < click_sensitinity:
            return True
        else:
            return False

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
            @self.console.console_flag
            def print():
                self.console.print(f"clicking mouse {mouse_button}")
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

    def exponential_moving_average(self, data):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(self.alpha * data[i] + (1 - self.alpha) * ema[i - 1])
        return ema[len(data) - 1]
