import pydirectinput
import time
from Console import GestureConsole
import math


class Mouse:
    def __init__(
        self,
        mouse_sensitivity=1,
        x_scale=1.3,
        y_scale=1.5,
        alpha=0.15,
        deadzone=15,  
        single_click_duration=1 / 5,
        is_relative=True,
        acceleration_factor=1.5 ,
        linear_factor=0.25
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
        self.screen_width, self.screen_height = pydirectinput.size()
        pydirectinput.FAILSAFE = False
        self.single_click_duration = single_click_duration
        self.mouse_sensitivity = float(mouse_sensitivity)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.deadzone = deadzone

        self.is_relative = is_relative
        self.relative_last_x = self.screen_width
        self.relative_last_y = self.screen_height
        self.acceleration_factor = acceleration_factor
        self.linear_factor = linear_factor

        self.left_down = False
        self.middle_down = False
        self.right_down = False
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
            * self.screen_width
        )
        y = int(
            (
                (self.y_scale * self.mouse_sensitivity) * y
                - (self.y_scale * self.mouse_sensitivity - 1) / 2 
            )
            * self.screen_height
        )

        if len(self.x_window) > 1:
            self.relative_last_x = self.x_window[len(self.x_window) - 1]
            self.relative_last_y = self.y_window[len(self.y_window) - 1]

        # Check if the movement is smaller than the specified radius
        last_x, last_y = pydirectinput.position()
        distance = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

        # Specify the radius distance
        ignore_small_movement = distance <= self.deadzone

        self.x_window.append(x)
        self.y_window.append(y)

        x = self.exponential_moving_average(self.x_window)
        y = self.exponential_moving_average(self.y_window)

        if len(self.x_window) > self.window_size:
            self.x_window.pop(0)
            self.y_window.pop(0)
            if mouse_button == "":
                if self.left_down:
                    self.console.print(f"releasing mouse left")
                    pydirectinput.mouseUp(button="left")
                    self.left_down = False
                if self.middle_down:
                    self.console.print(f"releasing mouse middle")
                    pydirectinput.mouseUp(button="middle")
                    self.middle_down = False
                if self.right_down:
                    self.console.print(f"releasing mouse right")
                    pydirectinput.mouseUp(button="right")
                    self.right_down = False
                elif not ignore_small_movement:
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
        if self.is_relative == True:
            # can't raise negative to an exponent
            x_diff = self.relative_last_x - x
            x_diff_abs = abs(x_diff * self.linear_factor)
            scaled_x = int(x_diff_abs**self.acceleration_factor) * (
                1 if x_diff >= 0 else -1
            )
            y_diff = self.relative_last_y - y
            y_diff_abs = abs(y_diff * self.linear_factor)
            scaled_y = int(y_diff_abs**self.acceleration_factor) * (
                1 if y_diff >= 0 else -1
            )
            pydirectinput.moveRel(scaled_x, scaled_y, relative=True)
        else:
            pydirectinput.moveTo(x, y)

    def click(self, x, y, mouse_button):
        """Handle mouse clicking.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            mouse_button (str): Mouse button to click.
        """
        mouse_down = self.left_down or self.middle_down or self.right_down
        if not mouse_down:
            self.console.print(f"clicking mouse {mouse_button}")
            pydirectinput.mouseDown(button=mouse_button)
            time.sleep(self.single_click_duration)

        if mouse_button == "left":
            if not self.left_down:
                pydirectinput.mouseDown(button=mouse_button)
                self.left_down = True

        if mouse_button == "middle":
            if not self.middle_down:
                pydirectinput.mouseDown(button=mouse_button)
                self.middle_down = True

        if mouse_button == "right":
            if not self.right_down:
                pydirectinput.mouseDown(button=mouse_button)
                self.right_down = True

        self.move(x, y)

    def exponential_moving_average(self, data):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(self.alpha * data[i] + (1 - self.alpha) * ema[i - 1])
        return int(ema[len(data) - 1])
