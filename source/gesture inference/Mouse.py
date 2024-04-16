import sys
if sys.platform == 'win32':
    import pydirectinput as pyinput
else:
    import pyautogui as pyinput
import time
import Console
import math
import threading


class Mouse(threading.Thread):
    def __init__(
        self,
        mouse_sensitivity=1,
        x_scale=1.3,
        y_scale=1.5,
        alpha=0.15,
        deadzone=15,
        single_click_duration=1 / 5,
        is_relative=True,
        acceleration_factor=1.5,
        linear_factor=0.25,
        flags=None,
        tps=120,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
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
        self.flags = flags
        self.screen_width, self.screen_height = pyinput.size()
        pyinput.FAILSAFE = False
        pyinput.PAUSE = False
        self.single_click_duration = single_click_duration
        self.mouse_sensitivity = float(mouse_sensitivity)
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.deadzone = deadzone

        self.tps = tps

        self.is_relative = is_relative
        self.relative_last_x = self.screen_width
        self.relative_last_y = self.screen_height
        self.acceleration_factor = acceleration_factor
        self.linear_factor = linear_factor

        self.left_down = False
        self.middle_down = False
        self.right_down = False

        # expontial moving average stuff
        self.x_window = []
        self.y_window = []
        self.window_size = 12
        self.alpha = alpha

    def toggle_mouse(self):
        self.flags["move_mouse_flag"] = not self.flags["move_mouse_flag"]

    def run(self):
        event = threading.Event()
        while True:
            if self.flags["move_mouse_flag"] and self.flags["hands"].location != []:
                mouse_button_text = ""
                hands = self.flags["hands"].result.hand_world_landmarks
                confidences = self.flags["hands"].confidence_vectors
                # check for race condition
                if (
                    len(hands) > self.flags["mouse_hand_num"]
                    and len(confidences) > self.flags["mouse_hand_num"]
                ):
                    hand = hands[self.flags["mouse_hand_num"]]

                    # index tip, thumb tip
                    if self.is_clicking(hand[8], hand[4], self.flags["click_sense"]):
                        mouse_button_text = "left"
                    # middle tip, thumb tip
                    elif self.is_clicking(hand[12], hand[4], self.flags["click_sense"]):
                        mouse_button_text = "middle"
                    # ring tip, thumb tip
                    elif self.is_clicking(hand[16], hand[4], self.flags["click_sense"]):
                        mouse_button_text = "right"

                    location = self.flags["hands"].location[
                        self.flags["mouse_hand_num"]
                    ]

                    self.control(location[0], location[1], mouse_button_text)
            else:
                self.lift_mouse_button()

            event.wait(timeout=1 / self.tps)

    def lift_mouse_button(self):
        if self.left_down:
            Console.print(f"releasing mouse left")
            pyinput.mouseUp(button="left")
            self.left_down = False
        if self.middle_down:
            Console.print(f"releasing mouse middle")
            pyinput.mouseUp(button="middle")
            self.middle_down = False
        if self.right_down:
            Console.print(f"releasing mouse right")
            pyinput.mouseUp(button="right")
            self.right_down = False

    def scale(self, x, y):
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
        return x, y

    def control(self, x: float, y: float, mouse_button: str):
        """Moves the mouse to XY coordinates and can perform single clicks, or click and drags when called repeatelly

        Args:
            x (float): x coordinate between 0 and 1
            y (float): y coordinate between 0 and 1
            mouse_button (string): can be "", "left", "middle", or "right"
        """

        x, y = self.scale(x, y)

        if len(self.x_window) > 1:
            self.relative_last_x = self.x_window[len(self.x_window) - 1]
            self.relative_last_y = self.y_window[len(self.y_window) - 1]

        # Check if the movement is smaller than the specified radius
        last_x, last_y = pyinput.position()
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
                self.lift_mouse_button()
            else:
                self.click(mouse_button)
            if not ignore_small_movement:
                self.move(x, y)

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
            y_diff = self.relative_last_y - y
            if x_diff > 300 or y_diff > 300:
                return

            x_diff_abs = abs(x_diff * self.linear_factor)
            scaled_x = int(x_diff_abs**self.acceleration_factor) * (
                1 if x_diff >= 0 else -1
            )
            y_diff_abs = abs(y_diff * self.linear_factor)
            scaled_y = int(y_diff_abs**self.acceleration_factor) * (
                1 if y_diff >= 0 else -1
            )
            if sys.platform == 'win32':
                pyinput.moveRel(scaled_x, scaled_y, relative=True)
            else:
                pyinput.moveRel(scaled_x, scaled_y)
        else:
            pyinput.moveTo(x, y)

    def click(self, mouse_button):
        """Handle mouse clicking.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            mouse_button (str): Mouse button to click.
        """
        if mouse_button == "left":
            if not self.left_down:
                Console.print(f"clicking mouse {mouse_button}")
                pyinput.mouseDown(button=mouse_button)
                self.left_down = True

        if mouse_button == "middle":
            if not self.middle_down:
                Console.print(f"clicking mouse {mouse_button}")
                # pyinput.mouseDown(button=mouse_button)
                pyinput.scroll(-1)
                self.middle_down = True

        if mouse_button == "right":
            if not self.right_down:
                Console.print(f"clicking mouse {mouse_button}")
                pyinput.mouseDown(button=mouse_button)
                self.right_down = True

    def exponential_moving_average(self, data):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(self.alpha * data[i] + (1 - self.alpha) * ema[i - 1])
        return int(ema[len(data) - 1])
