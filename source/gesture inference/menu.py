import pygame_menu
import os
from GetHands import GetHands
from Console import GestureConsole


class Menu:
    def __init__(self, window_width, window_height, flags):
        self.menu = pygame_menu.Menu(
            "Esc to toggle menu",
            window_width * 0.8,
            window_height * 0.8,
            theme=pygame_menu.themes.THEME_BLUE,
        )
        self.flags = flags
        self.toggle_mouse_key = flags["toggle_mouse_key"]
        self.models_folder = "models/"
        self.console = GestureConsole()
        self.setup_menu()

    def setup_menu(self):
        self.menu.add.selector(
            "Render Mode :",
            [("Normalized", True), ("World", False)],
            onchange=self.set_coords,
        )
        self.menu.add.selector(
            "Mouse Smoothing :",
            [("None", 1), ("Low", 2), ("Medium", 6), ("High", 12), ("Max", 24)],
            default=3,
            onchange=self.change_mouse_smooth,
        )
        self.menu.add.dropselect(
            "Number of hands :",
            ["1", "2", "3", "4"],
            default=1,
            onchange=self.change_hands_num,
        )

        self.menu.add.dropselect(
            "Keyboard Hand:",
            [("Hand 1", 0), ("Hand 2", 1)],
            default=0,
            onchange=self.set_keyboard_hand,
        )

        self.menu.add.dropselect(
            "Mouse Hand:",
            [("Hand 1", 0), ("Hand 2", 1)],
            default=1,
            onchange=self.set_mouse_hand,
        )
        self.menu.add.dropselect(
            "Key bindings:",
            [
                ("Flappybird", ["space", "none", "m", "p"]),
                ("Minecraft", ["none", "w", "e", "ctrlleft", "space"]),
            ],
            default=1,
            onchange=self.set_key_bindings,
        )
        models = self.find_files_with_ending(".pth", directory_path=self.models_folder)
        self.menu.add.dropselect(
            "Use Gesture Model :", models, onchange=self.change_gesture_model
        )

        self.menu.add.range_slider(
            "Click Sensitivity",
            default=70,
            range_values=(1, 150),
            increment=1,
            onchange=self.set_click_sense,
        )

        self.menu.add.toggle_switch("Enable Model", True, onchange=self.toggle_model)
        self.menu.add.button("Toggle Mouse", action=self.flags["mouse"].toggle_mouse)
        self.menu.add.toggle_switch(
            "Disable Mouse Control", False, onchange=self.lockout_mouse
        )
        self.menu.add.toggle_switch(
            "Mouse Relative Mode", True, onchange=self.mouse_relative
        )
        self.menu.add.toggle_switch(
            "Enable Console", True, onchange=self.enable_console
        )
        self.console
        self.menu.add.button("Quit", pygame_menu.events.EXIT)
        self.menu.enable()

    def set_key_bindings(self, value, keys):
        self.flags["key_bindings"] = keys

    def set_mouse_hand(self, value, num):
        self.flags["mouse_hand_num"] = num

    def set_keyboard_hand(self, value, num):
        self.flags["keyboard_hand_num"] = num

    def mouse_relative(self, current_state_value, **kwargs):
        self.flags["mouse"].is_relative = current_state_value

    def enable_console(self, current_state_value, **kwargs):
        self.console.printing = current_state_value

    def lockout_mouse(self, current_state_value, **kwargs):
        if current_state_value:
            self.flags["toggle_mouse_key"] = None
            self.flags["move_mouse_flag"] = False
        else:
            self.flags["toggle_mouse_key"] = self.toggle_mouse_key

    def find_files_with_ending(self, ending: str, directory_path=os.getcwd()):
        """returns a list of tuples of the strings found"""
        files = [
            (file,) for file in os.listdir(directory_path) if file.endswith(ending)
        ]
        return files

    def toggle_model(self, current_state_value, **kwargs):
        self.flags["run_model_flag"] = current_state_value

    def set_coords(self, value, mode):
        """Defines the coordinate space for rendering hands"""
        self.flags["render_hands_mode"] = mode

    def change_mouse_smooth(self, value, smooth):
        self.flags["mouse"].x_window = []
        self.flags["mouse"].y_window = []
        self.flags["mouse"].window_size = smooth

    def change_hands_num(self, value):
        self.flags["number_of_hands"] = value[1] + 1
        self.flags["hands"].stop()
        self.flags["hands"].join()
        self.flags["hands"] = GetHands(flags=self.flags)
        self.flags["hands"].start()

    def change_gesture_model(self, value):
        self.flags["gesture_model_path"] = (
            self.models_folder + value[0][0]
        )  # tuple within a list for some reason
        self.flags["hands"].stop()
        self.flags["hands"].join()
        self.flags["hands"] = GetHands(flags=self.flags)
        self.flags["hands"].start()

    def set_click_sense(self, value, **kwargs):
        self.flags["click_sense"] = value / 1000
