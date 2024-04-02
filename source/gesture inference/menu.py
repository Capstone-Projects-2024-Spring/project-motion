import pygame_menu
import os
from GetHands import GetHands

class Menu:
    def __init__(self, window_width, window_height, flags):
        self.menu = pygame_menu.Menu(
            "Esc to toggle menu",
            window_width * 0.5,
            window_height * 0.5,
            theme=pygame_menu.themes.THEME_BLUE,
        )
        self.flags = flags
        self.setup_menu()

    def setup_menu(self):
        self.menu.add.selector(
            "Render Mode :", [("Normalized", True), ("World", False)], onchange=self.set_coords
        )
        self.menu.add.selector(
            "Mouse Smoothing :",
            [("None", 1), ("Low", 2), ("Medium", 6), ("High", 12), ("Max", 24)],
            default=3,
            onchange=self.change_mouse_smooth,
        )
        self.menu.add.dropselect(
            "Number of hands :", ["1", "2", "3", "4"], onchange=self.change_hands_num
        )

        models = self.find_files_with_ending(".pth")
        self.menu.add.dropselect("Use Gesture Model :", models, onchange=self.change_gesture_model)

        self.menu.add.range_slider(
            "Click Sensitivity",
            default=70,
            range_values=(1, 150),
            increment=1,
            onchange=self.set_click_sense,
        )

        self.menu.add.button("Turn On Model", action=self.toggle_model)
        self.menu.add.button("Turn On Mouse", action=self.toggle_mouse)
        self.menu.add.button("Quit", pygame_menu.events.EXIT)
        self.menu.enable()

    def find_files_with_ending(self, ending: str, directory_path=os.getcwd()):
        """returns a list of tuples of the strings found"""
        files = [(file,) for file in os.listdir(directory_path) if file.endswith(ending)]
        return files

    def toggle_mouse(self):
        """Enable or disable mouse control"""
        self.flags["move_mouse_flag"] = not self.flags["move_mouse_flag"]

    def toggle_model(self):
        self.flags["run_model_flag"] = not self.flags["run_model_flag"]

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
        self.flags["gesture_model_path"] = value[0][0]  # tuple within a list for some reason
        self.flags["hands"].stop()
        self.flags["hands"].join()
        self.flags["hands"] = GetHands(flags=self.flags)
        self.flags["hands"].start()

    def set_click_sense(self, value, **kwargs):
        self.flags["click_sense"] = value / 1000