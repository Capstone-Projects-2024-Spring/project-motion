import pygame_menu
from GetHands import GetHands
import Console
from typing import Callable
from functools import partial
import csv

import sys
from os import path, chdir, listdir
bundle_dir = path.dirname(path.abspath(sys.argv[0]))
chdir(bundle_dir)
class Menu:
    def __init__(
        self, window_width, window_height, flags, set_game_func: Callable = None
    ):

        self.gesture_settings = pygame_menu.Menu(
            "Esc to toggle menu",
            window_width * 0.8,
            window_height * 0.8,
            theme=pygame_menu.themes.THEME_BLUE,
            menu_id="settings",
        )
        
        self.keybinds = []
        with open('keybinds.csv', mode ='r') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                self.keybinds.append((line[0], [key for key in line[1:]]))
                
        self.flags = flags
        self.toggle_mouse_key = flags["toggle_mouse_key"]
        self.models_folder_ff = "models/feedforward/"
        self.models_folder_lstm = "models/lstm/"
        self.setup_settings()
        self.gesture_settings.disable()


        self.set_game_func = set_game_func
        # self.set_game_func = self.configure_game_settings
        self.main_menu = pygame_menu.Menu(
            "Esc to toggle menu",
            window_width * 0.8,
            window_height * 0.8,
            theme=pygame_menu.themes.THEME_BLUE,
        )
        self.setup_main()
        self.main_menu.enable()

    def configure_game_settings(self, game_id):
        if game_id == 1:
            self.change_hands_num(["1", 1])
            self.set_mouse_hand(0, 0)
            self.set_key_1_bindings(0, ["space", "none", "m", "p"])
            self.set_key_2_bindings(4, ["space", "none", "m", "p"])
            self.change_gesture_model_ff([("flappy.pth",)])
            self.lockout_mouse(True)

        elif game_id == 2:
            self.change_hands_num(["2", 2])
            self.set_mouse_hand(0, 0)
            self.set_key_1_bindings(0, ["left", "up", "right", "none"])
            self.set_key_2_bindings(4, ["none", "none", "none", "space"])
            self.change_gesture_model_ff([("games.pth",)])
            self.lockout_mouse(True)

        elif game_id == 3:
            self.change_hands_num(["3", 3])
            self.set_mouse_hand(0, 0)
            self.set_key_1_bindings(0, ["left", "none", "right", "space"])
            self.set_key_2_bindings(4, ["left", "none", "right", "space"])
            self.change_gesture_model_ff([("jumpy.pth",)])
            self.lockout_mouse(True)

        elif game_id == 5:
            self.change_hands_num(["1", 1])
            self.set_mouse_hand(0, 0)
            self.set_key_1_bindings(0, ["up", "left", "right", "none", "down"])
            self.set_key_2_bindings(4, ["none"])
            self.change_gesture_model_ff([("tetris.pth",)])
            self.lockout_mouse(True)

        elif game_id == 0:
            self.change_hands_num(["0", 0])
            self.set_mouse_hand(0, 0)
            self.set_key_1_bindings(0, ["none"])
            self.set_key_2_bindings(4, ["none"])
            self.change_gesture_model_ff([("motion.pth",)])
            self.lockout_mouse(True)

        if self.set_game_func:
            self.set_game_func(game_id)

    def setup_main(self):
        self.main_menu.add.button(
            "Flappybird", action=partial(self.configure_game_settings, 1)
        )
        self.main_menu.add.button(
            "Asteroids", action=partial(self.configure_game_settings, 2)
        )
        self.main_menu.add.button(
            "Platformer", action=partial(self.configure_game_settings, 3)
        )
        self.main_menu.add.button(
            "Tetris", action=partial(self.configure_game_settings, 5)
        )
        self.main_menu.add.button(
            "No game", action=partial(self.configure_game_settings, 0)
        )

        link = self.main_menu.add.menu_link(self.gesture_settings, "settings")
        self.main_menu.add.button("Settings", action=link.open)
        self.main_menu.add.button("Close", action=self.main_menu.disable)
        self.main_menu.add.button("Quit", pygame_menu.events.EXIT)

    def setup_settings(self):
        self.gesture_settings.add.dropselect(
            "Camera:",
            self.flags["hands"].camera.menu_selector,
            default=0,
            onchange=self.set_camera,
        )
        self.gesture_settings.add.selector(
            "Render Mode :",
            [("Normalized", True), ("World", False)],
            onchange=self.set_coords,
        )
        self.gesture_settings.add.selector(
            "Mouse Smoothing :",
            [("None", 1), ("Low", 2), ("Medium", 6), ("High", 12), ("Max", 24)],
            default=3,
            onchange=self.change_mouse_smooth,
        )
        self.gesture_settings.add.dropselect(
            "Number of hands :",
            ["1", "2", "3", "4"],
            default=1,
            onchange=self.change_hands_num,
        )

        self.gesture_settings.add.dropselect(
            "Mouse Hand:",
            [("Hand 1", 0), ("Hand 2", 1)],
            default=1,
            onchange=self.set_mouse_hand,
        )
        # ["fist", "forwards", "backwards", "thumb", "pinky", "peace", "wave"]
        self.gesture_settings.add.dropselect(
            "Hand 1 Key bindings:",self.keybinds,
            onchange=self.set_key_1_bindings,
        )
        self.gesture_settings.add.dropselect(
            "Hand 2 Key bindings:",self.keybinds,
            onchange=self.set_key_2_bindings,
        )
        self.gesture_settings.add.dropselect(
            "Use FF Model :",
            self.find_files_with_ending(".pth", directory_path=self.models_folder_ff),
            onchange=self.change_gesture_model_ff,
        )
        models_lstm = self.find_files_with_ending(
            ".pth", directory_path=self.models_folder_lstm
        )
        self.gesture_settings.add.dropselect(
            "Use LSTM Model :", models_lstm, onchange=self.change_gesture_model_lstm
        )

        self.gesture_settings.add.range_slider(
            "Click Sensitivity",
            default=70,
            range_values=(1, 150),
            increment=1,
            onchange=self.set_click_sense,
        )

        self.gesture_settings.add.toggle_switch(
            "Enable Model", True, onchange=self.toggle_model
        )
        self.gesture_settings.add.button(
            "Toggle Mouse", action=self.flags["mouse"].toggle_mouse
        )
        self.gesture_settings.add.toggle_switch(
            "Disable Mouse Control", False, onchange=self.lockout_mouse
        )
        self.gesture_settings.add.toggle_switch(
            "Mouse Relative Mode", True, onchange=self.mouse_relative
        )
        self.gesture_settings.add.toggle_switch(
            "Toggle Keys", False, onchange=self.toggle_keys
        )
        self.gesture_settings.add.toggle_switch(
            "Enable Console", True, onchange=self.enable_console
        )

    def set_camera(self, value, cam):
        self.flags["hands"].camera.stop()
        self.flags["hands"].camera.start(cam)

    def set_key_1_bindings(self, value, keys):
        self.flags["hand_1_keyboard"].bindings = keys

    def set_key_2_bindings(self, value, keys):
        self.flags["hand_2_keyboard"].bindings = keys

    def set_mouse_hand(self, value, num):
        self.flags["mouse_hand_num"] = num

    def mouse_relative(self, current_state_value, **kwargs):
        self.flags["mouse"].is_relative = current_state_value

    def toggle_keys(self, current_state_value, **kwargs):
        self.flags["key_toggle_enabled"] = current_state_value

    def enable_console(self, current_state_value, **kwargs):
        Console.printing = current_state_value

    def lockout_mouse(self, current_state_value, **kwargs):
        if current_state_value:
            self.flags["toggle_mouse_key"] = None
            self.flags["move_mouse_flag"] = False
        else:
            self.flags["toggle_mouse_key"] = self.toggle_mouse_key

    def find_files_with_ending(self, ending: str, directory_path=""):
        """returns a list of tuples of the strings found"""
        files = [(file,) for file in listdir(directory_path) if file.endswith(ending)]
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

    def change_gesture_model_ff(self, value):
        self.flags["gesture_model_path"] = (
            self.models_folder_ff + value[0][0]
        )  # tuple within a list for some reason
        self.flags["hands"].stop()
        self.flags["hands"].join()
        self.flags["hands"] = GetHands(flags=self.flags)
        self.flags["hands"].start()

    def change_gesture_model_lstm(self, value):
        self.flags["gesture_model_path"] = (
            self.models_folder_lstm + value[0][0]
        )  # tuple within a list for some reason
        self.flags["hands"].stop()
        self.flags["hands"].join()
        self.flags["hands"] = GetHands(flags=self.flags)
        self.flags["hands"].start()

    def set_click_sense(self, value, **kwargs):
        self.flags["click_sense"] = value / 1000
