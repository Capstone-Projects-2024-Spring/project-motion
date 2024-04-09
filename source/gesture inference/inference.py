# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html

# pygame-ce
import pygame
from GetHands import GetHands
from Mouse import Mouse
from Keyboard import Keyboard
import os
from Console import GestureConsole
from menu import Menu
from Renderer import Renderer
from FlappyBird import flappybird
from EventHandler import GestureEventHandler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# global variables
pygame.init()

# communication object for gesture stuff
flags = {
    "render_hands_mode": True,
    "gesture_vector": [],
    "number_of_hands": 2,
    "move_mouse_flag": True,
    "run_model_flag": True,
    "gesture_model_path": "models/flappy.pth",
    "click_sense": 0.05,
    "hands": None,
    "running": True,
    "show_debug_text": True,
    "webcam_mode": 2,
    "toggle_mouse_key": "m",
    "min_confidence": 0.0,
    "gesture_list": [],
    "mouse_hand_num": 1,
    "keyboard_hand_num": 0,
    "key_bindings": ["space", "none", "m", "p"],
}

# custom console
console = GestureConsole()


def main() -> None:

    window_width = 800
    window_height = 800
    window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    mouse = Mouse(flags=flags)
    keyboard = Keyboard(
        threshold=0,
        flags=flags,
    )

    hands = GetHands(flags=flags)
    flags["hands"] = hands

    flags["mouse"] = mouse
    flags["keyboard"] = keyboard
    keyboard.start()
    mouse.start()

    menu = Menu(window_width, window_height, flags)

    event_handler = GestureEventHandler(menu, flags)

    game_loop(window, hands, event_handler, menu)
    pygame.quit()


def game_loop(
    window: pygame.display,
    hands: GetHands,
    event_handler: GestureEventHandler,
    menu: Menu,
):
    """Runs the pygame event loop and renders surfaces"""
    hands.start()

    font = pygame.font.Font("freesansbold.ttf", 30)
    renderer = Renderer(font, window, flags)
    menu_pygame = menu.menu

    clock = pygame.time.Clock()

    flappy_window_width = 864
    flappy_window_height = 936

    game_surface = pygame.Surface((flappy_window_width, flappy_window_height))
    game = flappybird.FlappyBirdGame(
        game_surface, flappy_window_width, flappy_window_height
    )

    tickrate = pygame.display.get_current_refresh_rate()
    tickrate = 60

    counter = 0

    while flags["running"]:
        counter += 1
        # changing number of hands creates a new hands object
        if flags["hands"] != None and hands != flags["hands"]:
            hands = flags["hands"]
        window_width, window_height = pygame.display.get_surface().get_size()
        window.fill((0, 0, 0))

        events = pygame.event.get()

        event_handler.handle_events(events)

        game_events(game, events, window)

        renderer.render_overlay(hands, clock)
        print_input_table(counter)
        if menu_pygame.is_enabled():
            menu_pygame.update(events)
            menu_pygame.draw(window)

        clock.tick(tickrate)

        pygame.display.update()


def game_events(game, events, window):
    game.events(events)
    window.blit(game.surface, (0, 0))
    game.tick()


def print_input_table(counter):
    if counter % 7 == 0:
        keys = pygame.key.get_pressed()
        clicks = pygame.mouse.get_pressed()
        keyString = []

        for i in range(len(keys)):
            if keys[i]:
                keyName = pygame.key.name(i)
                keyString.append([keyName])
        console.table(["key pressed (pygame)"], keyString, table_number=1)

        clicked = []
        if clicks[0]:
            clicked.append(["left"])
        if clicks[1]:
            clicked.append(["middle"])
        if clicks[2]:
            clicked.append(["right"])

        console.table(["mouse (pygame)"], clicked, table_number=2)
        console.update()


if __name__ == "__main__":
    main()
