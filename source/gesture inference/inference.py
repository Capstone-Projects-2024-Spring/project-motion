# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html

# pygame-ce
import pygame
from GetHands import GetHands
from Mouse import Mouse
from Keyboard import Keyboard
import os
import Console
from menu import Menu
from Renderer import Renderer
from FlappyBird import flappybird
from EventHandler import GestureEventHandler
from platformerGame import platformer
from asteroids import asteroids
from tetris import tetris
from fruitninja import FNgame

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
    "move_mouse_flag": False,
    "run_model_flag": True,
    "gesture_model_path": "models/lstm/brawl.pth",
    "click_sense": 0.05,
    "hands": None,
    "running": True,
    "show_debug_text": True,
    "webcam_mode": 1,
    "toggle_mouse_key": "m",
    "min_confidence": 0.0,
    "gesture_list": [],     
    "mouse_hand_num": 1,
    "keyboard_hand_num": 0,
    "hand_1_keyboard": None,
    "hand_2_keyboard": None,
    "key_toggle_enabled": False,
}

def main() -> None:

    window_width = 800
    window_height = 800
    window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    mouse = Mouse(flags=flags)
    keyboard = Keyboard(
        threshold=0,
        flags=flags,
        bindings=["left", "right", "up", "down", "none", "none", "none", "none"],
        hand_num=0,
    ) 
    keyboard2 = Keyboard(threshold=0, flags=flags, bindings=["none", "none", "none", "none", "z", "x", "c", "v"], hand_num=1)

    hands = GetHands(flags=flags)
    flags["hands"] = hands

    flags["mouse"] = mouse
    flags["hand_1_keyboard"] = keyboard
    flags["hand_2_keyboard"] = keyboard2
    keyboard.start()
    keyboard2.start()
    mouse.start()
    hands.start()
    game_loop(window, hands)
    pygame.quit()


def game_loop(
    window: pygame.display,
    hands: GetHands,
):
    window_width, window_height = pygame.display.get_surface().get_size()
    """Runs the pygame event loop and renders surfaces"""

    game = None

    def set_game(num):
        nonlocal game
        if num == 0:
            game = None
        if num == 1:
            Console.print("Flappybird")
            game = flappybird.FlappyBirdGame()
        if num == 2:
            Console.print("Asteroids")
            game = asteroids
        if num == 3:
            Console.print("Platformer")
            game = platformer
        if num == 4:
            Console.print("Fruit Ninja")
            game = FNgame
        if num == 5:
            Console.print("Tetris")  
            game = tetris  
            
    menu = Menu(window_width, window_height, flags, set_game_func=set_game)

    event_handler = GestureEventHandler(menu, flags)

    font = pygame.font.Font("freesansbold.ttf", 30)
    renderer = Renderer(font, window, flags)

    main_menu = menu.main_menu

    clock = pygame.time.Clock()

   # tickrate = pygame.display.get_current_refresh_rate()
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

        if main_menu.is_enabled():
            main_menu.draw(window)
            main_menu.update(events)

        clock.tick(tickrate)

        pygame.display.update()


def game_events(game, events, window):
    if game != None:
        game.events(events)
        window.blit(game.surface, (0, 0))
        game.tick()


def print_input_table(counter):
    if counter % 4 == 0:
        keys = pygame.key.get_pressed()
        clicks = pygame.mouse.get_pressed()
        keyString = []

        for i in range(len(keys)):
            if keys[i]:
                keyName = pygame.key.name(i)
                keyString.append([keyName])
                
        clicked = []
        if clicks[0]:
            clicked.append(["left"])
        if clicks[1]:
            clicked.append(["middle"])
        if clicks[2]:
            clicked.append(["right"])
    
        Console.table(["key pressed (pygame)"], keyString, table_number=1)
        Console.table(["mouse (pygame)"], clicked, table_number=2)
        Console.update()


if __name__ == "__main__":
    main()
