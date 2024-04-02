# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html

# pygame-ce
import pygame
from GetHands import GetHands
from RenderHands import RenderHands
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
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

flags = {
    "render_hands_mode": True,
    "gesture_vector": [],
    "number_of_hands": 2,
    "move_mouse_flag": False,
    "run_model_flag": True,
    "gesture_model_path": "simple.pth",
    "click_sense": 0.05,
    "hands": None,
    "running": True,
}

console = GestureConsole()


def main() -> None:
    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    mouse = Mouse()
    hands = GetHands(flags=flags)
    flags["mouse"] = mouse
    flags["hands"] = hands

    menu = Menu(window_width, window_height, flags)
    keyboard = Keyboard(
        threshold=0, toggle_key_threshold=0.3, toggle_mouse_func=menu.toggle_mouse
    )

    game_loop(window, hands, menu, mouse, keyboard)

    pygame.quit()


def game_loop(
    window: pygame.display,
    hands: GetHands,
    menu: Menu,
    mouse: Mouse,
    keyboard: Keyboard,
):
    """Runs the pygame event loop and renders surfaces"""
    hands.start()
    window_width, window_height = pygame.display.get_surface().get_size()
    renderer = Renderer(font, window, flags)
    menu_pygame = menu.menu

    flappy_surface = pygame.Surface((window_width, window_height))
    flappy_game = flappybird.FlappyBirdGame(flappy_surface, window_width, window_height)
    primitives = flappy_game.primitives

    event_handler = GestureEventHandler(hands, menu, mouse, keyboard, flags)

    while flags["running"]:
        # changing GetHands parameters creates a new hands object
        if flags["hands"] != None and hands != flags["hands"]:
            hands = flags["hands"]

        window_width, window_height = pygame.display.get_surface().get_size()
        window.fill((0, 0, 0))


        print_keyboard_table(pygame.key.get_pressed())

        events = pygame.event.get()
        event_handler.handle_events(events, window, renderer, clock, font)

        if flags["run_model_flag"] and len(hands.confidence_vectors) > 0:
            # send only the first hand confidence vector the gesture model output
            keyboard.gesture_input(hands.confidence_vectors[0])
        else:
            keyboard.release()

        if flags["move_mouse_flag"] and hands.location != []:
            event_handler.handle_mouse_control()

        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flags["running"] = False
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
                and primitives["is flying"] == False
                and primitives["is game over"] == False
            ):
                primitives["is flying"] = True
                primitives["is started"] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    primitives["is paused"] = not primitives["is paused"]
                    
        window.blit(flappy_game.surface, (0, 0))
        flappy_game.tick()
        
        frame = hands.frame.copy()
        renderer.render_webcam(frame, event_handler.webcam_mode)
        renderer.render_hands(hands)
        renderer.render_debug_text(event_handler.show_debug_text, hands, fps)

        if menu_pygame.is_enabled():
            menu_pygame.update(events)
            menu_pygame.draw(window)

        
        clock.tick(60)
        pygame.display.update()


def print_keyboard_table(keys):
    if keys[pygame.K_SPACE]:
        console.table(["key pressed"], [["space"]], table_number=1)
    else:
        console.table(["key pressed"], [[""]], table_number=1)


if __name__ == "__main__":
    main()
