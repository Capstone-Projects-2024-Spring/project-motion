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
    running = True
    is_menu_showing = True
    webcam_mode = 1
    show_debug_text = True
    is_fullscreen = False
    window_width, window_height = pygame.display.get_surface().get_size()
    menu_pygame = menu.menu
    renderer = Renderer(font, window, flags)

    while running:

        # changing GetHands parameters creates a new hands object
        if flags["hands"] != None and hands != flags["hands"]:
            hands = flags["hands"]


        window_width, window_height = pygame.display.get_surface().get_size()

        window.fill((0, 0, 0))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            console.table(["key pressed"], [["space"]], table_number=1)
        else:
            console.table(["key pressed"], [[""]], table_number=1)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                hands.stop()
                hands.join()
                running = False
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_m:
                    menu.toggle_mouse()

                if event.key == pygame.K_ESCAPE:
                    if is_menu_showing:
                        is_menu_showing = False
                        menu_pygame.disable()
                    else:
                        is_menu_showing = True
                        menu_pygame.enable()

                if event.key == pygame.K_F1:
                    webcam_mode += 1

                if event.key == pygame.K_F2:
                    show_debug_text = not show_debug_text

                if event.key == pygame.K_F3:
                    flags["render_hands_mode"] = not flags["render_hands_mode"]

                if event.key == pygame.K_F11:
                    is_fullscreen = not is_fullscreen
                    pygame.display.toggle_fullscreen()

                if event.key == pygame.K_m:
                    keyboard.press("m")
       
                if event.key == pygame.K_g:
                    flags["run_model_flag"] = not flags["run_model_flag"]

        location = hands.location.copy()

        if flags["move_mouse_flag"] and location != []:
            mouse_button_text = ""
            hand = hands.result.hand_world_landmarks[0]
            if mouse.is_clicking(hand[8], hand[4], flags["click_sense"]):
                mouse_button_text = "left"
            location = location[0]
            mouse.control(location[0], location[1], mouse_button_text)

        if flags["run_model_flag"] and len(hands.confidence_vectors) > 0:
            # send only the first hand confidence vector the gesture model output
            keyboard.gesture_input(hands.confidence_vectors[0])
        else:
            keyboard.release()

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        frame = hands.frame.copy()
        renderer.render_webcam(frame, webcam_mode)
        renderer.render_hands(hands)
        renderer.render_debug_text(show_debug_text,hands,fps)

        if menu_pygame.is_enabled():
            menu_pygame.update(events)
            menu_pygame.draw(window)

        # clock.tick(pygame.display.get_current_refresh_rate())
        clock.tick(60)
        pygame.display.update()


if __name__ == "__main__":
    main()
