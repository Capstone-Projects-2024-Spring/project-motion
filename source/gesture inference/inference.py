# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pydoc
import pygame
import pygame_menu
from GetHands import GetHands
from RenderHands import RenderHands
from Mouse import Mouse
from Keyboard import Keyboard
import os
from Console import GestureConsole

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
}

console = GestureConsole()

def main() -> None:
    """Main driver method which initilizes all children and starts pygame render pipeline"""

    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Test Hand Tracking Multithreaded")
 
    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0, 0, 0))
                              
    myRenderHands = RenderHands(hands_surface, render_scale=3)

    mouse_controls = Mouse(mouse_scale=2)

    keyboard = Keyboard(
        threshold=0, toggle_key_threshold=0.3, toggle_mouse_func=toggle_mouse
    )

    # control_mouse=mouse_controls.control,
    hands = GetHands(
        myRenderHands.render_hands,
        confidence=0.5,
        control_mouse=mouse_controls.control,
        flags=flags,
        keyboard=keyboard,
    )

    menu = pygame_menu.Menu(
        "Welcome",
        window_width * 0.8,
        window_height * 0.8,
        theme=pygame_menu.themes.THEME_BLUE,
    )

    menu.add.selector(
        "Render Mode :", [("Normalized", True), ("World", False)], onchange=set_coords
    )

    menu.add.button("Close Menu", pygame_menu.events.CLOSE)
    menu.add.button("Turn On Model", action=toggle_model)
    menu.add.button("Turn On Mouse", action=toggle_mouse)
    menu.add.button("Quit", pygame_menu.events.EXIT)
    menu.enable()

    print("game loop")
    game_loop(
        window,
        window_width,
        window_height,
        hands,
        hands_surface,
        menu,            
    )

    pygame.quit()


def toggle_mouse() -> None:
    """Enable or disable mouse control"""
    console.print("toggling mouse control")
    flags["move_mouse_flag"] = not flags["move_mouse_flag"]


def toggle_model() -> None:
    console.print("toggling model")
    flags["run_model_flag"] = not flags["run_model_flag"]

def set_coords(value, mode) -> None:
    """Defines the coordinate space for rendering hands

    Args:
        value (_type_): used by pygame_menu
        mode (_type_): True for normalized, False for world
    """
    flags["render_hands_mode"] = mode

def game_loop(
    window,
    window_width,
    window_height,
    hands:GetHands,
    hands_surface,
    menu,
):
    """Runs the pygame event loop and renders surfaces

    Args:
        window (_type_): The main pygame window
        window_width (_type_): Width of the pygame window
        window_height (_type_): Height of the pygame window
        hands (_type_): The GetHands class
        hands_surface (_type_): The surface that the hands are rendered on
        menu (_type_): the main menu
        gesture_list (_type_): the list of recognized gestures
        gesture_vector (_type_): one hot encoded binary vector for writing the correct label output to csv
    """
    hands.start()
    running = True

    while running:
        window.fill((0, 0, 0))
        events = pygame.event.get()
        for event in events: 
            if event.type == pygame.QUIT:
                hands.stop()
                running = False
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_m:
                    toggle_mouse()

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        # index = hands.confidence_vector
        # gesture_text = font.render(hands.gesture_list[index], False, (255, 255, 255))

        # window.blit(gesture_text, (window_width - window_width // 5, 0))
        window.blit(hands_surface, (0, 0))
        window.blit(fps, (0, 0))

        clock.tick(60)
        pygame.display.update()

if __name__ == "__main__":
    main()
