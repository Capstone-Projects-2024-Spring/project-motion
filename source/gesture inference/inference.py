# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pydoc

# this is the community edition of pygame
# pygame-ce
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
    "render_hands_mode": False,
    "gesture_vector": [],
    "number_of_hands": 2,
    "move_mouse_flag": False,
    "run_model_flag": True,
    "gesture_model_path": "motion.pth",
    "hands": None,
}

console = GestureConsole()


def main() -> None:
    """Main driver method which initilizes all children and starts pygame render pipeline"""

    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0, 0, 0))

    myRenderHands = RenderHands(hands_surface, render_scale=3)

    mouse_controls = Mouse(mouse_sensitivity=2)

    keyboard = Keyboard(
        threshold=0, toggle_key_threshold=0.3, toggle_mouse_func=toggle_mouse
    )

    # control_mouse=mouse_controls.control,
    hands = GetHands(
        myRenderHands.render_hands,
        control_mouse=mouse_controls.control,
        flags=flags,
        keyboard=keyboard,
    )

    menu = pygame_menu.Menu(
        "Esc to toggle menu",
        window_width * 0.5,
        window_height * 0.5,
        theme=pygame_menu.themes.THEME_BLUE,
    )

    menu.add.selector(
        "Render Mode :", [("World", False), ("Normalized", True)], onchange=set_coords
    )

    def change_mouse_smooth(value, smooth):
        nonlocal mouse_controls
        mouse_controls.x_window = []
        mouse_controls.y_window = []
        mouse_controls.window_size = smooth

    menu.add.selector(
        "Mouse Smoothing :",
        [("None", 1), ("Low", 2), ("Medium", 6), ("High", 12), ("Max", 24)],
        onchange=change_mouse_smooth,
    )

    def change_hands_num(value):
        flags["number_of_hands"] = value[1] + 1
        build_hands()

    menu.add.dropselect(
        "Number of hands :", ["1", "2", "3", "4"], onchange=change_hands_num
    )

    models = find_files_with_ending(".pth")

    def change_gesture_model(value):
        flags["gesture_model_path"] = value[0][0] #tuple within a list for some reason
        build_hands()

    menu.add.dropselect("Use Gesture Model :", models, onchange=change_gesture_model)

    def build_hands():
        nonlocal hands
        nonlocal mouse_controls
        nonlocal keyboard
        hands.stop()
        hands.join()
        hands = GetHands(
            myRenderHands.render_hands,
            control_mouse=mouse_controls.control,
            flags=flags,
            keyboard=keyboard,
        )
        flags["hands"] = hands
        hands.start()

    menu.add.button("Close Menu", pygame_menu.events.CLOSE)
    menu.add.button("Turn On Model", action=toggle_model)
    menu.add.button("Turn On Mouse", action=toggle_mouse)
    menu.add.button("Quit", pygame_menu.events.EXIT)
    menu.enable()

    game_loop(
        window,
        hands,
        hands_surface,
        menu,
        mouse_controls,
    )

    pygame.quit()


def find_files_with_ending(ending: str, directory_path=os.getcwd()):
    files = [(file,) for file in os.listdir(directory_path) if file.endswith(ending)]
    return files


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
    window: pygame.display,
    hands: GetHands,
    hands_surface: pygame.Surface,
    menu: pygame_menu.Menu,
    mouse_controls: Mouse,
):
    """Runs the pygame event loop and renders surfaces

    Args:
        window (_type_): The main pygame window
        hands (_type_): The GetHands class
        hands_surface (_type_): The surface that the hands are rendered on
        menu (_type_): the main menu
    """
    hands.start()
    running = True
    is_menu_showing = True
    is_webcam_fullscreen = False

    is_fullscreen = False

    while running:

        # changing GetHands parameters creates a new hands object
        if flags["hands"] != None and hands != flags["hands"]:
            hands = flags["hands"]

        window_width, window_height = pygame.display.get_surface().get_size()
        window.fill((0, 0, 0))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                hands.stop()
                hands.join()
                running = False
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_m:
                    toggle_mouse()

                if event.key == pygame.K_ESCAPE:
                    if is_menu_showing:
                        is_menu_showing = False
                        menu.disable()
                    else:
                        is_menu_showing = True
                        menu.enable()

                if event.key == pygame.K_F1:
                    is_webcam_fullscreen = not is_webcam_fullscreen

                if event.key == pygame.K_F11:
                    is_fullscreen = not is_fullscreen
                    pygame.display.toggle_fullscreen()

        location = hands.mouse_location.copy()
        if len(location) == 1:
            # console.print(location)
            location = location[0]
            mouse_controls.control(location[0], location[1], hands.click)

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        frame = hands.frame.copy()
        img_pygame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
        img_width = img_pygame.get_width()
        img_height = img_pygame.get_height()

        hand_surface_copy = pygame.transform.scale(
            hands_surface.copy(), (img_width * 0.5, img_height * 0.5)
        )
        img_pygame = pygame.transform.scale(
            img_pygame, (img_width * 0.5, img_height * 0.5)
        )

        if is_webcam_fullscreen:
            img_pygame = pygame.transform.scale(
                img_pygame, (window_width, window_height)
            )
            hand_surface_copy = pygame.transform.scale(
                hands_surface.copy(), (window_width, window_height)
            )

        window.blit(img_pygame, (0, 0))

        for index in range(len(hands.gestures)):
            gesture_text = font.render(hands.gestures[index], False, (255, 255, 255))
            window.blit(gesture_text, (window_width - window_width // 5, index * 40))

        delay_AI = font.render(
            str(round(hands.delay, 1)) + "ms", False, (255, 255, 255)
        )
        window.blit(fps, (0, 0))
        window.blit(delay_AI, (0, 40))

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        window.blit(hand_surface_copy, (0, 0))

        # clock.tick(pygame.display.get_current_refresh_rate())
        clock.tick(60)
        pygame.display.update()


if __name__ == "__main__":
    main()
