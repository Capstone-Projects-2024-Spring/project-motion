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
import textwrap

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
    """Main driver method which initilizes all children and starts pygame render pipeline"""

    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    renderHands = RenderHands(render_scale=3)

    mouse = Mouse()

    keyboard = Keyboard(
        threshold=0, toggle_key_threshold=0.3, toggle_mouse_func=toggle_mouse
    )

    # control_mouse=mouse_controls.control,
    hands = GetHands(
        flags=flags,
    )

    menu = pygame_menu.Menu(
        "Esc to toggle menu",
        window_width * 0.5,
        window_height * 0.5,
        theme=pygame_menu.themes.THEME_BLUE,
    )

    menu.add.selector(
        "Render Mode :", [("Normalized", True), ("World", False)], onchange=set_coords
    )

    def change_mouse_smooth(value, smooth):
        nonlocal mouse
        mouse.x_window = []
        mouse.y_window = []
        mouse.window_size = smooth

    menu.add.selector(
        "Mouse Smoothing :",
        [("None", 1), ("Low", 2), ("Medium", 6), ("High", 12), ("Max", 24)],
        default=3,
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
        flags["gesture_model_path"] = value[0][0]  # tuple within a list for some reason
        build_hands()

    menu.add.dropselect("Use Gesture Model :", models, onchange=change_gesture_model)

    def set_click_sense(value, **kwargs):
        nonlocal hands
        print(value)
        flags["click_sense"] = value / 1000

    menu.add.range_slider(
        "Click Sensitivity",
        default=70,
        range_values=(1, 150),
        increment=1,
        onchange=set_click_sense,
    )

    def build_hands():
        nonlocal hands
        nonlocal mouse
        nonlocal keyboard
        hands.stop()
        hands.join()
        hands = GetHands(
            flags=flags,
        )
        flags["hands"] = hands
        hands.start()

    menu.add.button("Turn On Model", action=toggle_model)
    menu.add.button("Turn On Mouse", action=toggle_mouse)
    menu.add.button("Quit", pygame_menu.events.EXIT)
    menu.enable()

    game_loop(window, hands, menu, mouse, keyboard, renderHands)

    pygame.quit()


def find_files_with_ending(ending: str, directory_path=os.getcwd()):
    """returns a list of tuples of the strings found"""
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
    menu: pygame_menu.Menu,
    mouse: Mouse,
    keyboard: Keyboard,
    renderHands: RenderHands,
):
    """Runs the pygame event loop and renders surfaces"""
    hands.start()
    running = True
    is_menu_showing = True
    webcam_mode = 1
    show_debug_text = True
    is_fullscreen = False
    counter = 0
    delay_AI = None
    window_width, window_height = pygame.display.get_surface().get_size()

    hand_surfaces = []
    for i in range(4):
        hand_surfaces.append(pygame.Surface((window_width, window_height)))
        hand_surfaces[i].set_colorkey((0, 0, 0))

    wrapper = textwrap.TextWrapper(width=20)
    instructions = "F1 to change webcam place. F2 to hide this text. F3 to change hand render mode. 'M' to toggle mouse control. 'G' to toggle gesture model."
    instructions = wrapper.wrap(text=instructions)

    while running:
        counter += 1

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
            console.print(hands.confidence_vectors)
            #send only the first hand confidence vector the gesture model output
            keyboard.gesture_input(hands.confidence_vectors[0])

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        frame = hands.frame.copy()
        img_pygame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
        img_width = img_pygame.get_width()
        img_height = img_pygame.get_height()

        for i in range(4):
            hand_surfaces[i] = pygame.transform.scale(
                hand_surfaces[i], (img_width * 0.5, img_height * 0.5)
            )

        # fullscreen webcam
        if webcam_mode % 3 == 0:
            renderHands.thickness = 15
            img_pygame = pygame.transform.scale(
                img_pygame, (window_width, window_height)
            )
            for i in range(4):
                hand_surfaces[i] = pygame.transform.scale(
                    hand_surfaces[i], (window_width, window_height)
                )
            window.blit(img_pygame, (0, 0))
        # windowed webcam
        elif webcam_mode % 3 == 1:
            renderHands.thickness = 5
            img_pygame = pygame.transform.scale(
                img_pygame, (img_width * 0.5, img_height * 0.5)
            )
            window.blit(img_pygame, (0, 0))
        # no webcam
        elif webcam_mode % 3 == 2:
            pass
        # use this again for putting hands in the corners
        img_width = img_pygame.get_width()
        img_height = img_pygame.get_height()

        if hands.location != []:
            for index in range(hands.num_hands_deteced):
                if flags["render_hands_mode"]:
                    landmarks = hands.result.hand_world_landmarks
                else:
                    landmarks = hands.result.hand_landmarks
                for i in range(hands.num_hands_deteced):
                    renderHands.render_hands(
                        landmarks[i],
                        flags["render_hands_mode"],
                        hands.location,
                        hands.velocity,
                        hand_surfaces[i],
                        i,
                    )
        else:
            for i in range(4):
                hand_surfaces[i].fill((0, 0, 0))

        if show_debug_text:
            for index in range(len(hands.gestures)):
                gesture_text = font.render(
                    hands.gestures[index], False, (255, 255, 255)
                )
                window.blit(gesture_text, (0, index * 40 + 80))

            if hands.delay != 0 and counter % 60 == 0:
                delay_AI = font.render(
                    "Webcam: " + str(round(1000 / hands.delay, 1)) + "fps",
                    False,
                    (255, 255, 255),
                )
            if delay_AI != None:
                window.blit(delay_AI, (0, 40))
            window.blit(fps, (0, 0))

            for index, instruc in enumerate(instructions):
                instructoins_text = font.render(instruc, False, (255, 255, 255))
                window.blit(instructoins_text, (0, 400 + index * 40))

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        corners = [
            (0, 0),
            (window_width - img_width, 0),
            (0, window_height - img_height),
            (window_width - img_width, window_height - img_height),
        ]

        for i in range(hands.num_hands_deteced):
            window.blit(hand_surfaces[i], corners[i])

        # clock.tick(pygame.display.get_current_refresh_rate())
        clock.tick(60)
        pygame.display.update()


if __name__ == "__main__":
    main()
