# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pydoc
import pygame
import pygame_menu
from GetHands import GetHands
from RenderHands import RenderHands
from Mouse import Mouse
from Writer import Writer

# global variables
pygame.init()
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

render_hands_mode = [True]
gesture_vector = []
number_of_hands = 1
move_mouse_flag = [False]


def main() -> None:
    """Main driver method which initilizes all children and starts pygame render pipeline"""

    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0, 0, 0))

    myRenderHands = RenderHands(hands_surface, 3)

    gesture_list = [
        "shoot",
        "west coast",
        "point left",
        "point right",
        "stop",
    ]

    myWriter = Writer(gesture_list=gesture_list, write_labels=False)

    mouse_controls = Mouse(mouse_scale=2)

    gesture_menu_selection = []

    for index, gesture in enumerate(gesture_list):
        gesture_vector.append("0")
        gesture_menu_selection.append((gesture_list[index], index))

    gesture_vector.append(False)

    # control_mouse=mouse_controls.control,
    hands = GetHands(
        myRenderHands.render_hands,
        render_hands_mode,
        show_window=True,
        surface=hands_surface,
        confidence=0.5,
        hands=number_of_hands,
        control_mouse=None,
        write_csv=myWriter.write,
        gesture_vector=gesture_vector,
        gesture_list=gesture_list,
        move_mouse_flag=move_mouse_flag,
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

    menu.add.dropselect(
        "Gesture :", gesture_menu_selection, onchange=set_current_gesture
    )
    menu.add.range_slider(
        "Hands",
        2,
        (1, 2, 3, 4),
        1,
        rangeslider_id="range_slider",
        value_format=lambda x: str(int(x)),
        onchange=lambda value: hands.build_model(value),
    )
    menu.add.button("Close Menu", pygame_menu.events.CLOSE)
    menu.add.button("Toggle Mouse", action=toggle_mouse)
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
        gesture_list,
        gesture_vector,
    )

    pygame.quit()


def toggle_mouse() -> None:
    """Enable or disable mouse control"""
    move_mouse_flag[0] = not move_mouse_flag[0]


def set_write_status() -> None:
    """Tell the the writer class to write data"""
    gesture_vector[len(gesture_vector) - 1] = not gesture_vector[
        len(gesture_vector) - 1
    ]


def set_coords(value, mode) -> None:
    """Defines the coordinate space for rendering hands

    Args:
        value (_type_): used by pygame_menu
        mode (_type_): True for normalized, False for world
    """
    render_hands_mode[0] = mode


def set_current_gesture(value, index) -> None:
    """Define the current gesutre of a matching gesture list

    Args:
        value (_type_): used by pygame_menu
        index (_type_): index of gesture in gesture list
    """
    for myIndex, gesture in enumerate(gesture_vector):
        gesture_vector[myIndex] = "0"
    gesture_vector[index] = "1"


def game_loop(
    window,
    window_width,
    window_height,
    hands,
    hands_surface,
    menu,
    gesture_list,
    gesture_vector,
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
                if event.key == pygame.K_SPACE:
                    set_write_status()

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        if gesture_vector[len(gesture_vector) - 1] == True:
            saving_data = font.render("saving data", False, (255, 255, 255))
        else:
            saving_data = font.render("press space", False, (255, 255, 255))

        for index, gesture in enumerate(gesture_list):
            if gesture_vector[index] == "1":
                gesture_text = font.render(gesture_list[index], False, (255, 255, 255))
                break
            else:
                gesture_text = font.render("no gesture", False, (255, 255, 255))

        window.blit(gesture_text, (window_width - window_width // 5, 0))
        window.blit(saving_data, (window_width - window_width // 5, 40))
        window.blit(hands_surface, (0, 0))
        window.blit(fps, (0, 0))

        clock.tick(60)
        pygame.display.update()


if __name__ == "__main__":
    pydoc.writedoc("main")