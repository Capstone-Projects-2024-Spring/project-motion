# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pygame
import pygame_menu
from RecordHands import RecordHands
from RenderHands import RenderHands
from Writer import Writer
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# global variables
pygame.init()
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

flags = {
    "gesture_vector": [],
    "number_of_hands": 1,
}


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
        "fist",
        "forwards",
        "backwards",
        "thumb",
        "pinky",
        "peace",
        "wave",
        "scroll up",
        "scroll down",
    ]

    gesture_menu_selection = []

    for index, gesture in enumerate(gesture_list):
        flags["gesture_vector"].append("0")
        gesture_menu_selection.append((gesture_list[index], index))

    flags["gesture_vector"].append(False)

    myWriter = Writer(gesture_list=gesture_list, write_labels=False)
    # control_mouse=mouse_controls.control,
    hands = RecordHands(
        myRenderHands.render_hands,
        show_window=True,
        surface=hands_surface,
        confidence=0.5,
        write_csv=myWriter.write,
        gesture_list=gesture_list,
        flags=flags,
    )

    menu = pygame_menu.Menu(
        "Welcome",
        window_width * 0.8,
        window_height * 0.8,
        theme=pygame_menu.themes.THEME_BLUE,
    )

    def change_write(value, tuple):
        (write_status, writer) = tuple
        writer.write_labels = write_status

    menu.add.selector(
        "Labels :",
        [
            ("From Selection", (True, myWriter)),
            ("Don't save labels", (False, myWriter)),
        ],
        onchange=change_write,
    )

    menu.add.dropselect(
        "Gesture :", gesture_menu_selection, onchange=set_current_gesture
    )

    menu.add.button("Close Menu", pygame_menu.events.CLOSE)
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
        flags["gesture_vector"],
        myWriterMakeCSV=myWriter.makeCSV,
    )

    pygame.quit()


def set_write_status() -> None:
    """Tell the the writer class to write data"""
    flags["gesture_vector"][len(flags["gesture_vector"]) - 1] = not flags[
        "gesture_vector"
    ][len(flags["gesture_vector"]) - 1]


def set_current_gesture(value, index) -> None:
    """Define the current gesutre of a matching gesture list

    Args:
        value (_type_): used by pygame_menu
        index (_type_): index of gesture in gesture list
    """
    for myIndex, gesture in enumerate(flags["gesture_vector"]):
        flags["gesture_vector"][myIndex] = "0"
    flags["gesture_vector"][index] = "1"


def game_loop(
    window,
    window_width,
    window_height,
    hands,
    hands_surface,
    menu,
    gesture_list,
    gesture_vector,
    myWriterMakeCSV,
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

    generatedCSV = False

    while running:
        window.fill((0, 0, 0))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                hands.stop()
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:

                    if not generatedCSV:
                        generatedCSV = True
                        myWriterMakeCSV()

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
    main()
