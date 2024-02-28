# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pygame
import pygame_menu
from RenderHands import RenderHands
from Writer import Writer
from Reader import Reader
# global variables
pygame.init()
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

render_hands_mode = [True]
gesture_vector = []
number_of_hands = 1
move_mouse_flag = [False]


def main():
    """Main driver method which initilizes all children and starts pygame render pipeline
    """

    window_width = 800
    window_height = 800
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Labeler")

    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0, 0, 0))

    myRenderHands = RenderHands(hands_surface, 3)
    myReader = Reader("manualData.csv")

    gesture_list = [
        "shoot",
        "west coast",
        "point left",
        "point right",
        "stop",
    ]

    #myWriter = Writer(gesture_list=gesture_list, write_labels=False)

    gesture_menu_selection = []

    for index, gesture in enumerate(gesture_list):
        gesture_vector.append("0")
        gesture_menu_selection.append((gesture_list[index], index))

    gesture_vector.append(False)

    # control_mouse=mouse_controls.control,

    menu = pygame_menu.Menu(
        "Welcome",
        window_width * 0.8,
        window_height * 0.8,
        theme=pygame_menu.themes.THEME_BLUE,
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
        hands_surface,
        menu,
        myReader,
        gesture_list,
        gesture_vector,
        myRenderHands
    )

    pygame.quit()


def set_current_gesture(value, index):
    for myIndex, gesture in enumerate(gesture_vector):
        gesture_vector[myIndex] = "0"
    gesture_vector[index] = "1"

def game_loop(
    window,
    window_width,
    window_height,
    hands_surface,
    menu,
    reader,
    gesture_list,
    gesture_vector,
    myRenderHands
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
    running = True

    while running:
        window.fill((0, 0, 0))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    hand = reader.read()
                    myRenderHands.render_hands(hand)

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
