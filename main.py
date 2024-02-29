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

gesture_vector = []


def main():
    """Main driver method which initilizes all children and starts pygame render pipeline"""

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

    myWriter = Writer(gesture_list=gesture_list, write_labels=True)

    gesture_menu_selection = []

    for index, gesture in enumerate(gesture_list):
        gesture_vector.append("0")
        gesture_menu_selection.append((gesture_list[index], index))

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
        myRenderHands,
        myWriter,
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
    myRenderHands,
    myWriter,
):

    running = True

    frame_multiplier = 1
    hand = reader.read()
    myRenderHands.render_hands(hand)

    zeros=[]
    for index in range(len(gesture_list)):
        zeros.append("0")

    current_gesture = ""

    while running:
        window.fill((0, 0, 0))
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(current_gesture)
                    for index in range(frame_multiplier):
                        hand = reader.read()
                        myWriter.write(hand, gesture_vector)

                    myRenderHands.render_hands(hand)

                if event.key == pygame.K_UP:
                    frame_multiplier += 1
                if event.key == pygame.K_DOWN:
                    frame_multiplier -= 1
                    if frame_multiplier < 1:
                        frame_multiplier = 1
  
                if event.key == pygame.K_LEFT:
                    hand = reader.go_back(frame_multiplier)
                    myWriter.remove_rows(frame_multiplier, reader.frame_count)
                    myRenderHands.render_hands(hand)

                if event.key == pygame.K_RIGHT:
                    print("not a gesture")
                    for index in range(frame_multiplier):
                        hand = reader.read()
                        myWriter.write(hand, zeros)

                    myRenderHands.render_hands(hand)

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        # frames per second
        frame_number = font.render(
            str(reader.frame_count) + "frame", False, (255, 255, 255)
        )

        if gesture_vector[len(gesture_vector) - 1] == True:
            saving_data = font.render("saving data", False, (255, 255, 255))
        else:
            saving_data = font.render("press space", False, (255, 255, 255))

        for index, gesture in enumerate(gesture_list):
            if gesture_vector[index] == "1":
                current_gesture = gesture_list[index]
                gesture_text = font.render(gesture_list[index], False, (255, 255, 255))
                break
            else:
                gesture_text = font.render("no gesture", False, (255, 255, 255))

        frame_multiplier_text = font.render(
            str(frame_multiplier), False, (255, 255, 255)
        )
        window.blit(gesture_text, (window_width - window_width // 5, 0))
        window.blit(saving_data, (window_width - window_width // 5, 40))
        window.blit(frame_multiplier_text, (window_width - window_width // 5, 80))
        window.blit(hands_surface, (0, 0))
        window.blit(frame_number, (0, 0))

        clock.tick(60)
        pygame.display.update()

if __name__ == "__main__":
    main()
