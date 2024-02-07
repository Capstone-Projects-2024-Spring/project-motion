# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pygame
import pygame_menu
from GetHands import GetHands
from RenderHands import RenderHands
import math
import pyautogui
print(pyautogui.size())
# global variables
pygame.init()
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

# pass this flag in a list because of pass by reference/value stuff i think
render_hands_mode = [True]
number_of_hands = 2
SENSITIVITY = 0.03

def mouse(x,y):
    pyautogui.moveTo(1*x*pyautogui.size().width, 1*y*pyautogui.size().height, duration = 0, _pause=False)

def click():
    pyautogui.click(_pause = False)

def main():

    window_width = 1200
    window_height = 1000
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0, 0, 0))

    myRenderHands = RenderHands(hands_surface, 3)

    hands = GetHands(
        myRenderHands.render_hands,
        render_hands_mode,
        surface=hands_surface,
        confidence=0.5,
        hands=number_of_hands,
        is_pinching=is_pinching,
        move_mouse=mouse
    )

    menu = pygame_menu.Menu("Welcome", 400, 300, theme=pygame_menu.themes.THEME_BLUE)

    menu.add.selector(
        "Render Mode :", [("Normalized", True), ("World", False)], onchange=set_coords
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
    menu.add.button("Quit", pygame_menu.events.EXIT)
    menu.enable()

    print("game loop")
    game_loop(window, window_width, window_height, hands, hands_surface, menu)

    pygame.quit()


def set_coords(value, mode):
    render_hands_mode[0] = mode

def is_pinching(tip1, tip2):
    distance = math.sqrt((tip1.x-tip2.x)**2 + (tip1.y-tip2.y)**2 + (tip1.z-tip2.z)**2)
    if distance < SENSITIVITY:
        return True
    else:
        return False

def game_loop(window, window_width, window_height, hands, hands_surface, menu):
    hands.start()
    running = True

    while running:
        window.fill((0, 0, 0))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                hands.stop()
                running = False

        if menu.is_enabled():
            menu.update(events)
            menu.draw(window)

        # frames per second
        fps = font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )

        window.blit(hands_surface, (0, 0))
        window.blit(fps, (0, 0))

        clock.tick(60)
        pygame.display.update()


if __name__ == "__main__":
    main()
