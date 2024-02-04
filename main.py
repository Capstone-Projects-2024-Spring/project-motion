# https://developers.google.com/mediapipe/framework/getting_started/gpu_support
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# https://pygame-menu.readthedocs.io/en/latest/_source/add_widgets.html
import pygame
import pygame_menu

from GetHands import GetHands

#global variables
pygame.init()
font = pygame.font.Font("freesansbold.ttf", 30)
clock = pygame.time.Clock()

def main():

    window_width = 800
    window_height = 600
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Test Hand Tracking Multithreaded")

    hands_surface = pygame.Surface((window_width, window_height))
    hands_surface.set_colorkey((0,0,0))
    hands = GetHands(
        render_hands, surface=hands_surface, confidence=0.5, hands=2
    )

    menu = pygame_menu.Menu('Welcome', 400, 300,
                       theme=pygame_menu.themes.THEME_BLUE)
    
    menu.add.selector('Difficulty :', [('Hard', 1), ('Easy', 2)], onchange=set_difficulty)
    menu.add.range_slider('Hands', 2, (0, 4), 1,
                      rangeslider_id='range_slider',
                      value_format=lambda x: str(int(x)))
    menu.add.button('Close Menu', pygame_menu.events.CLOSE)
    menu.add.button('Quit', pygame_menu.events.EXIT)
    menu.enable()

    print("game loop")
    game_loop(window, window_width, window_height, hands, hands_surface, menu)

    pygame.quit()


def render_hands(result, output_image, delay_ms, surface):
    """Used as function callback by Mediapipe hands model

    Args:
        result (Hands): list of hands and each hand's 21 landmarks
        output_image (_type_): _description_
        delay_ms ((float, float)): Webcam latency and AI processing latency
    """
    # Render hand landmarks
    # print(delay_ms)
    surface.fill((0, 0, 0))
    if result.hand_landmarks:
        # define colors for different hands
        hand_color = 0
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
        # get every hand detected
        for hand in result.hand_landmarks:
            # each hand has 21 landmarks
            for landmark in hand:
                render_hands_pygame(
                    colors[hand_color], landmark.x, landmark.y, surface, delay_ms
                )
            hand_color += 1


def render_hands_pygame(color, x, y, surface, delay_ms):
    w, h = surface.get_size()
    pygame.draw.circle(surface, color, (x * w, y * h), 5)
    delay_cam = font.render(str(delay_ms[0]) + "ms", False, (255, 255, 255))
    delay_AI = font.render(str(delay_ms[1]) + "ms", False, (255, 255, 255))
    surface.blit(delay_cam, (0, 30))
    surface.blit(delay_AI, (0, 60))

def set_difficulty(value, difficulty):
    # Do the job here !
    pass


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
