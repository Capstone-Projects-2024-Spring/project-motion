from flappybird import FlappyBirdGame
import pygame

if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    # Set up the display
    screen_width = 864
    screen_height = 936
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Flappy Bird")
    # Create an instance of the FlappyBirdGame class
    game = FlappyBirdGame(screen)
    run = True
    primitives = game.primitives

    while run:

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and primitives["is flying"] == False and primitives["is game over"] == False:
                primitives["is flying"] = True
                primitives["is started"] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    primitives["is paused"] = not primitives["is paused"]

        # Run a single frame of the game
        run = game.tick()

        screen.blit(game.surface, (0, 0))

        # Update the display
        pygame.display.update()

        clock.tick(60)

    pygame.quit()