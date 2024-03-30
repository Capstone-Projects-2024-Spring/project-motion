import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# images used for game
bg = pygame.image.load('asteroidpics/starbg.webp')
alien = pygame.image.load('asteroidpics/alienShip.png')
player = pygame.image.load('asteroidpics/spaceRocket.png')
star = pygame.image.load('asteroidpics/star.png')
asteroid1 = pygame.image.load('asteroidpics/asteroid1.png')
asteroid2 = pygame.image.load('asteroidpics/asteroid2.png')
asteroid3 = pygame.image.load('asteroidpics/asteroid3.png')

pygame.display.set_caption('Asteroids')
window = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))


run = True
clock = pygame.time.Clock()
game_over = False

def redraw_window():
        window.blit(bg,(0,0))

        pygame.display.update()

# game loop
while run:
    clock.tick(60)
    if not game_over:   
        redraw_window() 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()
