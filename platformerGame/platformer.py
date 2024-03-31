import pygame
import random

# initialize pygame
pygame.init()

#game window dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

#create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Jumpy')

# set frame rate
clock = pygame.time.Clock()
FPS = 60

# game variables
GRAVITY = 1
MAX_PLATFORMS = 10
SCROLL_THRESH = 200
scroll = 0
bg_scroll = 0
game_over = False
score = 0


# define colors
WHITE = (255, 255, 255)

# define font
font_small = pygame.font.SysFont('Lucida Sans', 20)
font_big = pygame.font.SysFont('Lucida Sans', 24)

# load images
bg_image = pygame.image.load('platformerGame/assets/bg.png').convert_alpha()
jumpy_image = pygame.image.load('platformerGame/assets/jump.png').convert_alpha()
platform_image = pygame.image.load('platformerGame/assets/wood.png').convert_alpha()

# function for outputting the text on the sceen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x,y))


# function for drawing the background
def draw_bg(bg_scroll):
    screen.blit(bg_image, (0,0 + bg_scroll))
    screen.blit(bg_image, (0, -600 + bg_scroll))


# player class
class Player():
    def __init__(self, x, y):
        self.image = pygame.transform.scale(jumpy_image, (45,45))
        self.width = 25
        self.height = 40
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.rect.center = (x,y)
        self.vel_y = 0
        self.flip = False
    
    def move(self):
        # reset variables
        dx = 0
        dy = 0
        scroll = 0
        # process key presses
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            dx = -10
            self.flip = True
        if key[pygame.K_RIGHT]:
            dx = 10
            self.flip = False
        
        # gravity
        self.vel_y += GRAVITY
        dy += self.vel_y
        
        # making sure character doesnt move off of screen
        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > SCREEN_WIDTH:
            dx = SCREEN_WIDTH - self.rect.right
        
        # check collision with platforms
        for platform in platform_group:
            # collision in the y direction
            if platform.rect.colliderect(self.rect.x, self.rect.y + dy, self.width, self.height):
                # check if above the platform
                if self.rect.bottom < platform.rect.centery:
                    if self.vel_y > 0:
                        self.rect.bottom = platform.rect.top
                        dy = 0
                        self.vel_y = -20

       
        # check if the player has bounced to the top of the screen
        if self.rect.top <= SCROLL_THRESH:
            # if player is jumping
            if self.vel_y < 0:
                scroll = -dy

        # update position
        self.rect.x += dx
        self.rect.y += dy + scroll 

        return scroll

    def draw(self):
        screen.blit(pygame.transform.flip(self.image, self.flip, False), (self.rect.x - 12, self.rect.y - 5))
        pygame.draw.rect(screen, WHITE, self.rect, 2)

# platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(platform_image, (width, 10))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
    def update(self, scroll):
        # update platforms vertical position
        self.rect.y += scroll

        # check if platform has gone off the screen
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()


# player instance
jumpy = Player(SCREEN_WIDTH//2, SCREEN_HEIGHT - 150)

# create sprite groups
platform_group = pygame.sprite.Group()

# create starting platform
platform = Platform(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT-50, 100)
platform_group.add(platform)

# game loop
run = True
while run:

    clock.tick(FPS)

    if not game_over:

        scroll = jumpy.move()

        # draw background
        bg_scroll += scroll
        if bg_scroll >= 600:
            bg_scroll = 0
        draw_bg(bg_scroll)

        # generate platforms
        if len(platform_group) < MAX_PLATFORMS:
            p_w = random.randint(50,70)
            p_x =  random.randint(0, SCREEN_WIDTH - p_w)
            p_y = platform.rect.y - random.randint(80, 120)
            platform = Platform(p_x, p_y, p_w)
            platform_group.add(platform)


        # update platforms
        platform_group.update(scroll)

        # draw characters
        platform_group.draw(screen)
        jumpy.draw()

        # check game over
        if jumpy.rect.top > SCREEN_HEIGHT:
            game_over = True
    
    else:
        draw_text("GAME OVER!", font_big, WHITE, 130, 200)
        draw_text("SCORE: " + str(score), font_big, WHITE, 150, 250)
        draw_text('PRESS SPACE TO PLAY AGAIN', font_big, WHITE, 40, 300)
        key = pygame.key.get_pressed()
        if key[pygame.K_SPACE]:
            # reset variables:
            game_over = False
            score = 0
            scroll = 0
            # reposition player
            jumpy.rect.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT-150)
            # reset platforms
            platform_group.empty()
            # recreate starting platform 
            platform = Platform(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT-50, 100)
            platform_group.add(platform)

    # event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False


    # update display
    pygame.display.update()
    

pygame.quit()