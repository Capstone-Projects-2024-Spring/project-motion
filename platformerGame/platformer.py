import pygame
import random
import os

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
fade_counter = 0

if os.path.exists('platformerGame/score.txt'):
    with open('platformerGame/score.txt', 'r') as file:
        high_score = int(file.read())
else:
    high_score = 0


# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (153, 217, 234)

# define font
font_small = pygame.font.Font('platformerGame/Catfiles.ttf', 24)
font_big = pygame.font.Font('platformerGame/Catfiles.ttf', 30)

# load images
bg_image = pygame.image.load('platformerGame/assets/bg.png').convert_alpha()
jumpy_image = pygame.image.load('platformerGame/assets/jump.png').convert_alpha()
platform_image = pygame.image.load('platformerGame/assets/wood.png').convert_alpha()

# function for outputting the text on the sceen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x,y))

# function for drawing info panel
def draw_panel():
    pygame.draw.rect(screen, PANEL, (0,0, SCREEN_WIDTH, 30))
    pygame.draw.line(screen, BLACK, (0, 30), (SCREEN_WIDTH, 30), 2)
    draw_text('SCORE: ' + str(score), font_small, BLACK, 0, 0)

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
        # pygame.draw.rect(screen, WHITE, self.rect, 2)

# platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, moving):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(platform_image, (width, 10))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.moving = moving
        self.move_counter = random.randint(0, 50)
        self.direction = random.choice([-1,1])
        self.speed = random.randint(1,2)
    
    def update(self, scroll):
        # move moving platforms side to side
        if self.moving == True:
            self.move_counter += 1
            self.rect.x += self.direction * self.speed
            


        # change platform direction if it has moved fully
        if self.move_counter >= 100 or self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.direction *= -1
            self.move_counter = 0

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
platform = Platform(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT-50, 100, False)
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
            p_type = random.randint(1,2)
            if p_type == 1 and score > 1000:
                p_moving = True
            else:
                p_moving = False
            platform = Platform(p_x, p_y, p_w, p_moving)
            platform_group.add(platform)


        # update platforms
        platform_group.update(scroll)

        # update score
        if scroll > 0:
            score += scroll
        
        # draw line at previous high score
        pygame.draw.line(screen, BLACK, (0, score-high_score + SCROLL_THRESH), (SCREEN_WIDTH, score-high_score + SCROLL_THRESH), 3)
        draw_text('HIGH SCORE', font_small, BLACK, SCREEN_WIDTH-200, score-high_score + SCROLL_THRESH)
        # draw characters
        platform_group.draw(screen)
        jumpy.draw()

        # draw panel
        draw_panel()

        # check game over
        if jumpy.rect.top > SCREEN_HEIGHT:
            game_over = True
    
    else:
        if fade_counter < SCREEN_WIDTH:
            fade_counter += 5
            for y in range(0, 6, 2):
                pygame.draw.rect(screen, BLACK, (0, y * 100, fade_counter, SCREEN_HEIGHT/6))
                pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH - fade_counter, (y + 1) * 100, SCREEN_WIDTH, SCREEN_HEIGHT/6))
        else:
            draw_text("GAME OVER!", font_big, WHITE, 100, 200)
            draw_text("SCORE: " + str(score), font_big, WHITE, 115, 250)
            draw_text('PRESS SPACE', font_big, WHITE, 95, 300)
            draw_text('TO PLAY AGAIN', font_big, WHITE, 80, 350)
            # update high score
            if score > high_score:
                high_score = score
                with open('platformerGame/score.txt', 'w') as file:
                    file.write(str(high_score))
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE]:
                # reset variables:
                game_over = False
                score = 0
                scroll = 0
                fade_counter = 0
                # reposition player
                jumpy.rect.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT-150)
                # reset platforms
                platform_group.empty()
                # recreate starting platform 
                platform = Platform(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT-50, 100, False)
                platform_group.add(platform)

    # event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if score > high_score:
                high_score = score
                with open('platformerGame/score.txt', 'w') as file:
                    file.write(str(high_score))
            run = False


    # update display
    pygame.display.update()
    

pygame.quit()