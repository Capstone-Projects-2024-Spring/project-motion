import pygame
from pygame.locals import *
import random

pygame.init()

#fps variables
clock = pygame.time.Clock()
fps = 60

# screen resolution & title
screen_width = 864
screen_height = 936
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Stella')

#define font
font = pygame.font.SysFont('Bauhaus 93', 60)

#define colours
white = (255, 255, 255)

#define game variable
ground_scroll = 0
scroll_speed = 4
flying = False
game_over = False
paused = False
pipe_gap = 175
pipe_frequency = 1500 # milliseconds
last_pipe = pygame.time.get_ticks() - pipe_frequency
score = 0
pass_pipe = False
started = False

#load image
bg = pygame.image.load('FlappyBird/img/bg.png')
ground_img = pygame.image.load('FlappyBird/img/ground.png')
restart_button_img = pygame.image.load('FlappyBird/img/restart.png')
exit_button_img = pygame.image.load('FlappyBird/img/exit.png')
start_img = pygame.image.load('FlappyBird/img/start.png')

def getHighScore():
    try:
        # Open the text file in read mode
        with open('FlappyBird/highscore.txt', 'r') as file:
            # Read the first number from the file
            high_score = int(file.readline())
            return f"Highscore: {high_score}"
    except FileNotFoundError:
        return "Highscore: No high score recorded yet"

def highscore_check(score):
    try:
        # Open the text file in read mode
        with open('FlappyBird/highscore.txt', 'r') as file:
            # Read the first number from the file
            current_score = int(file.readline())
    except FileNotFoundError:
        # If the file doesn't exist, create it with the initial score
        with open('FlappyBird/highscore.txt', 'w') as file:
            file.write(str(score))
        return

    # Compare the current score with the new score
    if score > current_score:
        # If the new score is greater, update the score in the file
        with open('FlappyBird/highscore.txt', 'w') as file:
            file.write(str(score))
        # print("New high score updated!")

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def reset_game():
    pipe_group.empty()
    flappy.rect.x = 100
    flappy.rect.y = int(screen_height / 2)
    score = 0
    return score

class Bird(pygame.sprite.Sprite):
    def __init__ (self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range (1, 4):
            img = pygame.image.load(f'FlappyBird/img/bird{num}.png')
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False
        
    def update(self):

        if paused == False:
            if flying == True:
                #gravity
                self.vel += 0.5
                if self.vel > 8:
                    self.vel = 8
            
            if self.rect.bottom < 768:
                self.rect.y += int(self.vel)

            if game_over == False:
                #jump
                if pygame.key.get_pressed()[K_SPACE] == 1 and self.clicked == False:
                    self.clicked = True
                    self.vel = -10
                if pygame.key.get_pressed()[K_SPACE] == 0:
                    self.clicked = False
                    

                #handle animation
                self.counter += 1
                flap_cooldown = 5

                if self.counter > flap_cooldown:
                    self.counter = 0
                    self.index += 1
                    if self.index >= len(self.images):
                        self.index = 0
                self.image = self.images[self.index]

                #rotate the bird
                self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
            else:
                self.image = pygame.transform.rotate(self.images[self.index], -90)


class Pipe(pygame.sprite.Sprite):
    def __init__ (self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('FlappyBird/img/pipe.png')
        self.rect = self.image.get_rect()
        # position 1 is from the top, -1 is from the bottom
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
        if position == -1:
            self.rect.topleft = [x, y + int(pipe_gap / 2)]

    def update(self):
        if paused == False:
            self.rect.x -= scroll_speed
            if self.rect.right < 0:
                self.kill()

class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):
        action  = False
        
        #get mouse position
        pos = pygame.mouse.get_pos()

        #check if mouse is over the button
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        #draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))

        return action
    
class Show():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):
        #draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))

bird_group = pygame.sprite.Group()
pipe_group = pygame.sprite.Group()

flappy = Bird(100, int(screen_height/2))
bird_group.add(flappy)

#create restart button instance
restart_button = Button(screen_width // 2 - 50, screen_height // 2 - 125, restart_button_img)
exit_button = Button(screen_width // 2 - 50, screen_height // 2 - 50, exit_button_img)
start_image = Show((screen_width // 2 - start_img.get_width() // 2), (screen_height // 2 - start_img.get_height() // 2), start_img)

run = True
while run:

    #fps
    clock.tick(fps)

    # draw background
    screen.blit(bg, (0,0))

    #bird
    bird_group.draw(screen)
    bird_group.update()

    #pipes
    pipe_group.draw(screen)

    #draw the ground
    screen.blit(ground_img, (ground_scroll, 768))

    #check the score
    if len(pipe_group) > 0:
        if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left\
            and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right\
            and pass_pipe == False:
            pass_pipe = True
        if pass_pipe == True:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                score += 1
                pass_pipe = False

    
    draw_text(getHighScore(), font, white, int(screen_width/2 - 100), 20)
    draw_text(str(score), font, white, int(screen_width/2), 75)
    
    if started == False and game_over == False and flying == False:
        start_image.draw()


    # look for collition
    if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0:
        game_over = True

    # check if bird hit ground
    if flappy.rect.bottom >= 768:
        game_over = True
        flying = False

    if game_over == False and flying == True and paused == False:

        #generate new pipes 
        time_now = pygame.time.get_ticks()
        if time_now - last_pipe > pipe_frequency:
            pipe_height = random.randint(-100, 100)
            btm_pipe = Pipe(screen_width, int(screen_height/2) + pipe_height, -1)
            top_pipe = Pipe(screen_width, int(screen_height/2) + pipe_height, 1)
            pipe_group.add(btm_pipe)
            pipe_group.add(top_pipe)
            last_pipe = time_now

        #scroll the ground
        ground_scroll -= scroll_speed
        if abs(ground_scroll) > 35:
            ground_scroll = 0
        
        pipe_group.update()

    # check for game over and reset
    if game_over == True:
        # check high score and save it here
        highscore_check(score)

        if restart_button.draw() == True:
            game_over = False
            score = reset_game()
        if exit_button.draw() == True:
            run = False
    
    if paused == True and game_over == False:
        if restart_button.draw() == True:
            # check high score and save it here
            highscore_check(score)
            score = reset_game()
            paused = False
        if exit_button.draw() == True:
            run = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and flying == False and game_over == False:
            flying = True
            started = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused

    pygame.display.update()

pygame.quit()