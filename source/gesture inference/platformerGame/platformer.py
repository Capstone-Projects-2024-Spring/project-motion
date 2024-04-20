import pygame
import random
import sys
from os import path, chdir
bundle_dir = path.dirname(path.abspath(__file__))
chdir(bundle_dir)
class Enemy(pygame.sprite.Sprite):
	def __init__(self, SCREEN_WIDTH, y, sprite_sheet, scale):
		pygame.sprite.Sprite.__init__(self)
		#define variables
		self.animation_list = []
		self.frame_index = 0
		self.update_time = pygame.time.get_ticks()
		self.direction = random.choice([-1, 1])
		if self.direction == 1:
			self.flip = True
		else:
			self.flip = False

		#load images from spritesheet
		animation_steps = 8
		for animation in range(animation_steps):
			image = sprite_sheet.get_image(animation, 32, 32, scale, (0, 0, 0))
			image = pygame.transform.flip(image, self.flip, False)
			image.set_colorkey((0, 0, 0))
			self.animation_list.append(image)
		
		#select starting image and create rectangle from it
		self.image = self.animation_list[self.frame_index]
		self.rect = self.image.get_rect()

		if self.direction == 1:
			self.rect.x = 0
		else:
			self.rect.x = SCREEN_WIDTH
		self.rect.y = y

	def update(self, scroll, SCREEN_WIDTH):
		#update animation
		ANIMATION_COOLDOWN = 50
		#update image depending on current frame
		self.image = self.animation_list[self.frame_index]
		#check if enough time has passed since the last update
		if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
			self.update_time = pygame.time.get_ticks()
			self.frame_index += 1
		#if the animation has run out then reset back to the start
		if self.frame_index >= len(self.animation_list):
			self.frame_index = 0

		#move enemy
		self.rect.x += self.direction * 2
		self.rect.y += scroll

		#check if gone off screen
		if self.rect.right < 0 or self.rect.left > SCREEN_WIDTH:
			self.kill()

class SpriteSheet():
	def __init__(self, image):
		self.sheet = image

	def get_image(self, frame, width, height, scale, colour):
		image = pygame.Surface((width, height)).convert_alpha()
		image.blit(self.sheet, (0, 0), ((frame * width), 0, width, height))
		image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
		image.set_colorkey(colour)

		return image

# initialize pygame
pygame.init()

#game window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

#create game window
surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
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

if path.exists('score.txt'):
    with open('score.txt', 'r') as file:
        high_score = int(file.read())
else:
    high_score = 0


# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (153, 217, 234)

# define font
font_small = pygame.font.Font('Catfiles.ttf', 24)
font_big = pygame.font.Font('Catfiles.ttf', 30)

# load images

pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
bg_image = pygame.image.load('assets/bg.png').convert_alpha()
jumpy_image = pygame.image.load('assets/jump.png').convert_alpha()
platform_image = pygame.image.load('assets/wood.png').convert_alpha()
bird_sheet_img = pygame.image.load('assets/bird.png').convert_alpha()
bird_sheet = SpriteSheet(bird_sheet_img)


# function for outputting the text on the sceen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    surface.blit(img, (x,y))

# function for drawing info panel
def draw_panel():
    pygame.draw.rect(surface, PANEL, (0,0, SCREEN_WIDTH, 30))
    pygame.draw.line(surface, BLACK, (0, 30), (SCREEN_WIDTH, 30), 2)
    draw_text('SCORE: ' + str(score), font_small, BLACK, 0, 0)

# function for drawing the background
def draw_bg(bg_scroll):
    surface.blit(bg_image, (0,0 + bg_scroll))
    surface.blit(bg_image, (0, -600 + bg_scroll))


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
        
        # making sure character doesnt move off of surface
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
                        self.vel_y = -30

       
        # check if the player has bounced to the top of the surface
        if self.rect.top <= SCROLL_THRESH:
            # if player is jumping
            if self.vel_y < 0:
                scroll = -dy

        # update position
        self.rect.x += dx
        self.rect.y += dy + scroll

        # update mask
        self.mask = pygame.mask.from_surface(self.image)

        return scroll

    def draw(self):
        surface.blit(pygame.transform.flip(self.image, self.flip, False), (self.rect.x - 12, self.rect.y - 5))
        # pygame.draw.rect(surface, WHITE, self.rect, 2)

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

        # check if platform has gone off the surface
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()


# player instance
jumpy = Player(SCREEN_WIDTH//2, SCREEN_HEIGHT - 150)

# create sprite groups
platform_group = pygame.sprite.Group()
enemy_group = pygame.sprite.Group()

# create starting platform
platform = Platform(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT-50, 100, False)
platform_group.add(platform)



#create game window
surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Jumpy')

# set frame rate
clock = pygame.time.Clock()
FPS = 60

#game loop
def tick():
    global clock
    global jumpy
    global platform_group
    global platform
    global enemy_group
    global score
    global scroll
    global bg_image
    global bg_scroll
    global game_over
    global fade_counter
    global surface
    global high_score

    clock.tick(FPS)

    if game_over == False:
        scroll = jumpy.move()

        #draw background
        bg_scroll += scroll
        if bg_scroll >= 600:
            bg_scroll = 0
        draw_bg(bg_scroll)

        #generate platforms
        if len(platform_group) < MAX_PLATFORMS:
            p_w = random.randint(120, 150)
            p_x = random.randint(0, SCREEN_WIDTH - p_w)
            p_y = platform.rect.y - random.randint(80, 150)
            p_type = random.randint(1, 2)
            if p_type == 1 and score > 1000:
                p_moving = True
            else:
                p_moving = False
            platform = Platform(p_x, p_y, p_w, p_moving)
            platform_group.add(platform)

        #update platforms
        platform_group.update(scroll)

        #generate enemies
        if len(enemy_group) == 0 and score > 1500:
            enemy = Enemy(SCREEN_WIDTH, 100, bird_sheet, 1.5)
            enemy_group.add(enemy)

        #update enemies
        enemy_group.update(scroll, SCREEN_WIDTH)

        #update score
        if scroll > 0:
            score += scroll

        #draw line at previous high score
        pygame.draw.line(surface, WHITE, (0, score - high_score + SCROLL_THRESH), (SCREEN_WIDTH, score - high_score + SCROLL_THRESH), 3)
        draw_text('HIGH SCORE', font_small, WHITE, SCREEN_WIDTH - 130, score - high_score + SCROLL_THRESH)

        #draw sprites
        platform_group.draw(surface)
        enemy_group.draw(surface)
        jumpy.draw()

        #draw panel
        draw_panel()

        #check game over
        if jumpy.rect.top > SCREEN_HEIGHT:
            game_over = True

        #check for collision with enemies
        if pygame.sprite.spritecollide(jumpy, enemy_group, False):
            if pygame.sprite.spritecollide(jumpy, enemy_group, False, pygame.sprite.collide_mask):
                game_over = True

    else:
        if fade_counter < SCREEN_WIDTH:
            fade_counter += 5
            for y in range(0, 6, 2):
                pygame.draw.rect(surface, BLACK, (0, y * 133, fade_counter, 133))
                pygame.draw.rect(surface, BLACK, (SCREEN_WIDTH - fade_counter, (y + 1) * 133, SCREEN_WIDTH, 133))
        else:
            draw_text('GAME OVER!', font_big, WHITE, 300, 200)
            draw_text('SCORE: ' + str(score), font_big, WHITE, 275, 250)
            draw_text('PRESS SPACE', font_big, WHITE, 295, 300)
            draw_text('TO PLAY AGAIN', font_big, WHITE, 280, 350)
            # update high score
            if score > high_score:
                high_score = score
                with open('score.txt', 'w') as file:
                    file.write(str(high_score))
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE]:
                #reset variables
                game_over = False
                score = 0
                scroll = 0
                fade_counter = 0
                #reposition jumpy
                jumpy.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
                #reset enemies
                enemy_group.empty()
                #reset platforms
                platform_group.empty()
                #create starting platform
                platform = Platform(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 50, 100, False)
                platform_group.add(platform)


def events(events):
    global high_score
    for event in events:
        if event.type == pygame.QUIT:
            #update high score
            if score > high_score:
                high_score = score
                with open('score.txt', 'w') as file:
                    file.write(str(high_score))

