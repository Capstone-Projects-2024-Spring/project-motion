import pygame
import random
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class SpriteSheet():
	def __init__(self, image):
		self.sheet = image

	def get_image(self, frame, width, height, scale, colour):
		image = pygame.Surface((width, height)).convert_alpha()
		image.blit(self.sheet, (0, 0), ((frame * width), 0, width, height))
		image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
		image.set_colorkey(colour)

		return image


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


# initialize pygame

class Platformer():
    def __init__(self):
        
        self.stuff = {
            "SCREEN_WIDTH":400,
            "SCREEN_HEIGHT": 600,
            "screen": pygame.Surface((400, 600)),
            "GRAVITY": 1,
            "MAX_PLATFORMS": 10,
            "SCROLL_THRESH": 200,
            "scroll": 0,
            "bg_scroll": 0,
            "game_over": False,
            "score": 0,
            "fade_counter": 0,
            "high_score": None,
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "PANEL": (153, 217, 234),
            "font_small": pygame.font.Font('platformerGame/Catfiles.ttf', 24),
            "font_big": pygame.font.Font('platformerGame/Catfiles.ttf', 30),
            "bg_image": pygame.image.load('platformerGame/assets/bg.png').convert_alpha(),
            "jumpy_image": pygame.image.load('platformerGame/assets/jump.png').convert_alpha(),
            "platform_image": pygame.image.load('platformerGame/assets/wood.png').convert_alpha(),
            "bird_sheet_img": pygame.image.load('platformerGame/assets/bird.png').convert_alpha(),
            "bird_sheet": SpriteSheet(pygame.image.load('platformerGame/assets/bird.png').convert_alpha()),
            "platform_group": None,
            "platform": None
        }
        
        self.surface = self.stuff["screen"]
        
                # player instance
        self.jumpy = Player(400//2, 600, self.stuff)

        # create sprite groups
        self.platform_group = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()

        # create starting platform
        self.platform = Platform(400//2 - 50, 600-50, 100, False, self.stuff)
        self.platform_group.add(self.platform)
        
        self.stuff["platform_group"] = self.platform_group
        self.stuff["platform"] = self.platform

                
        if os.path.exists('platformerGame/score.txt'):
            with open('platformerGame/score.txt', 'r') as file:
                self.stuff["high_score"] = int(file.read())
        else:
            self.stuff["high_score"] = 0
            
            
            
        # function for outputting the text on the sceen
    def draw_text(self, text, font, text_col, x, y, stuff):
        img = font.render(text, True, text_col)
        stuff["screen"].blit(img, (x,y))

    # function for drawing info panel
    def draw_panel(self, stuff):
        pygame.draw.rect(stuff["screen"], stuff["PANEL"], (0,0, stuff["SCREEN_WIDTH"], 30))
        pygame.draw.line(stuff["screen"], stuff["BLACK"], (0, 30), (stuff["SCREEN_WIDTH"], 30), 2)
        self.draw_text('SCORE: ' + str(stuff["score"]), stuff["font_small"], stuff["BLACK"], 0, 0, stuff)

    # function for drawing the background
    def draw_bg(self, stuff):
        stuff["screen"].blit(stuff["bg_image"], (0,0 + stuff["bg_scroll"]))
        stuff["screen"].blit(stuff["bg_image"], (0, -600 + stuff["bg_scroll"]))
            
    def tick(self):
        if self.stuff["game_over"] == False:
            self.stuff["scroll"] = self.jumpy.move(self.stuff)

            #draw background
            self.stuff["bg_scroll"] += self.stuff["scroll"]
            if self.stuff["bg_scroll"] >= 600:
                self.stuff["bg_scroll"] = 0
            self.draw_bg(self.stuff)

            #generate platforms
            if len(self.platform_group) < self.stuff["MAX_PLATFORMS"]:
                p_w = random.randint(50, 70)
                p_x = random.randint(0, self.stuff["SCREEN_WIDTH"]- p_w)
                p_y = self.stuff["platform"].rect.y - random.randint(80, 120)
                p_type = random.randint(1, 2)
                if p_type == 1 and self.stuff["score"] > 1000:
                    p_moving = True
                else:
                    p_moving = False
                self.stuff["platform"] = Platform(p_x, p_y, p_w, p_moving, self.stuff)
                self.stuff["platform_group"].add(self.stuff["platform"])

            #update platforms
            self.platform_group.update(self.stuff)

            #generate enemies
            if len(self.enemy_group) == 0 and self.stuff["score"] > 1500:
                enemy = Enemy(self.stuff["SCREEN_WIDTH"], 100, self.stuff["bird_sheet"], 1.5)
                self.enemy_group.add(enemy)

            #update enemies
            self.enemy_group.update(self.stuff["scroll"], self.stuff["SCREEN_WIDTH"])

            #update score
            if self.stuff["scroll"] > 0:
                self.stuff["score"] += self.stuff["scroll"]

            #draw line at previous high score
            pygame.draw.line(self.stuff["screen"], self.stuff["WHITE"], (0, self.stuff["score"] - self.stuff["high_score"] + self.stuff["SCROLL_THRESH"]), (self.stuff["SCREEN_WIDTH"], self.stuff["score"] - self.stuff["high_score"] + self.stuff["SCROLL_THRESH"]), 3)
            self.draw_text('HIGH SCORE', self.stuff["font_small"], self.stuff["WHITE"], self.stuff["SCREEN_WIDTH"]- 130, self.stuff["score"] - self.stuff["high_score"] + self.stuff["SCROLL_THRESH"], self.stuff)

            #draw sprites
            self.platform_group.draw(self.stuff["screen"])
            self.enemy_group.draw(self.stuff["screen"])
            self.jumpy.draw(self.stuff)

            #draw panel
            self.draw_panel(self.stuff)

            #check game over
            if self.jumpy.rect.top > self.stuff["SCREEN_HEIGHT"]:
                self.stuff["game_over"] = True

            #check for collision with enemies
            if pygame.sprite.spritecollide(self.jumpy, self.enemy_group, False):
                if pygame.sprite.spritecollide(self.jumpy, self.enemy_group, False, pygame.sprite.collide_mask):
                    self.stuff["game_over"] = True

        else:
            if self.stuff["fade_counter"] < self.stuff["SCREEN_WIDTH"]:
                self.stuff["fade_counter"] += 5
                for y in range(0, 6, 2):
                    pygame.draw.rect(self.stuff["screen"], self.stuff["BLACK"], (0, y * 100, self.stuff["fade_counter"], 100))
                    pygame.draw.rect(self.stuff["screen"], self.stuff["BLACK"], (self.stuff["SCREEN_WIDTH"]- self.stuff["fade_counter"], (y + 1) * 100, self.stuff["SCREEN_WIDTH"], 100))
            else:
                self.draw_text('GAME OVER!', self.stuff["font_big"], self.stuff["WHITE"], 100, 200, self.stuff)
                self.draw_text('SCORE: ' + str(self.stuff["score"]), self.stuff["font_big"], self.stuff["WHITE"], 115, 250, self.stuff)
                self.draw_text('PRESS SPACE', self.stuff["font_big"], self.stuff["WHITE"], 95, 300, self.stuff)
                self.draw_text('TO PLAY AGAIN', self.stuff["font_big"], self.stuff["WHITE"], 80, 350, self.stuff)
                # update high score
                if self.stuff["score"] > self.stuff["high_score"]:
                    self.stuff["high_score"] = self.stuff["score"]
                    with open('platformerGame/score.txt', 'w') as file:
                        file.write(str(self.stuff["high_score"]))
                key = pygame.key.get_pressed()
                if key[pygame.K_SPACE]:
                    #reset variables
                    self.stuff["game_over"] = False
                    self.stuff["score"] = 0
                    self.stuff["scroll"] = 0
                    self.stuff["fade_counter"] = 0
                    #reposition jumpy
                    self.jumpy.rect.center = (self.stuff["SCREEN_WIDTH"]// 2, self.stuff["SCREEN_HEIGHT"] - 150)
                    #reset enemies
                    self.enemy_group.empty()
                    #reset platforms
                    self.platform_group.empty()
                    #create starting platform
                    platform = Platform(self.stuff["SCREEN_WIDTH"]// 2 - 50, self.stuff["SCREEN_HEIGHT"] - 50, 100, False, self.stuff)
                    self.platform_group.add(platform)
                    
    def events(self, events):
        #event handler
        for event in events:
            if event.type == pygame.QUIT:
                #update high score
                if self.stuff["score"] > self.stuff["high_score"]:
                    self.stuff["high_score"] = self.stuff["score"]
                    with open('platformerGame/score.txt', 'w') as file:
                        file.write(str(self.stuff["high_score"]))
                run = False





# player class
class Player():
    def __init__(self, x, y, stuff):
        self.image = pygame.transform.scale(stuff["jumpy_image"], (45,45))
        self.width = 25
        self.height = 40
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.rect.center = (x,y)
        self.vel_y = 0
        self.flip = False
        self.stuff = stuff
    
    def move(self, stuff):
        # reset variables
        dx = 0
        dy = 0
        stuff["scroll"] = 0
        # process key presses
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            dx = -10
            self.flip = True
        if key[pygame.K_RIGHT]:
            dx = 10
            self.flip = False
        
        # stuff["GRAVITY"]
        self.vel_y += stuff["GRAVITY"]
        dy += self.vel_y
        
        # making sure character doesnt move off of stuff["screen"]
        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > stuff["SCREEN_WIDTH"]:
            dx = stuff["SCREEN_WIDTH"]- self.rect.right
        
        # check collision with platforms
        for platform in self.stuff["platform_group"]:
            # collision in the y direction
            if platform.rect.colliderect(self.rect.x, self.rect.y + dy, self.width, self.height):
                # check if above the platform
                if self.rect.bottom < platform.rect.centery:
                    if self.vel_y > 0:
                        self.rect.bottom = platform.rect.top
                        dy = 0
                        self.vel_y = -20

    
        # check if the player has bounced to the top of the stuff["screen"]
        if self.rect.top <= stuff["SCROLL_THRESH"]:
            # if player is jumping
            if self.vel_y < 0:
                stuff["scroll"] = -dy

        # update position
        self.rect.x += dx
        self.rect.y += dy + stuff["scroll"]

        # update mask
        self.mask = pygame.mask.from_surface(self.image)

        return stuff["scroll"]

    def draw(self, stuff):
        stuff["screen"].blit(pygame.transform.flip(self.image, self.flip, False), (self.rect.x - 12, self.rect.y - 5))
        # pygame.draw.rect(stuff["screen"], WHITE, self.rect, 2)

# platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, moving, stuff):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(stuff["platform_image"], (width, 10))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.moving = moving
        self.move_counter = random.randint(0, 50)
        self.direction = random.choice([-1,1])
        self.speed = random.randint(1,2)
    
    def update(self, stuff):
        # move moving platforms side to side
        if self.moving == True:
            self.move_counter += 1
            self.rect.x += self.direction * self.speed
            


        # change platform direction if it has moved fully
        if self.move_counter >= 100 or self.rect.left < 0 or self.rect.right > stuff["SCREEN_WIDTH"]:
            self.direction *= -1
            self.move_counter = 0

        # update platforms vertical position
        self.rect.y += stuff["scroll"]

        # check if platform has gone off the stuff["screen"]
        if self.rect.top > stuff["SCREEN_HEIGHT"]:
            self.kill()

