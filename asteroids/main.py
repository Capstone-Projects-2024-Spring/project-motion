import math
import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# images used for game
bg = pygame.image.load('asteroidpics/starbg.webp')
alien = pygame.image.load('asteroidpics/alienShip.png')
player_rocket = pygame.image.load('asteroidpics/spaceRocket.png')
star = pygame.image.load('asteroidpics/star.png')
asteroid1 = pygame.image.load('asteroidpics/asteroid1.png')
asteroid2 = pygame.image.load('asteroidpics/asteroid2.png')
asteroid3 = pygame.image.load('asteroidpics/asteroid3.png')

pygame.display.set_caption('Asteroids')
window = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))


run = True
clock = pygame.time.Clock()
game_over = False

class Player(object):
     def __init__(self):
         self.img = player_rocket
         self.width = self.img.get_width()
         self.height = self.img.get_height()
         self.x = SCREEN_WIDTH//2 
         self.y = SCREEN_HEIGHT//2 
         self.angle = 0
         self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
         self.rotated_rect = self.rotated_surface.get_rect()
         self.rotated_rect.center = (self.x,self.y)
         self.cosine = math.cos(math.radians(self.angle + 90))
         self.sine = math.sin(math.radians(self.angle + 90))
         self.head = (self.x + self.cosine * self.width//2, self.y - self.sine * self.height//2)

     def draw(self,window):
          window.blit(self.rotated_surface,self.rotated_rect)

     def turn_left(self):
          self.angle +=5
          self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
          self.rotated_rect = self.rotated_surface.get_rect()
          self.rotated_rect.center = (self.x,self.y)
          self.cosine = math.cos(math.radians(self.angle + 90))
          self.sine = math.sin(math.radians(self.angle +90))
          self.head = (self.x + self.cosine + self.width//2, self.y - self.sine * self.height/2)
     
     def turn_right(self):
          self.angle -=5
          self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
          self.rotated_rect = self.rotated_surface.get_rect()
          self.rotated_rect.center = (self.x,self.y)
          self.cosine = math.cos(math.radians(self.angle + 90))
          self.sine = math.sin(math.radians(self.angle +90))
          self.head = (self.x + self.cosine + self.width//2, self.y - self.sine * self.height/2)     

     def move_forward(self):
        self.x += self.cosine * 6
        self.y -= self.sine * 6
        self.rotatedSurf = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedSurf.get_rect()
        self.rotatedRect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (self.x + self.cosine * self.width // 2, self.y - self.sine * self.height // 2)

     def update_location(self):
          if self.x > SCREEN_WIDTH+ 50:
            self.x = 0
          elif self.x < 0 - self.width:
            self.x = SCREEN_WIDTH
          elif self.y < -50:
            self.y = SCREEN_HEIGHT
          elif self.y > SCREEN_HEIGHT + 50:
            self.y = 0 


class Bullet(object):
     def __init__(self):
          self.point = player.head 
          self.x, self.y = self.point
          self.width = 4
          self.height = 4
          self.c = player.cosine 
          self.s = player.sine 
          self.xv = self.c * 10
          self.yv = self.s * 10

     def move(self):
        self.x += self.xv
        self.y -= self.yv

     def draw(self, window):
        pygame.draw.rect(window, (255, 255, 255), [self.x, self.y, self.width, self.height])
     
     def check_off_screen(self):
          if self.x < -50 or self.x > SCREEN_WIDTH or self.y > SCREEN_HEIGHT or self.y < -50:
               return True    




              

def redraw_window():
        window.blit(bg,(0,0))
        player.draw(window)

        for b in bullets:
             b.draw(window)
             

        pygame.display.update()

player = Player()
bullets = []
# game loop
while run:
    clock.tick(60)
    if not game_over: 
         player.update_location()
         for b in bullets:
              b.move() 
              if b.check_off_screen():
                  bullets.pop(bullets.index(b))

         keys = pygame.key.get_pressed()
         if keys[pygame.K_LEFT]:
                  player.turn_left()
         if keys[pygame.K_RIGHT]:
              player.turn_right()
         if keys[pygame.K_UP]:  
              player.move_forward()
                    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game_over:
                         bullets.append(Bullet())
    redraw_window()
pygame.quit()
