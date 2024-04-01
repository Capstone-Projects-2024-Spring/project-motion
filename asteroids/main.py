import math
import random
import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# images used for game
bg = pygame.image.load('asteroidpics/starbg.webp')
alien = pygame.image.load('asteroidpics/alienShip.png')
playerShip = pygame.image.load('asteroidpics/spaceRocket.png')
star = pygame.image.load('asteroidpics/star.png')
asteroid1 = pygame.image.load('asteroidpics/asteroid1.png')
asteroid2 = pygame.image.load('asteroidpics/asteroid2.png')
asteroid3 = pygame.image.load('asteroidpics/asteroid3.png')

pygame.display.set_caption('Asteroids')
window = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))


run = True
clock = pygame.time.Clock()
game_over = False
lives = 3
score = 0

class Player(object):
     def __init__(self):
         self.img = playerShip
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
          self.head = (self.x + self.cosine * self.width//2, self.y - self.sine * self.height/2)
     
     def turn_right(self):
          self.angle -=5
          self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
          self.rotated_rect = self.rotated_surface.get_rect()
          self.rotated_rect.center = (self.x,self.y)
          self.cosine = math.cos(math.radians(self.angle + 90))
          self.sine = math.sin(math.radians(self.angle +90))
          self.head = (self.x + self.cosine * self.width//2, self.y - self.sine * self.height/2)     

     def move_forward(self):
        self.x += self.cosine * 6
        self.y -= self.sine * 6
        self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
        self.rotated_rect = self.rotated_surface.get_rect()
        self.rotated_rect.center = (self.x, self.y)
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
          # bullets are 4 pixels big
          self.w = 4
          self.h = 4
          self.c = player.cosine
          self.s = player.sine
          self.xv = self.c * 10
          self.yv = self.s * 10
     def move(self):
          self.x += self.xv
          self.y -= self.yv

     def draw(self,window):
          pygame.draw.rect(window, (255,255,255),[self.x,self.y,self.w,self.h])  
     def check_off_screen(self):
          if self.x < -50 or self.x > SCREEN_WIDTH or self.y > SCREEN_HEIGHT or self.y < -50:
                        return True    

class Asteroid(object):
     def __init__(self, rank):
          self.rank = rank
          if self.rank ==1:
                 self.image = asteroid1
          elif self.rank ==2:
                 self.image = asteroid2
          else:
                 self.image = asteroid3
          self.w = 50 * rank
          self.h = 50 * rank
          self.ran_point = random.choice([(random.randrange(0, SCREEN_WIDTH-self.w), random.choice([-1*self.h - 5, SCREEN_HEIGHT + 5])), (random.choice([-1*self.w - 5, SCREEN_WIDTH + 5]), random.randrange(0, SCREEN_HEIGHT - self.h))])
          self.x, self.y = self.ran_point
          if self.x < SCREEN_WIDTH//2:
               self.xdir = 1
          else:
               self.xdir = -1
          if self.y < SCREEN_HEIGHT//2:
               self.ydir = 1
          else:
               self.ydir = -1
          self.xv = self.xdir * random.randrange(1,3)
          self.yv = self.ydir * random.randrange(1,3)

     def draw(self, window):
          window.blit(self.image, (self.x, self.y))                      


def redraw_window():
        window.blit(bg,(0,0))
        font = pygame.font.SysFont('arial',20)
        livesText = font.render('Lives: ' + str(lives),1,(255,255,255))
        playe_again_text = font.render('Press the Space Bar to Player Again!', 1, (255,255,255))
        score_text = font.render('Score: ' + str(score),1,(255,255,255))
        player.draw(window)
        for a in asteroids:
             a.draw(window)
        for b in player_bullets:
            b.draw(window)
        if game_over:
            window.blit(playe_again_text,(SCREEN_WIDTH//2-playe_again_text.get_width()//2, SCREEN_HEIGHT//2 - playe_again_text.get_height()//2))
        window.blit(score_text,(SCREEN_WIDTH-score_text.get_width()-25,25))
        window.blit(livesText,(25,25))
        pygame.display.update()

player = Player()
player_bullets = []
asteroids = []
count = 0
# game loop
while run:
    clock.tick(60)
    count +=1
    if not game_over: 
         if count%50 == 0:
              ran = random.choice([1,1,1,2,2,3])
              asteroids.append(Asteroid(ran))
         player.update_location()
         for b in player_bullets:
              b.move()
              if b.check_off_screen():
                  player_bullets.pop(player_bullets.index(b))


         for a in asteroids:
              a.x += a.xv
              a.y += a.yv


              if (player.x >= a.x and player.x <= a.x +a.w) or (player.x + player.width >= a.x and player.x + player.width <= a.x + a.w):
                   if (player.y >= a.y and player.y <= a.y +a.h) or (player.y + player.height >= a.y and player.y + player.height <= a.y +a.h ):
                       lives -= 1
                       asteroids.pop(asteroids.index(a))
                       break

               # breaking larger asteroids into smaller ones when shot at
              for b in player_bullets:
                  if (b.x >= a.x and b.x <= a.x + a.w) or b.x + b.w >= a.x and b.x + b.w <= a.x + a.w:
                      if (b.y >= a.y and b.y <= a.y + a.h) or b.y + b.h >= a.y and b.y + b.h <= a.y + a.h:
                           if a.rank ==3:
                                score += 10
                                na1 = Asteroid(2)
                                na2 = Asteroid(2)
                                na1.x = a.x
                                na2.x = a.x
                                na1.y = a.y
                                na2.y = a.y 
                                asteroids.append(na1)
                                asteroids.append(na2)
                           elif a.rank ==2:
                                score +=20
                                na1 = Asteroid(1)
                                na2 = Asteroid(1)
                                na1.x = a.x
                                na2.x = a.x
                                na1.y = a.y
                                na2.y = a.y 
                                asteroids.append(na1)
                                asteroids.append(na2)
                           else:
                                score += 30     
                           asteroids.pop(asteroids.index(a))
                           player_bullets.pop(player_bullets.index(b))

         if lives <= 0:
              game_over = True                  

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
                     player_bullets.append(Bullet()) 
                 else:
                      game_over = False
                      lives = 3
                      score = 0
                      asteroids.clear()

    redraw_window()
pygame.quit()