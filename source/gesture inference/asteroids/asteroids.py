import math
import random
import pygame
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# images used for game
bg = pygame.image.load("asteroidpics/starbg.webp")
alien = pygame.image.load("asteroidpics/alienShip.png")
playerShip = pygame.image.load("asteroidpics/spaceRocket.png")
star = pygame.image.load("asteroidpics/star.png")
asteroid1 = pygame.image.load("asteroidpics/asteroid1.png")
asteroid2 = pygame.image.load("asteroidpics/asteroid2.png")
asteroid3 = pygame.image.load("asteroidpics/asteroid3.png")



# game surface set up
pygame.display.set_caption("Asteroids")
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# game state vars
run = True
clock = pygame.time.Clock()
game_over = False
lives = 3
score = 0
rapid_fire = False
rapid_fire_start = -1


# Player class + attributes and methods
class Player(object):
    def __init__(self):
        self.img = playerShip
        self.width = self.img.get_width()
        self.height = self.img.get_height()
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.angle = 0
        # rotate the player ship based on the curr angle
        self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
        self.rotated_rect = self.rotated_surface.get_rect()
        self.rotated_rect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        # calculates the front point of the ship for shooting the bullets
        self.head = (
            self.x + self.cosine * self.width // 2,
            self.y - self.sine * self.height // 2,
        )

    # draw the rotated player ship on the surface
    def draw(self, surface):
        surface.blit(self.rotated_surface, self.rotated_rect)

    # turn the ship left by increasing the angle
    def turn_left(self):
        self.angle += 5
        self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
        self.rotated_rect = self.rotated_surface.get_rect()
        self.rotated_rect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (
            self.x + self.cosine * self.width // 2,
            self.y - self.sine * self.height / 2,
        )

    # turn the ship right by decreasing the angle
    def turn_right(self):
        self.angle -= 5
        self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
        self.rotated_rect = self.rotated_surface.get_rect()
        self.rotated_rect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (
            self.x + self.cosine * self.width // 2,
            self.y - self.sine * self.height / 2,
        )

    # move the ship forward in the direction it is facing
    def move_forward(self):
        self.x += self.cosine * 6
        self.y -= self.sine * 6
        self.rotated_surface = pygame.transform.rotate(self.img, self.angle)
        self.rotated_rect = self.rotated_surface.get_rect()
        self.rotated_rect.center = (self.x, self.y)
        self.cosine = math.cos(math.radians(self.angle + 90))
        self.sine = math.sin(math.radians(self.angle + 90))
        self.head = (
            self.x + self.cosine * self.width // 2,
            self.y - self.sine * self.height // 2,
        )

    # update the location of the ship when it goes off the screen
    def update_location(self):
        if self.x > SCREEN_WIDTH + 50:
            self.x = 0
        elif self.x < 0 - self.width:
            self.x = SCREEN_WIDTH
        elif self.y < -50:
            self.y = SCREEN_HEIGHT
        elif self.y > SCREEN_HEIGHT + 50:
            self.y = 0
            
# game entities
player = Player()
player_bullets = []
asteroids = []
count = 0
stars = []
aliens = []
alien_bullets = []


# Bullet class + attributes and methods
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

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), [self.x, self.y, self.w, self.h])

    def check_off_screen(self):
        if (
            self.x < -50
            or self.x > SCREEN_WIDTH
            or self.y > SCREEN_HEIGHT
            or self.y < -50
        ):
            return True


# Asteroid class + attributes and methods
class Asteroid(object):
    def __init__(self, rank):
        self.rank = rank
        if self.rank == 1:
            self.image = asteroid1
        elif self.rank == 2:
            self.image = asteroid2
        else:
            self.image = asteroid3
        self.w = 50 * rank
        self.h = 50 * rank
        self.ran_point = random.choice(
            [
                (
                    random.randrange(0, SCREEN_WIDTH - self.w),
                    random.choice([-1 * self.h - 5, SCREEN_HEIGHT + 5]),
                ),
                (
                    random.choice([-1 * self.w - 5, SCREEN_WIDTH + 5]),
                    random.randrange(0, SCREEN_HEIGHT - self.h),
                ),
            ]
        )
        self.x, self.y = self.ran_point
        if self.x < SCREEN_WIDTH // 2:
            self.xdir = 1
        else:
            self.xdir = -1
        if self.y < SCREEN_HEIGHT // 2:
            self.ydir = 1
        else:
            self.ydir = -1
        self.xv = self.xdir * random.randrange(1, 3)
        self.yv = self.ydir * random.randrange(1, 3)

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))


# Star class + attributes and methods
class Star(object):
    def __init__(self):
        self.img = star
        self.w = self.img.get_width()
        self.h = self.img.get_height()
        self.ran_point = random.choice(
            [
                (
                    random.randrange(0, SCREEN_WIDTH - self.w),
                    random.choice([-1 * self.h - 5, SCREEN_HEIGHT + 5]),
                ),
                (
                    random.choice([-1 * self.w - 5, SCREEN_WIDTH + 5]),
                    random.randrange(0, SCREEN_HEIGHT - self.h),
                ),
            ]
        )
        self.x, self.y = self.ran_point
        if self.x < SCREEN_WIDTH // 2:
            self.xdir = 1
        else:
            self.xdir = -1
        if self.y < SCREEN_HEIGHT // 2:
            self.ydir = 1
        else:
            self.ydir = -1
        self.xv = self.xdir * 2
        self.yv = self.ydir * 2

    def draw(self, surface):
        surface.blit(self.img, (self.x, self.y))


# Alien class + attributes and methods
class Alien(object):
    def __init__(self):
        self.img = alien
        self.w = self.img.get_width()
        self.h = self.img.get_height()
        self.ran_point = random.choice(
            [
                (
                    random.randrange(0, SCREEN_WIDTH - self.w),
                    random.choice([-1 * self.h - 5, SCREEN_HEIGHT + 5]),
                ),
                (
                    random.choice([-1 * self.w - 5, SCREEN_WIDTH + 5]),
                    random.randrange(0, SCREEN_HEIGHT - self.h),
                ),
            ]
        )
        self.x, self.y = self.ran_point
        if self.x < SCREEN_WIDTH // 2:
            self.xdir = 1
        else:
            self.xdir = -1
        if self.y < SCREEN_HEIGHT // 2:
            self.ydir = 1
        else:
            self.ydir = -1
        self.xv = self.xdir * 2

        self.yv = self.ydir * 2

    def draw(self, surface):
        surface.blit(self.img, (self.x, self.y))


# Alien Bullets class + attributes and methods
class AlienBullet(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = 4
        self.h = 4
        self.dx, self.dy = player.x - self.x, player.y - self.y
        self.dist = math.hypot(self.dx, self.dy)
        self.dx, self.dy = self.dx / self.dist, self.dy / self.dist
        self.xv = self.dx * 5
        self.yv = self.dy * 5

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), [self.x, self.y, self.w, self.h])


# updates game surface with the current game state
def redraw_window():
    surface.blit(bg, (0, 0))
    font = pygame.font.SysFont("arial", 20)
    livesText = font.render("Lives: " + str(lives), 1, (255, 255, 255))
    playe_again_text = font.render(
        "Press the Space Bar to Player Again!", 1, (255, 255, 255)
    )
    score_text = font.render("Score: " + str(score), 1, (255, 255, 255))
    player.draw(surface)
    for a in asteroids:
        a.draw(surface)
    for b in player_bullets:
        b.draw(surface)
    for s in stars:
        s.draw(surface)
    for a in aliens:
        a.draw(surface)
    for b in alien_bullets:
        b.draw(surface)

    if rapid_fire:
        pygame.draw.rect(surface, (0, 0, 0), [SCREEN_WIDTH // 2 - 51, 19, 102, 22])
        pygame.draw.rect(
            surface,
            (255, 255, 255),
            [
                SCREEN_WIDTH // 2 - 50,
                20,
                100 - 100 * (count - rapid_fire_start) / 500,
                20,
            ],
        )
    if game_over:
        surface.blit(
            playe_again_text,
            (
                SCREEN_WIDTH // 2 - playe_again_text.get_width() // 2,
                SCREEN_HEIGHT // 2 - playe_again_text.get_height() // 2,
            ),
        )
    surface.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 25, 25))
    surface.blit(livesText, (25, 25))
    pygame.display.update()



# game loop
def tick():
    global game_over
    global player_bullets 
    global player
    global asteroids 
    global count
    global stars 
    global aliens 
    global alien_bullets 
    global lives
    global ran
    global rapid_fire_start
    global rapid_fire

    clock.tick(60)
    count += 1

    if not game_over:
        if count % 50 == 0:
            ran = random.choice([1, 1, 1, 2, 2, 3])
            asteroids.append(Asteroid(ran))
        if count % 1000 == 0:
            stars.append(Star())
        if count % 750 == 0:
            aliens.append(Alien())
        for i, a in enumerate(aliens):
            a.x += a.xv
            a.y += a.yv
            if (
                a.x > SCREEN_WIDTH + 150
                or a.x + a.w < -100
                or a.y > SCREEN_HEIGHT + 150
                or a.y + a.h < -100
            ):
                aliens.pop(i)
            if count % 60 == 0:
                alien_bullets.append(AlienBullet(a.x + a.w // 2, a.y + a.h // 2))
            for b in player_bullets:
                if (
                    (b.x >= a.x and b.x <= a.x + a.w)
                    or b.x + b.w >= a.x
                    and b.x + b.w <= a.x + a.w
                ):
                    if (
                        (b.y >= a.y and b.y <= a.y + a.h)
                        or b.y + b.h >= a.y
                        and b.y + b.h <= a.y + a.h
                    ):
                        aliens.pop(i)
                        score += 50
                        break
        for i, b in enumerate(alien_bullets):
            b.x += b.xv
            b.y += b.yv

            if (
                (
                    b.x >= player.x - player.width // 2
                    and b.x <= player.x + player.width // 2
                )
                or b.x + b.w >= player.x - player.width // 2
                and b.x + b.w <= player.x + player.width // 2
            ):
                if (
                    (
                        b.y >= player.y - player.height // 2
                        and b.y <= player.y + player.height // 2
                    )
                    or b.y + b.h >= player.y - player.height // 2
                    and b.y + b.h <= player.y + player.height // 2
                ):
                    lives -= 1
                    alien_bullets.pop(i)
                    break

        player.update_location()
        for b in player_bullets:
            b.move()
            if b.check_off_screen():
                player_bullets.pop(player_bullets.index(b))

        for a in asteroids:
            a.x += a.xv
            a.y += a.yv

            if (
                a.x >= player.x - player.width // 2
                and a.x <= player.x + player.width // 2
            ) or (
                a.x + a.w <= player.x + player.width // 2
                and a.x + a.w >= player.x - player.width // 2
            ):
                if (
                    a.y >= player.y - player.height // 2
                    and a.y <= player.y + player.height // 2
                ) or (
                    a.y + a.h >= player.y - player.height // 2
                    and a.y + a.h <= player.y + player.height // 2
                ):
                    lives -= 1
                    asteroids.pop(asteroids.index(a))
                    break

            # breaking larger asteroids into smaller ones when shot at
            for b in player_bullets:
                if (
                    (b.x >= a.x and b.x <= a.x + a.w)
                    or b.x + b.w >= a.x
                    and b.x + b.w <= a.x + a.w
                ):
                    if (
                        (b.y >= a.y and b.y <= a.y + a.h)
                        or b.y + b.h >= a.y
                        and b.y + b.h <= a.y + a.h
                    ):
                        # breaks asteroids and adds to score based on asteroids type
                        if a.rank == 3:
                            score += 10
                            na1 = Asteroid(2)
                            na2 = Asteroid(2)
                            na1.x = a.x
                            na2.x = a.x
                            na1.y = a.y
                            na2.y = a.y
                            asteroids.append(na1)
                            asteroids.append(na2)
                        elif a.rank == 2:
                            score += 20
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
                        break

        for s in stars:
            s.x += s.xv
            s.y += s.yv
            if (
                s.x < -100
                or s.x > SCREEN_WIDTH + 100
                or s.y > SCREEN_HEIGHT + 100
                or s.y < -100 - SCREEN_HEIGHT
            ):
                stars.pop(stars.index(s))
                break
            for b in player_bullets:
                if (
                    (b.x >= s.x and b.x <= s.x + s.w)
                    or b.x + b.w >= s.x
                    and b.x + b.w <= s.x + s.w
                ):
                    if (
                        (b.y >= s.y and b.y <= s.y + s.h)
                        or b.y + b.h >= s.y
                        and b.y + b.h <= s.y + s.h
                    ):
                        rapid_fire = True
                        rapid_fire_start = count
                        stars.pop(stars.index(s))
                        player_bullets.pop(player_bullets.index(b))
                        break

        if lives <= 0:
            game_over = True
        if rapid_fire_start != -1:
            if count - rapid_fire_start > 500:
                rapid_fire = False
                rapid_fire_start = -1

        # keys for movement/rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.turn_left()
        if keys[pygame.K_RIGHT]:
            player.turn_right()
        if keys[pygame.K_UP]:
            player.move_forward()
        if keys[pygame.K_SPACE]:
            if rapid_fire:
                player_bullets.append(Bullet())

    # redraws the surface with the game's current state
    redraw_window()


def events(events):
    
    global lives
    global score
    global asteroids
    global alien_bullets
    global stars
    global player_bullets
    global player
    global rapid_fire
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not game_over:
                    if not rapid_fire:
                        player_bullets.append(Bullet())
                else:
                    game_over = False
                    lives = 3
                    score = 0
                    asteroids.clear()
                    alien_bullets.clear()
                    stars.clear()
            # pause key
            elif event.key == pygame.K_p:
                paused = True
                while paused:
                    for pause_event in pygame.event.get():
                        if pause_event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        elif pause_event.type == pygame.KEYDOWN:
                            if pause_event.key == pygame.K_p:
                                paused = False
                            elif pause_event.key == pygame.K_q:
                                pygame.quit()
                                exit()

                    pygame.time.wait(100)
            # quit key
            elif event.key == pygame.K_q:
                pass
