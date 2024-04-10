import pygame
from pygame.locals import *
import random
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class FlappyBirdGame:
    def __init__(self):

        self.surface = pygame.Surface((864, 936))
        (screen_width, screen_height) = self.surface.get_size()
        self.screen_width = screen_width
        self.screen_height = screen_height
        pygame.display.set_caption("Flappy Stella")

        self.font = pygame.font.SysFont("Bauhaus 93", 60)
        self.white = (255, 255, 255)

        self.primitives = {
            "ground scroll": 0,
            "scroll speed": 4,
            "is flying": False,
            "is game over": False,
            "is paused": False,
            "pipe gap": 175,
            "pipe frequency": 1500,
            "last pipe": pygame.time.get_ticks() - 1500,
            "score": 0,
            "pass pipe": False,
            "is started": False,
        }

        self.bg = pygame.image.load("FlappyBird/img/bg.png")
        self.ground_img = pygame.image.load("FlappyBird/img/ground.png")
        self.restart_button_img = pygame.image.load("FlappyBird/img/restart.png")
        self.exit_button_img = pygame.image.load("FlappyBird/img/exit.png")
        self.start_img = pygame.image.load("FlappyBird/img/start.png")

        self.bird_group = pygame.sprite.Group()
        self.pipe_group = pygame.sprite.Group()
        self.flappy = self.create_bird(100, int(self.screen_height / 2))
        self.bird_group.add(self.flappy)
        
        self.restart_button = self.create_button(
            self.screen_width // 2 - 50,
            self.screen_height // 2 - 125,
            self.restart_button_img,
        )
        self.exit_button = self.create_button(
            self.screen_width // 2 - 50,
            self.screen_height // 2 - 50,
            self.exit_button_img,
        )
        self.start_image = self.create_show(
            (self.screen_width // 2 - self.start_img.get_width() // 2),
            (self.screen_height // 2 - self.start_img.get_height() // 2),
            self.start_img,
        )
        
    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pass
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
                and self.primitives["is flying"] == False
                and self.primitives["is game over"] == False
            ):
                self.primitives["is flying"] = True
                self.primitives["is started"] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.primitives["is paused"] = not self.primitives["is paused"]

    def create_bird(self, x, y):
        return Bird(x, y, self.primitives)

    def create_button(self, x, y, image):
        return Button(x, y, image, self.surface)

    def create_show(self, x, y, image):
        return Show(x, y, image, self.surface)

    def get_high_score(self):
        try:
            with open("FlappyBird/highscore.txt", "r") as file:
                high_score = int(file.readline())
                return f"Highscore: {high_score}"
        except FileNotFoundError:
            return "Highscore: No high score recorded yet"

    def highscore_check(self, score):
        try:
            with open("FlappyBird/highscore.txt", "r") as file:
                current_score = int(file.readline())
        except FileNotFoundError:
            with open("FlappyBird/highscore.txt", "w") as file:
                file.write(str(score))
            return

        if score > current_score:
            with open("FlappyBird/highscore.txt", "w") as file:
                file.write(str(score))

    def draw_text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.surface.blit(img, (x, y))

    def reset_game(self):
        self.pipe_group.empty()
        self.flappy.rect.x = 100
        self.flappy.rect.y = int(self.screen_height / 2)
        self.primitives["score"] = 0  # score = 0

    def tick(self):
        self.surface.blit(self.bg, (0, 0))

        self.bird_group.draw(self.surface)
        self.bird_group.update()

        self.pipe_group.draw(self.surface)

        self.surface.blit(self.ground_img, (self.primitives["ground scroll"], 768))  # ground_scroll

        if len(self.pipe_group) > 0:
            if (
                self.bird_group.sprites()[0].rect.left
                > self.pipe_group.sprites()[0].rect.left
                and self.bird_group.sprites()[0].rect.right
                < self.pipe_group.sprites()[0].rect.right
                and not self.primitives["pass pipe"]  # pass_pipe
            ):
                self.primitives["pass pipe"] = True  # pass_pipe = True
            if self.primitives["pass pipe"]:  # pass_pipe
                if (
                    self.bird_group.sprites()[0].rect.left
                    > self.pipe_group.sprites()[0].rect.right
                ):
                    self.primitives["score"] += 1  # score += 1
                    self.primitives["pass pipe"] = False  # pass_pipe = False

        self.draw_text(
            self.get_high_score(),
            self.font,
            self.white,
            int(self.screen_width / 2 - 100),
            20,
        )
        self.draw_text(
            str(self.primitives["score"]),
            self.font,
            self.white,
            int(self.screen_width / 2),
            75,  # score
        )

        if (
            not self.primitives["is started"]
            and not self.primitives["is game over"]
            and not self.primitives["is flying"]
        ):  # started, game_over, flying
            self.start_image.draw()

        if (
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False)
            or self.flappy.rect.top < 0
        ):
            self.primitives["is game over"] = True  # game_over = True

        if self.flappy.rect.bottom >= 768:
            self.primitives["is game over"] = True  # game_over = True
            self.primitives["is flying"] = False  # flying = False

        if (
            not self.primitives["is game over"] and self.primitives["is flying"] and not self.primitives["is paused"]
        ):  # game_over, flying, paused
            time_now = pygame.time.get_ticks()
            if (
                time_now - self.primitives["last pipe"] > self.primitives["pipe frequency"]
            ):  # last_pipe, pipe_frequency
                pipe_height = random.randint(-100, 100)
                btm_pipe = Pipe(
                    self.screen_width,
                    int(self.screen_height / 2) + pipe_height,
                    -1,
                    self.primitives
                )
                top_pipe = Pipe(
                    self.screen_width,
                    int(self.screen_height / 2) + pipe_height,
                    1,
                    self.primitives
                )
                self.pipe_group.add(btm_pipe)
                self.pipe_group.add(top_pipe)
                self.primitives["last pipe"] = time_now  # last_pipe = time_now

            self.primitives["ground scroll"] -= self.primitives["scroll speed"]  # ground_scroll -= scroll_speed
            if abs(self.primitives["ground scroll"]) > 35:  # ground_scroll
                self.primitives["ground scroll"] = 0  # ground_scroll = 0

            self.pipe_group.update()

        if self.primitives["is game over"]:  # game_over
            self.highscore_check(self.primitives["score"])  # score

            if self.restart_button.draw():
                self.primitives["is game over"] = False  # game_over = False
                self.reset_game()
            if self.exit_button.draw():
                return False

        if self.primitives["is paused"] and not self.primitives["is game over"]:  # paused, game_over
            if self.restart_button.draw():
                self.highscore_check(self.primitives["score"])  # score
                self.reset_game()
                self.primitives["is paused"] = False  # paused = False
            if self.exit_button.draw():
                return False

        return True


class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, primitives):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f"FlappyBird/img/bird{num}.png")
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False
        self.primitives = primitives

    def update(self):
        if not self.primitives["is paused"]:  # paused
            if self.primitives["is flying"]:  # flying
                self.vel += 0.5
                if self.vel > 8:
                    self.vel = 8

            if self.rect.bottom < 768:
                self.rect.y += int(self.vel)

            if not self.primitives["is game over"]:  # game_over
                if pygame.key.get_pressed()[K_SPACE] == 1 and not self.clicked:
                    self.clicked = True
                    self.vel = -10
                if pygame.key.get_pressed()[K_SPACE] == 0:
                    self.clicked = False

                self.counter += 1
                flap_cooldown = 5

                if self.counter > flap_cooldown:
                    self.counter = 0
                    self.index += 1
                    if self.index >= len(self.images):
                        self.index = 0
                self.image = self.images[self.index]

                self.image = pygame.transform.rotate(
                    self.images[self.index], self.vel * -2
                )
            else:
                self.image = pygame.transform.rotate(self.images[self.index], -90)


class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position, primitives):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("FlappyBird/img/pipe.png")
        self.rect = self.image.get_rect()
        self.primitives = primitives
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(self.primitives["pipe gap"] / 2)]  # pipe_gap
        if position == -1:
            self.rect.topleft = [x, y + int(self.primitives["pipe gap"] / 2)]  # pipe_gap
        

    def update(self):
        if not self.primitives["is paused"]:  # paused
            self.rect.x -= self.primitives["scroll speed"]  # scroll_speed
            if self.rect.right < 0:
                self.kill()


class Button:
    def __init__(self, x, y, image, surface):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.surface = surface

    def draw(self):
        action = False
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        self.surface.blit(self.image, (self.rect.x, self.rect.y))
        return action


class Show:
    def __init__(self, x, y, image, surface):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.surface = surface

    def draw(self):
        self.surface.blit(self.image, (self.rect.x, self.rect.y))

