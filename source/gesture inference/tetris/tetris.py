#https://github.com/clear-code-projects/pygame_tetris 

from sys import exit
from os.path import join
from random import choice
from pygame.time import get_ticks
from pygame.image import load
from os import path
import os
import pygame 

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

surface = pygame.Surface((800, 800))

# Variables
COLUMNS = 10
ROWS = 20
CELL_SIZE = 25
GAME_WIDTH, GAME_HEIGHT = COLUMNS * CELL_SIZE, ROWS * CELL_SIZE
#GAME_WIDTH = 800
#GAME_HEIGHT = 800
SIDEBAR_WIDTH = 200
PREVIEW_HEIGHT_FRACTION = 0.7
SCORE_HEIGHT_FRACTION = 1 - PREVIEW_HEIGHT_FRACTION
PADDING = 30
WINDOW_WIDTH = GAME_WIDTH + SIDEBAR_WIDTH + PADDING * 3
WINDOW_HEIGHT = GAME_HEIGHT + PADDING * 2
UPDATE_START_SPEED = 200
MOVE_WAIT_TIME = 200
ROTATE_WAIT_TIME = 200
BLOCK_OFFSET = pygame.Vector2(COLUMNS // 2, -1)

YELLOW = '#f1e60d'
RED = '#e51b20'
BLUE = '#204b9b'
GREEN = '#65b32e'
PURPLE = '#7b217f'
CYAN = '#6cc6d9'
ORANGE = '#f07e13'
GRAY = '#1C1C1C'
BLACK = '#000000'
LINE_COLOR = '#00008B'

SHAPES = {
	'T': {'shape': [(0,0), (-1,0), (1,0), (0,-1)], 'color': PURPLE},
	'O': {'shape': [(0,0), (0,-1), (1,0), (1,-1)], 'color': YELLOW},
	'J': {'shape': [(0,0), (0,-1), (0,1), (-1,1)], 'color': BLUE},
	'L': {'shape': [(0,0), (0,-1), (0,1), (1,1)], 'color': ORANGE},
	'I': {'shape': [(0,0), (0,-1), (0,-2), (0,1)], 'color': CYAN},
	'S': {'shape': [(0,0), (-1,0), (0,-1), (1,-1)], 'color': GREEN},
	'Z': {'shape': [(0,0), (1,0), (0,-1), (-1,-1)], 'color': RED}
}

SCORE_DATA = {1: 40, 2: 100, 3: 300, 4: 1200}

class TetrisGame:
	def __init__(self, get_next_shape, update_score, display_surface):

		self.surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
		self.display_surface = display_surface
		self.rect = self.surface.get_rect(topleft = (PADDING, PADDING))
		self.sprites = pygame.sprite.Group()

		self.get_next_shape = get_next_shape
		self.update_score = update_score

		self.line_surface = self.surface.copy()
		self.line_surface.fill((0,255,0))
		self.line_surface.set_colorkey((0,255,0))
		self.line_surface.set_alpha(120)

		self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
		self.tetromino = Tetromino(
			choice(list(SHAPES.keys())), 
			self.sprites, 
			self.create_new_tetromino,
			self.field_data)

		# timer 
		self.down_speed = UPDATE_START_SPEED
		self.down_speed_faster = self.down_speed * 0.3
		self.down_pressed = False
		self.timers = {
			'vertical move': Timer(self.down_speed, True, self.move_down),
			'horizontal move': Timer(MOVE_WAIT_TIME),
			'rotate': Timer(ROTATE_WAIT_TIME)
		}
		self.timers['vertical move'].activate()

		# score
		self.current_level = 1
		self.current_score = 0
		self.current_lines = 0

		# sound 
		self.landing_sound = pygame.mixer.Sound(join(dname,'sound','landing.wav'))
		self.landing_sound.set_volume(0.01)

	def calculate_score(self, num_lines):
		self.current_lines += num_lines
		self.current_score += SCORE_DATA[num_lines] * self.current_level

		if self.current_lines / 10 > self.current_level:
			self.current_level += 1
			self.down_speed *= 0.75
			self.down_speed_faster = self.down_speed * 0.3
			self.timers['vertical move'].duration = self.down_speed
			
		self.update_score(self.current_lines, self.current_score, self.current_level)

	def check_game_over(self):
		for block in self.tetromino.blocks:
			if block.pos.y < 0:
				exit()

	def create_new_tetromino(self):
		self.landing_sound.play()
		self.check_game_over()
		self.check_finished_rows()
		self.tetromino = Tetromino(
			self.get_next_shape(), 
			self.sprites, 
			self.create_new_tetromino,
			self.field_data)

	def timer_update(self):
		for timer in self.timers.values():
			timer.update()

	def move_down(self):
		self.tetromino.move_down()

	def draw_grid(self):

		for col in range(1, COLUMNS):
			x = col * CELL_SIZE
			pygame.draw.line(self.line_surface, LINE_COLOR, (x,0), (x,self.surface.get_height()), 1)

		for row in range(1, ROWS):
			y = row * CELL_SIZE
			pygame.draw.line(self.line_surface, LINE_COLOR, (0,y), (self.surface.get_width(),y))

		self.surface.blit(self.line_surface, (0,0))

	def input(self):
		keys = pygame.key.get_pressed()

		# LEFT AND RIGHT MOVEMENT
		if not self.timers['horizontal move'].active:
			if keys[pygame.K_LEFT]:
				self.tetromino.move_horizontal(-1)
				self.timers['horizontal move'].activate()
			if keys[pygame.K_RIGHT]:
				self.tetromino.move_horizontal(1)	
				self.timers['horizontal move'].activate()

		# ROTATION
		if not self.timers['rotate'].active:
			if keys[pygame.K_UP]:
				self.tetromino.rotate()
				self.timers['rotate'].activate()

		# DOWN
		if not self.down_pressed and keys[pygame.K_DOWN]:
			self.down_pressed = True
			self.timers['vertical move'].duration = self.down_speed_faster

		if self.down_pressed and not keys[pygame.K_DOWN]:
			self.down_pressed = False
			self.timers['vertical move'].duration = self.down_speed

	def check_finished_rows(self):

		# get the full row indexes 
		delete_rows = []
		for i, row in enumerate(self.field_data):
			if all(row):
				delete_rows.append(i)

		if delete_rows:
			for delete_row in delete_rows:

				# delete full rows
				for block in self.field_data[delete_row]:
					block.kill()

				# move down blocks
				for row in self.field_data:
					for block in row:
						if block and block.pos.y < delete_row:
							block.pos.y += 1

			# rebuild the field data 
			self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
			for block in self.sprites:
				self.field_data[int(block.pos.y)][int(block.pos.x)] = block

			# update score
			self.calculate_score(len(delete_rows))

	def run(self):
		self.input()
		self.timer_update()
		self.sprites.update()

		# TETRIS SCREEN 
		self.surface.fill(BLACK)
		self.sprites.draw(self.surface)

		self.draw_grid()
		self.display_surface.blit(self.surface, (PADDING,PADDING))
		pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)

class Tetromino:
	def __init__(self, shape, group, create_new_tetromino, field_data):

		self.shape = shape
		self.block_positions = SHAPES[shape]['shape']
		self.color = SHAPES[shape]['color']
		self.create_new_tetromino = create_new_tetromino
		self.field_data = field_data

		# create blocks
		self.blocks = [Block(group, pos, self.color) for pos in self.block_positions]

	# collisions
	def next_move_horizontal_collide(self, blocks, amount):
		collision_list = [block.horizontal_collide(int(block.pos.x + amount), self.field_data) for block in self.blocks]
		return True if any(collision_list) else False

	def next_move_vertical_collide(self, blocks, amount):
		collision_list = [block.vertical_collide(int(block.pos.y + amount), self.field_data) for block in self.blocks]
		return True if any(collision_list) else False

	# movement
	def move_horizontal(self, amount):
		if not self.next_move_horizontal_collide(self.blocks, amount):
			for block in self.blocks:
				block.pos.x += amount

	def move_down(self):
		if not self.next_move_vertical_collide(self.blocks, 1):
			for block in self.blocks:
				block.pos.y += 1
		else:
			for block in self.blocks:
				self.field_data[int(block.pos.y)][int(block.pos.x)] = block
			self.create_new_tetromino()

	def rotate(self):
		if self.shape != 'O':

			pivot_pos = self.blocks[0].pos

			new_block_positions = [block.rotate(pivot_pos) for block in self.blocks]

			for pos in new_block_positions:
				# horizontal 
				if pos.x < 0 or pos.x >= COLUMNS:
					return

				# field check -> collision with other pieces
				if self.field_data[int(pos.y)][int(pos.x)]:
					return

				# vertical / floor check
				if pos.y > ROWS:
					return

			for i, block in enumerate(self.blocks):
				block.pos = new_block_positions[i]

class Block(pygame.sprite.Sprite):
	def __init__(self, group, pos, color):
		
		super().__init__(group)
		self.image = pygame.Surface((CELL_SIZE,CELL_SIZE))
		self.image.fill(color)
		
		self.pos = pygame.Vector2(pos) + BLOCK_OFFSET
		self.rect = self.image.get_rect(topleft = self.pos * CELL_SIZE)

	def rotate(self, pivot_pos):

		return pivot_pos + (self.pos - pivot_pos).rotate(90)

	def horizontal_collide(self, x, field_data):
		if not 0 <= x < COLUMNS:
			return True

		if field_data[int(self.pos.y)][x]:
			return True

	def vertical_collide(self, y, field_data):
		if y >= ROWS:
			return True

		if y >= 0 and field_data[y][int(self.pos.x)]:
			return True

	def update(self):

		self.rect.topleft = self.pos * CELL_SIZE


class Score:
	def __init__(self):
		self.surface = pygame.Surface((SIDEBAR_WIDTH,GAME_HEIGHT * SCORE_HEIGHT_FRACTION - PADDING))
		self.rect = self.surface.get_rect(bottomright = (WINDOW_WIDTH - PADDING,WINDOW_HEIGHT - PADDING))
		self.display_surface = pygame.display.get_surface()
		self.font = pygame.font.Font(join(dname,'graphics','Russo_One.ttf'), 30)
		self.increment_height = self.surface.get_height() / 3

		self.score = 0
		self.level = 1
		self.lines = 0

	def display_text(self, pos, text):
		text_surface = self.font.render(f'{text[0]}: {text[1]}', True, 'white')
		text_rext = text_surface.get_rect(center = pos)
		self.surface.blit(text_surface, text_rext)

	def run(self):

		self.surface.fill(BLACK)
		for i, text in enumerate([('Score',self.score), ('Level', self.level), ('Lines', self.lines)]):
			x = self.surface.get_width() / 2
			y = self.increment_height / 2 + i * self.increment_height
			self.display_text((x,y), text)

		self.display_surface.blit(self.surface,self.rect)
		pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)

# The preview for the next shape in the side bar
class Preview:
    def __init__(self):

        self.display_surface = pygame.display.get_surface()
        self.surface = pygame.Surface((SIDEBAR_WIDTH, GAME_HEIGHT * PREVIEW_HEIGHT_FRACTION))
        self.rect = self.surface.get_rect(topright = (WINDOW_WIDTH - PADDING,PADDING))

		# shapes
        self.shape_surfaces = {shape:load(path.join(dname,'graphics',f'{shape}.png')).convert_alpha() for shape in SHAPES.keys()}

		# image position data
        self.increment_height = self.surface.get_height() / 3

    def display_pieces(self, shapes):
        for i, shape in enumerate(shapes):
            shape_surface = self.shape_surfaces[shape]
            scaled_shape_surface = pygame.transform.scale(shape_surface, (int(shape_surface.get_width() * 0.5), int(shape_surface.get_height() * 0.5)))
            x = self.surface.get_width() / 2
            y = self.increment_height / 2 + i * self.increment_height
            rect = scaled_shape_surface.get_rect(center = (x,y))
            self.surface.blit(scaled_shape_surface,rect)
            
    def run(self, next_shapes):
        self.surface.fill(BLACK)
        self.display_pieces(next_shapes)
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)

class Timer:
    
	def __init__(self, duration, repeated = False, func = None):
		self.repeated = repeated
		self.func = func
		self.duration = duration

		self.start_time = 0
		self.active = False

	def activate(self):
		self.active = True
		self.start_time = get_ticks()

	def deactivate(self):
		self.active = False
		self.start_time = 0

	def update(self):
		current_time = get_ticks()
		if current_time - self.start_time >= self.duration and self.active:
			
			if self.func and self.start_time != 0:
				self.func()

			self.deactivate()

			if self.repeated:
				self.activate()
    
class Main:
    def __init__(self):
        global surface
        #pygame.init()
        #self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.display_surface = surface
        self.surface = surface
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('TETRIS')
		

		# shapes
        self.next_shapes = [choice(list(SHAPES.keys())) for shape in range(3)]

        self.game = TetrisGame(self.get_next_shape, self.update_score, self.display_surface)
        self.score = Score()
        self.preview = Preview()

        self.music = pygame.mixer.Sound(join(dname,'sound','03. A-Type Music (Korobeiniki).mp3'))
        self.music.set_volume(0.05)
        self.music.play(-1)

    def update_score(self, lines, score, level):
        self.score.lines = lines
        self.score.score = score
        self.score.level = level
        
    def get_next_shape(self):
        next_shape = self.next_shapes.pop(0)
        self.next_shapes.append(choice(list(SHAPES.keys())))
        return next_shape
    
    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                ...
                # pygame.quit()
                # exit()

    def tick(self):
        # while True:

        # background window
        #self.display_surface.fill(GRAY)
        # Load and scale the background image
        self.background_image = pygame.image.load(os.path.join(dname,'graphics','360_F_269505479_qX1rCsjKJiwYgtXsay8iGk1ssdBiNBx3.jpg'))
        #self.font = pygame.font.Font(join(dname,'graphics','Russo_One.ttf'), 30)
        #self.shape_surfaces = {shape:load(path.join(dname,'graphics',f'{shape}.png')).convert_alpha() for shape in SHAPES.keys()}

        
        background_image = pygame.transform.scale(self.background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        # Drawing the background image on the display game surface
        self.display_surface.blit(background_image, (0, 0))
        
        # components
        self.game.run()
        self.score.run()
        self.preview.run(self.next_shapes)

			# update
            # pygame.display.update()
            # self.clock.tick()

# if __name__ == '__main__':
# 	main = Main()
# 	main.run()