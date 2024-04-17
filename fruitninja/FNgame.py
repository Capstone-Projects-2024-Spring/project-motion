import pygame
import sys
import random

# Initializing Pygame to use library
pygame.init()

# Creation of game window's display
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
window_title = pygame.display.set_caption('Fruit Ninja Game - Motion Capstone')
backdrop = pygame.image.load('fruitninja/pics/backdrop.png').convert()

# Game Data
fruits = ['Watermelon', 'Mango', 'Pineapple', 
          'Coconut', 'Strawberry', 'Green Apple', 
          'Red Apple', 'Kiwi', 'Banana', 'Orange']
score = 0 
lives = 3 
stats = {} # dictionary for holding data of objects

def main_menu_screen():
    window.blit(backdrop, (0,0))




# Main Game's Loop
running = True #manages game loop
game_end = True #terminates while loop when out of lives
round_one = True #in true state only for beginning of game
while running:
    if round_one:
        main_menu_screen()
        round_one = False

    for event in pygame.event.get(): #iterates through all pygame events
        if event.type == pygame.QUIT: #closing game window event
            running = False
    
    pygame.display.flip()



pygame.quit() #terminates pygame
sys.exit() #cleanly stops execution