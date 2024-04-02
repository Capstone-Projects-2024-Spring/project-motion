import pygame
import sys
import random

# Initializing Pygame to use library
pygame.init()

# Creation of game window's display
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))

# Main Game's Loop
running = True
while running:
    for event in pygame.event.get(): #iterates through all pygame events
        if event.type == pygame.QUIT: #closing game window event
            running = False


fruits = ['Watermelon', 'Mango', 'Pineapple', 
          'Coconut', 'Strawberry', 'Green Apple', 
          'Red Apple', 'Kiwi', 'Banana', 'Orange']
current_score = 0 #starting score
lives = 3 #total life of gamer

pygame.quit() #terminates pygame
sys.exit() #cleanly stops execution