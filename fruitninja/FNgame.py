import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Pygame data & Creating Game Window
clock = pygame.time.Clock()
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
window.fill((0,0,0))
window_title = pygame.display.set_caption('Fruit Ninja Game -- Motion Capstone Project Mini Game')
backdrop = pygame.image.load('fruitninja/pics/backdrop.png').convert()

# Game Data
fruits = ['Watermelon', 'Mango', 'Pineapple', 
          'Coconut', 'Strawberry', 'Green Apple', 
          'Red Apple', 'Kiwi', 'Banana', 'Orange']
score = 0 
lives = 3 
stats = {} # dictionary for holding game object data

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)

# Fonts & Text
title_font = pygame.font.SysFont(None, 60)
button_font = pygame.font.SysFont(None, 40)
score_font = pygame.font.SysFont(None, 30)
score_text = score_font.render('Score : ' + str(score), True, (255, 255, 255))


def main_menu():
    window.fill(GREEN)

    # Draws the title
    title_text = title_font.render("FRUIT NINJA", True, WHITE)
    title_rect = title_text.get_rect(center=(window_width/2, window_height/4))
    window.blit(title_text, title_rect)

    # Draws the "Play" button
    play_text = button_font.render("Play", True, WHITE)
    play_rect = play_text.get_rect(center=(window_width/2, window_height/2))
    pygame.draw.rect(window, WHITE, play_rect.inflate(10, 5), 1)
    window.blit(play_text, play_rect)

    # Draws the "Quit" button
    quit_text = button_font.render("Quit", True, WHITE)
    quit_rect = quit_text.get_rect(center=(window_width/2, window_height/1.5))
    pygame.draw.rect(window, WHITE, quit_rect.inflate(10, 5), 1)
    window.blit(quit_text, quit_rect)

    # Updates display
    pygame.display.flip()

    return play_rect, quit_rect



running = True #manages game loop
starting_game = True #in true state for beginning of game
game_end = True #terminates while loop when out of lives
in_menu = True #beginning game state
# Main game Loop
while running:

    if starting_game:
        play_rect, quit_rect = main_menu()
        starting_game = False

    for event in pygame.event.get(): #iterates through all pygame events
        if event.type == pygame.QUIT: #closing game window event
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if in_menu:
                # Check if play button clicked
                if play_rect.collidepoint(mouse_pos):
                    in_menu = False
                    # start game here
                    print("starting")
                    window.blit(backdrop, (0,0)) #inserts backdrop into game's window
                    window.blit(score_text, (0,0)) #inserts score into top of window

                # Check if quit button clicked
                elif quit_rect.collidepoint(mouse_pos):
                    pygame.quit()
                    sys.exit()
    


    
    pygame.display.flip()
    clock.tick(30) # loops game at 30 fps


pygame.quit() #terminates pygame
sys.exit() #cleanly stops execution