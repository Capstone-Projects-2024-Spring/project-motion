import pygame
import random
import sys
import os

# Setting current working directory to the directory this script is to handle relative paths correctly
abspath = os.path.abspath(__file__)  # Get absolute path of current script
dname = os.path.dirname(abspath)  # Extract directory part of absolute path
os.chdir(dname)  # Change current working directory to the script directory

# Color constants
green = (50, 200, 50)
black = (0, 0, 0)
white = (255, 255, 255)

class GameFN:
    """Main game class for Fruit Ninja style game."""

    # Constructor
    def __init__(self):
        """Initializes the game, creates window, and loads resources."""
        pygame.init()
        self.window_width = 800
        self.window_height = 600
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Fruit Ninja")
        self.backdrop = pygame.image.load("pics/backdrop.png").convert()
        self.window.fill((0, 0, 0))
        self.fps = 18
        self.timer = pygame.time.Clock()

        self.score = 0
        self.strikes = 0
        self.stats = {} # 2D dictionary of throwables and correlated stats
        self.throwables = ["bomb", "orange", "strawberry", "watermelon", "redapple", "greenapple", "coconut", "banana"] # 1D list of throwables
        self.init_objects()
        self.mouse_trails = []  # Stores positions of the mouse for drawing the trail

        self.title_font = pygame.font.SysFont(None, 60)
        self.button_font = pygame.font.SysFont(None, 40)
        self.score_font = pygame.font.SysFont(None, 30)
        self.countdown_font = pygame.font.SysFont(None, 100)
        
        self.running = True
        self.in_menu = True
        self.game_start = True
        self.game_end = True
        self.paused = False
    
    def init_objects(self):
        """Propogates the stats dictionary once"""
        for obj in self.throwables:
            self.make_throwables(obj)
    
    def main_menu(self):
        """Displays the main menu with play and quit options."""
        self.window.fill(green)

        # Draws the title
        title_text = self.title_font.render("FRUIT NINJA", True, white)
        title_rect = title_text.get_rect(center=(self.window_width/2, self.window_height/4))
        self.window.blit(title_text, title_rect)

        # Draws Play button
        play_text = self.button_font.render("Play", True, white)
        play_rect = play_text.get_rect(center=(self.window_width/2, self.window_height/2))
        pygame.draw.rect(self.window, white, play_rect.inflate(10, 5), 1)
        self.window.blit(play_text, play_rect)

        # Draws Quit button
        quit_text = self.button_font.render("Quit", True, white)
        quit_rect = quit_text.get_rect(center=(self.window_width/2, self.window_height/1.5))
        pygame.draw.rect(self.window, white, quit_rect.inflate(10, 5), 1)
        self.window.blit(quit_text, quit_rect)

        # Draws the pause instruction
        pause_instruction_text = self.score_font.render("Press 'P' to pause the game", True, white)
        pause_instruction_rect = pause_instruction_text.get_rect(center=(self.window_width/2, self.window_height * 0.85))
        self.window.blit(pause_instruction_text, pause_instruction_rect)

        # Updates display
        pygame.display.flip()
        return play_rect, quit_rect

    def restart_menu(self):
        """Displays the game over screen and waits for the player to restart or quit."""
        self.window.blit(self.backdrop, (0, 0)) # clears the screen
        self.render_words("Game Over", 80, self.window_width/2-10, self.window_height/4, black)
        self.render_words(f"Score: {self.score}", 100, self.window_width/2-3, self.window_height/2-7, green)
        self.render_words("Press any key to restart", 70, self.window_width/2, self.window_height*3/4, white)
        pygame.display.flip()

        choosing = True
        while choosing: #loops until player chooses an event
            for event in pygame.event.get():
                if event.type == pygame.KEYUP: #restart event
                    choosing = False
                if event.type == pygame.QUIT: #quit event
                    pygame.quit()
    
    def display_pause_screen(self):
        """Displays the pause screen with options to resume or quit."""
        # Display the pause message
        pause_text = self.title_font.render("Paused", True, white)
        pause_rect = pause_text.get_rect(center=(self.window_width / 2, self.window_height / 3 + 70))
        self.window.blit(pause_text, pause_rect)

        # Display a quit option
        quit_text = self.button_font.render("Quit Game", True, white)
        quit_rect = quit_text.get_rect(center=(self.window_width / 2, self.window_height / 1.5))
        pygame.draw.rect(self.window, white, quit_rect.inflate(20, 10), 2)
        self.window.blit(quit_text, quit_rect)

        pygame.display.update()  # Update the display to show the pause screen

        return quit_rect

    def display_countdown(self):
        """Displays a countdown from 3 to 1 on the screen with a white border around the numbers."""
        background_snapshot = self.window.copy()  # Take snapshot of current game screen
        offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Offsets for the border effect

        for number in range(3, 0, -1):
            # Restore the snapshot each time to clear the previous number
            self.window.blit(background_snapshot, (0, 0))

            # Render countdown number with border
            text = str(number)
            font_color = (100, 100, 100)  # Dark gray
            border_color = white 
            countdown_text = self.countdown_font.render(text, True, border_color)
            countdown_rect = countdown_text.get_rect(center=(self.window_width / 2, self.window_height / 3 + 20))

            # Draw border
            for offset in offsets:
                border_position = (countdown_rect.x + offset[0], countdown_rect.y + offset[1])
                self.window.blit(countdown_text, border_position)

            # Draw main text over the border
            countdown_text = self.countdown_font.render(text, True, font_color)
            self.window.blit(countdown_text, countdown_rect)

            pygame.display.update()
            pygame.time.wait(800)  # Pause interval timing


    def render_words(self, input, size, x, y, color):
        """Helper function to render text on the screen."""
        font = pygame.font.SysFont(None, size)  # Font object with default system font
        surface = font.render(input, True, color)  # Render the text onto new surface with antialiasing
        rect = surface.get_rect()  # Get rectangular area of surface which text is drawn onto
        rect.midtop = (x, y)  # Set middle-top coordinate of text rectangle for positioning
        self.window.blit(surface, rect)  # Draws text surface onto window at specified text rect position

    def render_strikes(self, x, y, strikes, picture):
        """Displays the number of strikes (lives lost) on the screen."""
        for i in range(strikes):
            pic = pygame.image.load(picture)
            pic_rect = pic.get_rect()
            pic_rect.x = int(x + 35 * i)
            pic_rect.y = y
            self.window.blit(pic, pic_rect)

    def make_throwables(self, throwable):
        """Creates and randomizes components of each throwable object."""
        item_path = "pics/" + throwable + ".png"
        # Set a default color, can customize per fruit type
        aura_color = self.get_aura_color(throwable)
        # for key:value pairs in the stats dictonary (2D), the keys are throwables
        # and the values are the dictionary of key:value pairs below
        self.stats[throwable] = {
            "x_pos": random.randint(50, self.window_width - 50),
            "y_pos": self.window_height + 5,
            "dx/dt": random.randint(-15, 15),
            "dy/dt": random.randint(-68, -50),
            "time": 0,
            "pic": pygame.image.load(item_path),
            "throwing": False,
            "struck": False,
            "angle": 0,
            "angular_velocity": random.randint(-20, 20),
            "aura": {  # Properties here
                "active": False,
                "radius": 0,
                "color": aura_color,  # Use the dynamic color
                "lifespan": 10  # Before fading away
            }
        }
        if random.random() < 0.6:
            self.stats[throwable]["throwing"] = False
        else:
            self.stats[throwable]["throwing"] = True
    
    def get_aura_color(self, throwable):
        """Returns an RGBA color based on the type of throwable."""
        colors = {
            "orange": (255, 165, 0, 128),
            "strawberry": (255, 0, 100, 128),
            "watermelon": (0, 255, 100, 128),
            "redapple": (255, 0, 0, 128),
            "greenapple": (0, 255, 0, 128),
            "coconut": (255, 255, 255, 128),
            "banana": (255, 255, 0, 128),
            "bomb": (128, 128, 128, 128)  # Gray for bombs
        }
        return colors.get(throwable, (255, 255, 255, 128))

    def shake_screen(self, duration=100, intensity=5):
        """Action occurs when bombs are hit"""
        shake_time = pygame.time.get_ticks() + duration
        while pygame.time.get_ticks() < shake_time:
            random_offset = (random.randint(-intensity, intensity), random.randint(-intensity, intensity))
            for key, value in self.stats.items():
                self.window.blit(value["pic"], (value["x_pos"] + random_offset[0], value["y_pos"] + random_offset[1]))
            pygame.display.update()
            self.window.fill(black)  # Clear after each shake


    def make_physics(self):
        """Handles the physics and interactions of all throwables."""
        for key, value in self.stats.items():
            if not value["throwing"]:
                self.make_throwables(key)
            else:
                # Movement in different directions
                value["x_pos"] += value["dx/dt"]
                value["y_pos"] += value["dy/dt"]
                value["dy/dt"] += (value["time"])
                value["time"] += 1
                value["angle"] += value["angular_velocity"]

                # Rotatation before being struck
                if not value["struck"]:
                    value["pic"] = pygame.transform.rotate(pygame.image.load("pics/" + key + ".png"), value["angle"])
                # Rotatation after being struck
                elif value["struck"] and key != "bomb":
                    value["angle"] += random.randint(1,100)
                    value["pic"] = pygame.transform.rotate(pygame.image.load("pics/cut_" + key + ".png"), value["angle"])

                # Check if throwable is within screen bounds
                if value["y_pos"] > self.window_width:
                    self.make_throwables(key)
                else:
                    self.window.blit(value["pic"], (value["x_pos"], value["y_pos"]))

                # Collision detection with mouse click
                cursor_pos = pygame.mouse.get_pos()

                # Defined bounds of the object for clear collision detection
                x_min = value["x_pos"]
                x_max = value["x_pos"] + 80
                y_min = value["y_pos"]
                y_max = value["y_pos"] + 80

                # Check if the cursor is within the bounds of the object
                cursor_within_x_bounds = x_min < cursor_pos[0] < x_max
                cursor_within_y_bounds = y_min < cursor_pos[1] < y_max

                # Perform the check using simplified conditions
                if not value["struck"] and cursor_within_x_bounds and cursor_within_y_bounds:
                    # Activate the aura effect
                    new_color = self.get_aura_color(key)  # Get new color based on throwable type
                    value['aura']['color'] = new_color  # Update the color dynamically
                    value['aura']['active'] = True
                    value['aura']['radius'] = 0  # Starting radius

                    if key != "bomb": # when fruits are struck
                        self.score += 1
                        cut_path = "pics/cut_" + key + ".png"
                    else: # when bombs are struck
                        self.shake_screen()
                        self.strikes += 1
                        if self.strikes >= 3:
                            self.game_end = True
                            self.restart_menu()
                        cut_path = "pics/explode.png"

                    value["pic"] = pygame.image.load(cut_path)
                    value["dx/dt"] += self.fly_left_or_right(random.random())
                    value["struck"] = True

    def fly_left_or_right(self, halfchance):
        if(halfchance >= 0.5):
            return 20 # Return 20 to move it right
        else:
            return -20 # Return -20 to move it left
    
    def update_and_draw_auras(self):
        for key, value in self.stats.items():
            aura = value["aura"]
            if aura["active"]:
                # Increase the radius for expansion effect
                aura["radius"] += 7
                aura["lifespan"] -= 1
                
                # Only draw if there's remaining lifespan
                if aura["lifespan"] > 0:
                    # Create a temporary surface with per-pixel alpha
                    temp_surface = pygame.Surface((aura["radius"] * 2, aura["radius"] * 2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surface, aura["color"], (aura["radius"], aura["radius"]), aura["radius"])
                    
                    # Blit the temp surface to the window at the correct position
                    self.window.blit(temp_surface, (value["x_pos"], value["y_pos"]))
                else:
                    aura["active"] = False  # Disable aura when lifespan ends


    
    def draw_mouse_trail(self):
        """Updates and draws the mouse trail based on cursor movement."""
        self.mouse_trails.append(pygame.mouse.get_pos())  # Append current mouse position
        if len(self.mouse_trails) > 3:  # Keep only last 3 positions for a smooth trail
            self.mouse_trails.pop(0)

        if len(self.mouse_trails) > 1:
            pygame.draw.lines(self.window, (220,220,220), False, self.mouse_trails, 4)

    def run(self):
        """Main game loop."""
        quit_rect = None  # To keep track of the quit button rect
        while self.running:
            self.timer.tick(self.fps) #loops game at specified fps

            if self.game_start: #starting state
                play_rect, quit_rect = self.main_menu()
                self.game_start = False

            if self.game_end: #restart state
                self.score = 0
                self.strikes = 0
                self.game_end = False

            for event in pygame.event.get(): #iterates through all pygame events
                if event.type == pygame.QUIT: #closing game window event
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.in_menu:
                        # Check if play button clicked
                        if play_rect.collidepoint(mouse_pos):
                            self.display_countdown()
                            self.in_menu = False
                            
                        # Check if quit button clicked
                        elif quit_rect.collidepoint(mouse_pos):
                            pygame.quit()
                            sys.exit()
                    elif self.paused:
                        if quit_rect:
                            pygame.quit()
                            sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        if self.paused:  # If game is already paused, prepare to unpause
                            self.display_countdown()  # Show countdown before unpausing
                            self.paused = False  # Unpause after countdown
                        else:  # If the game is not paused, pause it
                            self.paused = True
                            
            if self.paused:
                self.display_pause_screen()
                continue  # Skip the rest of the loop

            if not self.in_menu:
                self.window.blit(self.backdrop, (0, 0))
                self.render_words(f"Score: {self.score}", 40, 80, 0, green)
                self.render_strikes(self.window_width - 130, 4, 3, "pics/emptystrike.png")
                self.render_strikes(self.window_width - 130, 4, self.strikes, "pics/redstrike.png")
                self.make_physics()
                self.draw_mouse_trail()
                self.update_and_draw_auras()

            pygame.display.update()

if __name__ == '__main__':
    game = GameFN()
    game.run()

pygame.quit() #terminates pygame
sys.exit() #cleanly stops execution