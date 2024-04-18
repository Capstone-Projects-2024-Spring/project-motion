import unittest
from unittest.mock import Mock
import pygame
from platformer import Player, Platform, tick, events, SpriteSheet, Enemy  # Adjust this import based on your actual file organization

# Mock Pygame's display and other necessary parts to avoid initialization issues
pygame.display.set_mode = Mock()
pygame.font.Font = Mock(return_value=Mock(render=Mock(return_value=pygame.Surface((50, 50)))))

class TestGameComponents(unittest.TestCase):
    def setUp(self):
        # Initialize Pygame environment
        pygame.init()
        self.screen_width = 800
        self.screen_height = 800
        self.player = Player(self.screen_width // 2, self.screen_height - 150)
        self.platform = Platform(self.screen_width // 2 - 50, self.screen_height - 50, 100, False)

    def test_player_initial_position(self):
        """ Test if the player is initialized in the correct position. """
        self.assertEqual(self.player.rect.centerx, self.screen_width // 2)
        self.assertEqual(self.player.rect.centery, self.screen_height - 150)

    def test_player_move_no_input(self):
        """ Test player movement with no keys pressed. """
        initial_x = self.player.rect.x
        initial_y = self.player.rect.y
        self.player.move()  # Assumes move will handle no key press appropriately
        self.assertEqual(self.player.rect.x, initial_x)
        self.assertEqual(self.player.rect.y, initial_y + self.player.vel_y)

    def test_platform_initial_position(self):
        """ Test the initial position of the platform. """
        self.assertEqual(self.platform.rect.x, self.screen_width // 2 - 50)
        self.assertEqual(self.platform.rect.y, self.screen_height - 50)

    def test_game_tick(self):
        """ Test the game tick function, which encapsulates the game loop logic. """
        # This will need a more integrated approach or to be split into smaller testable functions
        pass

class TestEnemy(unittest.TestCase):
    def setUp(self):
        # Initialize Pygame and create a screen
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Setup for SpriteSheet and Enemy
        self.test_image = pygame.Surface((32, 32))
        self.test_image.fill((255, 0, 0))  # A simple red square for visibility
        self.sprite_sheet = SpriteSheet(self.test_image)
        self.enemy = Enemy(self.screen_width, 100, self.sprite_sheet, 1.5)


    def test_update_position(self):
        """Test the update method that moves the enemy."""
        initial_x = self.enemy.rect.x
        self.enemy.update(0, self.screen_width)  # No scroll, simulate movement
        expected_x_change = 2 if self.enemy.direction == 1 else -2
        self.assertEqual(self.enemy.rect.x, initial_x + expected_x_change)

    def test_off_screen_removal(self):
        """Test that enemy is removed (killed) when moving off screen."""
        self.enemy.rect.x = -100  # Force off screen
        self.enemy.update(0, self.screen_width)
        # Check if the sprite is removed from all groups
        self.assertTrue(self.enemy not in self.enemy.groups())


if __name__ == '__main__':
    unittest.main()
    unittest.main(verbosity=2)
