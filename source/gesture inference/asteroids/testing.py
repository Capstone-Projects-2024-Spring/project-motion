import unittest
import pygame
import asteroids

class TestGame(unittest.TestCase):

    def setUp(self):
        pygame.init()
        self.mock_surface = pygame.Surface((800, 600))

        # Initialize game state with actual Pygame objects
        self.game = asteroids
        self.game.run = True
        self.game.lives = 3
        self.game.score = 0
        self.game.asteroids = []
        self.game.player_bullets = []
        self.game.stars = []
        self.game.aliens = []
        self.game.alien_bullets = []
        self.game.player = asteroids.Player()
        self.game.surface = self.mock_surface  # Assuming your game uses a 'surface' attribute

    def tearDown(self):
        pygame.quit()


    def test_asteroid_initial_position(self):
        asteroid = asteroids.Asteroid(1)
        print(f"Testing x: 0 <= {asteroid.x} <= {asteroids.SCREEN_WIDTH}")
        print(f"Testing y: {-1 * asteroid.h - 5} <= {asteroid.y} <= {asteroids.SCREEN_HEIGHT + 5}")
        self.assertTrue(0 <= asteroid.x <= asteroids.SCREEN_WIDTH)
        self.assertTrue(-1 * asteroid.h - 5 <= asteroid.y <= asteroids.SCREEN_HEIGHT + 5)

    def test_bullet_movement(self):
        """Test bullet moves correctly."""
        bullet = asteroids.Bullet()
        old_x, old_y = bullet.x, bullet.y
        bullet.move()
        self.assertNotEqual((old_x, old_y), (bullet.x, bullet.y))

    def test_asteroid_initial_position(self):
        """Test asteroid spawns within expected range."""
        asteroid = asteroids.Asteroid(1)
        self.assertTrue(0 <= asteroid.x <= asteroids.SCREEN_WIDTH)
        self.assertTrue(-1 * asteroid.h - 5 <= asteroid.y <= asteroids.SCREEN_HEIGHT + 5)

    def test_game_over(self):
        """Test game over conditions."""
        self.game.lives = 0
        self.game.tick()
        self.assertTrue(self.game.game_over)

    def test_rapid_fire_mode(self):
        """Test rapid fire mode toggles correctly."""
        self.game.rapid_fire_start = self.game.count
        self.game.tick()  # Assuming this tick would push it over the 500 tick threshold
        self.assertFalse(self.game.rapid_fire)

    