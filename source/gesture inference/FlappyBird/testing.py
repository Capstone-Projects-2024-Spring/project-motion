# Ensure this import path is correct
from flappybird import FlappyBirdGame  
import unittest
import pygame

class TestFlappyBirdGame(unittest.TestCase):
    def setUp(self):
        # Initialize Pygame and create a game instance
        pygame.init()
        self.game = FlappyBirdGame()

    def test_initial_state(self):
        # Check initial game state conditions
        self.assertEqual(self.game.screen_width, 864)
        self.assertEqual(self.game.screen_height, 936)
        self.assertFalse(self.game.primitives["is started"])
        self.assertFalse(self.game.primitives["is game over"])
        self.assertFalse(self.game.primitives["is flying"])
        self.assertEqual(self.game.primitives["score"], 0)

    def tearDown(self):
        pygame.quit()

if __name__ == "__main__":
    unittest.main()
