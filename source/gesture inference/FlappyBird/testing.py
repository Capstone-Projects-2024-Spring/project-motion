import unittest
from unittest.mock import Mock, patch
import pygame

from flappybird import FlappyBirdGame, Bird, Pipe, Button, Show  # Ensure correct import paths


class TestFlappyBirdGame(unittest.TestCase):
    def setUp(self):
        # Initialize the game
        self.game = FlappyBirdGame()

    def test_initialization(self):
        """Test that game initializes with correct default values."""
        self.assertFalse(self.game.primitives["is flying"])
        self.assertFalse(self.game.primitives["is game over"])
        self.assertEqual(self.game.primitives["score"], 0)


    def test_game_over_by_collision(self):
        """Test game over triggered by collision."""
        self.game.flappy.rect.top = -1  # Simulate bird going out of bounds
        self.game.tick()
        self.assertTrue(self.game.primitives["is game over"])

    def test_scoring(self):
        """Test scoring when passing pipes."""
        self.game.primitives["pass pipe"] = True
        self.game.tick()
        self.assertEqual(self.game.primitives["score"], 1)

if __name__ == '__main__':
    unittest.main()
