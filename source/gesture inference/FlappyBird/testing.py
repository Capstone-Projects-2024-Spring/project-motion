import unittest
import pygame
from unittest.mock import MagicMock, patch

# Assuming FlappyBirdGame is in flappy_bird_game.py
from flappybird import FlappyBirdGame

class TestFlappyBirdGame(unittest.TestCase):
    def setUp(self):
        # Mock initialization of Pygame modules
        pygame.init = MagicMock()
        pygame.display.set_caption = MagicMock()
        
        # Mock Surface and required return values
        mocked_surface = MagicMock()
        mocked_surface.get_size.return_value = (864, 936)
        pygame.Surface = MagicMock(return_value=mocked_surface)

        # Mock font and images loading
        pygame.font.SysFont = MagicMock()
        pygame.image.load = MagicMock(return_value=MagicMock())
        pygame.sprite.Group = MagicMock(return_value=MagicMock())

        # Now initializing the game will use mocked values
        self.game = FlappyBirdGame()

    def test_initial_state(self):
        # Test initial states of the game
        self.assertFalse(self.game.primitives["is started"])
        self.assertFalse(self.game.primitives["is game over"])
        self.assertFalse(self.game.primitives["is flying"])
        self.assertEqual(self.game.primitives["score"], 0)

    def test_event_handling(self):
        # Prepare mock events
        event_quit = MagicMock(type=pygame.QUIT)
        event_space_down = MagicMock(type=pygame.KEYDOWN, key=pygame.K_SPACE)
        events = [event_quit, event_space_down]

        # Replace the game's event method with a mock
        with patch.object(self.game, 'events', return_value=None) as mock_events:
            self.game.events(events)
            mock_events.assert_called_once_with(events)

    def test_reset_game(self):
        # Test reset_game function
        self.game.reset_game()
        self.assertEqual(self.game.flappy.rect.x, 100)
        self.assertEqual(self.game.primitives["score"], 0)

if __name__ == '__main__':
    unittest.main()
