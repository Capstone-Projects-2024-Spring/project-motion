import unittest
from unittest.mock import MagicMock, patch
import pygame

# Assuming the game's code is in a module named tetris
from tetris import TetrisGame, Tetromino, Block, Timer

class TestTetrisGame(unittest.TestCase):
    def setUp(self):
        # Mock pygame's font and mixer to prevent errors on environments without proper display
        pygame.init()
        pygame.display.set_mode = MagicMock()
        pygame.font.Font = MagicMock()
        pygame.mixer.Sound = MagicMock()
        
        # Prepare the game components for testing
        self.display_surface = pygame.display.set_mode((800, 600))
        self.get_next_shape = MagicMock(return_value='T')
        self.update_score = MagicMock()

        self.game = TetrisGame(self.get_next_shape, self.update_score, self.display_surface)

    def test_score_calculation(self):
        # Test score calculation for clearing lines
        self.game.calculate_score(1)
        self.update_score.assert_called_with(1, 40, 1)  # Expecting score for 1 line at level 1
        self.assertEqual(self.game.current_score, 40)

    def test_create_new_tetromino(self):
        # Test the creation of a new tetromino
        old_tetromino = self.game.tetromino
        self.game.create_new_tetromino()
        self.assertNotEqual(old_tetromino, self.game.tetromino)
    
    @patch('tetris.restart', side_effect=SystemExit)
    def test_game_over(self, mock_restart):
        # Position a block in a game-over state
        self.game.tetromino.blocks[0].pos.y = -1
        # Expect SystemExit to be raised when checking for game over
        with self.assertRaises(SystemExit):
            self.game.check_game_over()
        # Assert that restart was called, leading to the SystemExit
        mock_restart.assert_called_once()

    def test_move_down(self):
        initial_y = [block.pos.y for block in self.game.tetromino.blocks]
        self.game.move_down()  # Perform the move down action
        expected_y = [y + 1 for y in initial_y]  # Expect each block to move down by 1
        actual_y = [block.pos.y for block in self.game.tetromino.blocks]
        self.assertEqual(expected_y, actual_y, "Blocks did not move down as expected")

    def test_rotate_tetromino(self):
        # Test tetromino rotation
        initial_positions = [block.pos for block in self.game.tetromino.blocks]
        self.game.tetromino.rotate()
        new_positions = [block.pos for block in self.game.tetromino.blocks]
        self.assertNotEqual(initial_positions, new_positions)

if __name__ == '__main__':
    unittest.main()
