import unittest
from GetHands import GetHands

class GetHandsTester(unittest.TestCase):
    
    def setUp(self) -> None:
        self.flags = {
            "render_hands_mode": True,
            "gesture_vector": [],
            "number_of_hands": 2,
            "move_mouse_flag": False,
            "run_model_flag": True,
            "gesture_model_path": "models/lstm/brawl.pth",
            "click_sense": 0.05,
            "hands": None,
            "running": True,
            "show_debug_text": True,
            "webcam_mode": 1,
            "toggle_mouse_key": "m",
            "min_confidence": 0.0,
            "gesture_list": [],     
            "mouse_hand_num": 1,
            "keyboard_hand_num": 0,
            "hand_1_keyboard": None,
            "hand_2_keyboard": None,
            "key_toggle_enabled": False,
        }
        self.hands = GetHands(flags=self.flags)
    def test_GetHands_create(self):
        
        self.assertEqual('foo'.upper(), 'FOO')