import pygame


class GestureEventHandler:
    def __init__(self, menu, flags):
        self.menu = menu
        self.flags = flags
        self.is_menu_showing = True
        self.is_fullscreen = False

    def handle_events(self, events):
        for event in events:
            self.handle_event(event)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.flags["hands"].stop()
            self.flags["hands"].join()
            self.flags["running"] = False

        elif event.type == pygame.KEYDOWN:
            self.handle_keydown(event)

    def handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            self.toggle_menu()
        if event.key == pygame.K_F1:
            self.flags["webcam_mode"] += 1
        if event.key == pygame.K_F2:
            self.flags["show_debug_text"] = not self.flags["show_debug_text"]
        if event.key == pygame.K_F3:
            self.flags["render_hands_mode"] = not self.flags["render_hands_mode"]
        if event.key == pygame.K_F11:
            self.toggle_fullscreen()
        if event.key == pygame.K_g:
            self.flags["run_model_flag"] = not self.flags["run_model_flag"]
        
        if type(self.flags["toggle_mouse_key"]) == str:
            if event.key == pygame.key.key_code(self.flags["toggle_mouse_key"]):
                self.menu.toggle_mouse()

    def keyboard_mouse(self):
        if self.flags["run_model_flag"] and len(self.flags["hands"].confidence_vectors) > 0:
            # send only the first hand confidence vector the gesture model output
            self.flags["keyboard"].gesture_input(self.flags["hands"].confidence_vectors[0])
        else:
            self.flags["keyboard"].release()

        if self.flags["move_mouse_flag"] and self.flags["hands"].location != []:
            self.handle_mouse_control()


    def toggle_menu(self):
        self.is_menu_showing = not self.is_menu_showing
        if self.is_menu_showing:
            self.menu.menu.enable()
        else:
            self.menu.menu.disable()

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        # pygame.display.toggle_fullscreen()

    def handle_mouse_control(self):
        mouse_button_text = ""
        hand = self.flags["hands"].result.hand_world_landmarks[0]
        if self.flags["mouse"].is_clicking(hand[8], hand[4], self.flags["click_sense"]):
            mouse_button_text = "left"
        location = self.flags["hands"].location[0]
        self.flags["mouse"].control(location[0], location[1], mouse_button_text)
