import pygame

class GestureEventHandler:
    def __init__(self, hands, menu, mouse, keyboard, flags):
        self.hands = hands
        self.menu = menu
        self.mouse = mouse
        self.keyboard = keyboard
        self.flags = flags
        self.is_menu_showing = True
        self.webcam_mode = 1
        self.show_debug_text = True
        self.is_fullscreen = False

    def handle_events(self, events, window, renderer, clock, font):
        for event in events:
            self.handle_event(event, renderer, clock, font)

    def handle_event(self, event, renderer, clock, font):
        if event.type == pygame.QUIT:
            self.hands.stop()
            self.hands.join()
            self.flags["running"] = False

        elif event.type == pygame.KEYDOWN:
            self.handle_keydown(event)


    def handle_keydown(self, event):
        if event.key == pygame.K_m:
            self.menu.toggle_mouse()
        elif event.key == pygame.K_ESCAPE:
            self.toggle_menu()
        elif event.key == pygame.K_F1:
            self.webcam_mode += 1
        elif event.key == pygame.K_F2:
            self.show_debug_text = not self.show_debug_text
        elif event.key == pygame.K_F3:
            self.flags["render_hands_mode"] = not self.flags["render_hands_mode"]
        elif event.key == pygame.K_F11:
            self.toggle_fullscreen()
        elif event.key == pygame.K_m:
            self.keyboard.press("m")
        elif event.key == pygame.K_g:
            self.flags["run_model_flag"] = not self.flags["run_model_flag"]

    def toggle_menu(self):
        self.is_menu_showing = not self.is_menu_showing
        if self.is_menu_showing:
            self.menu.menu.enable()
        else:
            self.menu.menu.disable()

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        #pygame.display.toggle_fullscreen()

    def handle_mouse_control(self):
        mouse_button_text = ""
        hand = self.hands.result.hand_world_landmarks[0]
        if self.mouse.is_clicking(hand[8], hand[4], self.flags["click_sense"]):
            mouse_button_text = "left"
        location = self.hands.location[0]
        self.mouse.control(location[0], location[1], mouse_button_text)