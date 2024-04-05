import pygame


class GestureEventHandler:
    def __init__(self, menu, flags):
        self.menu = menu
        self.flags = flags
        self.is_menu_showing = True

    def handle_events(self, events):
        for event in events:
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
            pygame.display.toggle_fullscreen()
        if event.key == pygame.K_g:
            self.flags["run_model_flag"] = not self.flags["run_model_flag"]

        if type(self.flags["toggle_mouse_key"]) == str:
            if event.key == pygame.key.key_code(self.flags["toggle_mouse_key"]):
                self.flags["mouse"].toggle_mouse()

    def toggle_menu(self):
        self.is_menu_showing = not self.is_menu_showing
        if self.is_menu_showing:
            self.menu.menu.enable()
        else:
            self.menu.menu.disable()

