import pygame
import textwrap
from RenderHands import RenderHands


class Renderer:
    def __init__(self, font, window, flags):
        self.font = font
        self.window = window
        self.window_width, self.window_height = window.get_size()
        self.flags = flags
        self.counter = 0
        self.wrapper = textwrap.TextWrapper(width=20)
        self.instructions = "F1 to change webcam place. F2 to hide this text. F3 to change hand render mode. 'M' to toggle mouse control. 'G' to toggle gesture model."
        self.instructions_wrapped = self.wrapper.wrap(text=self.instructions)
        self.delay_AI = None
        self.img_pygame = None
        self.webcam_width = None
        self.webcam_height = None
        self.is_webcam_fullscreen = False
        self.hand_surfaces = []
        self.renderHands = RenderHands(render_scale=3)

        for i in range(4):
            self.hand_surfaces.append(
                pygame.Surface((self.window_width, self.window_height))
            )
            self.hand_surfaces[i].set_colorkey((0, 0, 0))
            
    def render_overlay(self,hands, clock):
        frame = hands.frame.copy()
        self.render_webcam(frame, self.flags["webcam_mode"])
        self.render_hands(hands)
        fps = self.font.render(
            str(round(clock.get_fps(), 1)) + "fps", False, (255, 255, 255)
        )
        self.render_debug_text(self.flags["show_debug_text"], hands, fps)

    def render_webcam(self, frame, webcam_mode):
        self.img_pygame = pygame.image.frombuffer(
            frame.tobytes(), frame.shape[1::-1], "BGR"
        )
        self.webcam_width = self.img_pygame.get_width() * 0.5
        self.webcam_height = self.img_pygame.get_height() * 0.5
        # fullscreen
        if webcam_mode % 3 == 0:
            self.is_webcam_fullscreen = True
            self.img_pygame = pygame.transform.scale(
                self.img_pygame, (self.window_width, self.window_height)
            )
            self.window.blit(self.img_pygame, (0, 0))
        # corner
        elif webcam_mode % 3 == 1:
            self.is_webcam_fullscreen = False
            self.img_pygame = pygame.transform.scale(
                self.img_pygame, (self.webcam_width, self.webcam_height)
            )
            self.window.blit(self.img_pygame, (0, 0))
        # no webcam
        elif webcam_mode % 3 == 2:
            self.is_webcam_fullscreen = False

    def render_hands(self, hands):
        if hands.location != []:
            for index in range(hands.num_hands_detected):
                if self.flags["render_hands_mode"]:
                    landmarks = hands.result.hand_world_landmarks
                else:
                    landmarks = hands.result.hand_landmarks

                for i in range(hands.num_hands_detected):
                    # Transform hand_surfaces to the same size as img_pygame
                    if self.is_webcam_fullscreen == False:
                        self.hand_surfaces[i] = pygame.transform.scale(
                            self.hand_surfaces[i],
                            (self.webcam_width, self.webcam_height),
                        )
                        self.renderHands.thickness = 10
                    else:
                        self.hand_surfaces[i] = pygame.transform.scale(
                            self.hand_surfaces[i],
                            (self.window_width, self.window_height),
                        )
                        self.renderHands.thickness = 25
                    self.renderHands.render_hands(
                        landmarks[i],
                        self.flags["render_hands_mode"],
                        hands.location,
                        hands.velocity,
                        self.hand_surfaces[i],
                        i,
                    )

        else:
            for i in range(4):
                self.hand_surfaces[i].fill((0, 0, 0)) 

        corners = [
            (0, 0),
            (self.window_width - self.webcam_width, 0),
            (0, self.window_height - self.webcam_height),
            (
                self.window_width - self.webcam_width,
                self.window_height - self.webcam_height,
            ),
        ]
        if self.is_webcam_fullscreen == False:
            for i in range(hands.num_hands_detected):
                self.window.blit(self.hand_surfaces[i], corners[i])
        else:
            # fullscreen just render hands on top of each other
            for i in range(hands.num_hands_detected):
                self.window.blit(self.hand_surfaces[i], corners[0])

    def render_debug_text(self, show_debug_text, hands, fps):
        self.counter += 1
        if show_debug_text:
            for index in range(len(hands.gestures)):
                gesture_text = self.font.render(
                    hands.gestures[index], False, (255, 255, 255)
                )
                self.window.blit(gesture_text, (0, index * 40 + 80))

            if hands.delay != 0 and self.counter % 60 == 0:
                self.delay_AI = self.font.render(
                    "Webcam: " + str(round(1000 / hands.delay, 1)) + "fps",
                    False,
                    (255, 255, 255),
                )
            if self.delay_AI is not None:
                self.window.blit(self.delay_AI, (0, 40))
            self.window.blit(fps, (0, 0))

            for index, instruc in enumerate(self.instructions_wrapped):
                instructions_text = self.font.render(instruc, False, (255, 255, 255))
                self.window.blit(instructions_text, (0, 400 + index * 40))
