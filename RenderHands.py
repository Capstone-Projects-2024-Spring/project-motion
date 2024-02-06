import pygame

class RenderHands():

    def __init__(self, surface, hand_scale) -> None:
        self.surface = surface
        self.hand_scale = hand_scale
        self.font = pygame.font.Font("freesansbold.ttf", 30)

    def connections(self,landmarks):
        xy = []

        w, h = self.surface.get_size()

        for hand in landmarks:
            for point in hand:
                xy.append(((point.x* self.hand_scale + 0.5)*w, (point.y* self.hand_scale+0.5)*h))
                
        #thumb
        self.render_lines(xy[0], xy[1])
        self.render_lines(xy[1], xy[2])
        self.render_lines(xy[2], xy[3])
        self.render_lines(xy[3], xy[4])

        #index
        self.render_lines(xy[0], xy[5])
        self.render_lines(xy[5], xy[6])
        self.render_lines(xy[6], xy[7])
        self.render_lines(xy[7], xy[8])

        #middle
        self.render_lines(xy[9], xy[10])
        self.render_lines(xy[10], xy[11])
        self.render_lines(xy[11], xy[12])

        #ring
        self.render_lines(xy[13], xy[14])
        self.render_lines(xy[14], xy[15])
        self.render_lines(xy[15], xy[16])

        #pinky
        self.render_lines(xy[0], xy[17])
        self.render_lines(xy[17], xy[18])
        self.render_lines(xy[18], xy[19])
        self.render_lines(xy[19], xy[20])

        #knuckle
        self.render_lines(xy[5], xy[9])
        self.render_lines(xy[9], xy[13])
        self.render_lines(xy[13], xy[17])

    def render_lines(self, start, end):
            
        pygame.draw.line(self.surface, (255,255,255), start, end, 5)

    def render_hands(self, result, output_image, delay_ms, surface, mode):
        """Used as function callback by Mediapipe hands model

        This seems backwards, but:
        result.hand_landmarks are the real world coordinates, in reference to the camera fov
        result.hand_world_landmarks normalized to the origin of the hand

        Args:
            result (Hands): list of hands and each hand's 21 landmarks
            output_image (_type_): _description_
            delay_ms ((float, float)): Webcam latency and AI processing latency
        """
        # Render hand landmarks
        # print(delay_ms)

        surface.fill((0, 0, 0))
        if result:
            if mode[0]:
                hand_points = result.hand_world_landmarks
                w, h = surface.get_size()
                pygame.draw.circle(surface, (255,123,123), (0.5* w, 0.5 * h), 5)
                
            else:
                hand_points = result.hand_landmarks

            if hand_points:
                # define colors for different hands

                self.connections(hand_points)

                hand_color = 0
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
                # get every hand detected
                for hand in hand_points:
                    # each hand has 21 landmarks
                    for landmark in hand:
                        self.render_hands_pygame(
                            colors[hand_color], landmark.x, landmark.y, surface, delay_ms, mode[0]
                        )
                    hand_color += 1


    def render_hands_pygame(self,color, x, y, surface, delay_ms, mode):

        w, h = self.surface.get_size()

        if mode:
            x *= self.hand_scale
            y *= self.hand_scale
            x += 0.5
            y += 0.5
            

        pygame.draw.circle(surface, color, (x * w, y * h), 5)
        delay_cam = self.font.render(str(round(delay_ms[0], 1)) + "ms", False, (255, 255, 255))
        delay_AI = self.font.render(str(round(delay_ms[1], 1)) + "ms", False, (255, 255, 255))
        surface.blit(delay_cam, (0, 30))
        surface.blit(delay_AI, (0, 60))

            

