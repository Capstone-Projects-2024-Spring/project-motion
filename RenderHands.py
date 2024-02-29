import pygame


class RenderHands:
    """Given the Mediapipe hands output data, renders the hands in a normilzed view or camera perspective view on a pygame surface"""

    def __init__(self, surface, hand_scale):
        """Create Render Hand object using a pygame surface and a scaling factor

        Args:
            surface (pygame.surface): pygame surface to render a hand on
            hand_scale (float): multiplier to change the size at which the hand is rendered at
        """
        self.surface = surface
        self.hand_scale = hand_scale
        self.font = pygame.font.Font("freesansbold.ttf", 30)
        self.last_velocity = [(0.5, 0.5)]

    def connections(self, landmarks, mode):
        """Renders lines between hand joints

        Args:
            landmarks (results): requires the direct output from mediapipe
            mode (bool): render either normalized or perspective
        """

        xy = []

        w, h = self.surface.get_size()

        for hand in landmarks:
            for point in hand:
                if mode:
                    xy.append(
                        (
                            (point.x * self.hand_scale + 0.5) * w,
                            (point.y * self.hand_scale + 0.5) * h,
                        )
                    )
                else:
                    xy.append(
                        (
                            point.x * w,
                            point.y * h,
                        )
                    )

        # thumb
        self.render_line(xy[0], xy[1])
        self.render_line(xy[1], xy[2])
        self.render_line(xy[2], xy[3])
        self.render_line(xy[3], xy[4])
        # index
        self.render_line(xy[0], xy[5])
        self.render_line(xy[5], xy[6])
        self.render_line(xy[6], xy[7])
        self.render_line(xy[7], xy[8])
        # middle
        self.render_line(xy[9], xy[10])
        self.render_line(xy[10], xy[11])
        self.render_line(xy[11], xy[12])
        # ring
        self.render_line(xy[13], xy[14])
        self.render_line(xy[14], xy[15])
        self.render_line(xy[15], xy[16])
        # pinky
        self.render_line(xy[0], xy[17])
        self.render_line(xy[17], xy[18])
        self.render_line(xy[18], xy[19])
        self.render_line(xy[19], xy[20])
        # knuckle
        self.render_line(xy[5], xy[9])
        self.render_line(xy[9], xy[13])
        self.render_line(xy[13], xy[17])

    def render_line(self, start, end):
        """Wrapper function for pygame's render line. Will render a white line with width=5

        Args:
            start (int): line start position
            end (int): line end position
        """
        pygame.draw.line(self.surface, (255, 255, 255), start, end, 5)

    def render_hands(
        self, result, output_image, delay_ms, surface, mode, origins, velocity, pinch
    ):
        """ Renders the hands and other associated data from Mediapipe onto a pygame surface.

        Args:
            result (Mediapipe.hands.result): This contains handedness and 21 hand landmark locations in normilized and world coordinates
            output_image (Mediapipe.Image): The image used to detect hands
            delay_ms (float): time taken to process the hands image
            surface (pygame.surface): pygame surface to render the hands on
            mode (bool): wether to render hands in normalized or world coordinates
            origins ((float,float)): an array of tuples containing the origins of one or more hands
            velocity ((float,float)): an array of tuples containing the velocitys of one or more hands
            pinch (str): gesture data
        """
        surface.fill((0, 0, 0))
        # Render hand landmarks
        # print(delay_ms)
        if pinch != "":
            text = self.font.render(pinch, False, (255, 255, 255))
            surface.blit(text, (0, 90))

        w, h = surface.get_size()
        if result.handedness != []:
            if mode[0]:
                hand_points = result.hand_world_landmarks
                pygame.draw.circle(surface, (255, 0, 255), (0.5 * w, 0.5 * h), 5)
                # pygame.draw.circle(
                #     surface,
                #     (255, 255, 0),
                #     ((velocity[0][0] + 0.5) * w, (velocity[0][1] + 0.5) * h),
                #     5,
                # )
                pygame.draw.line(
                    self.surface,
                    (255, 255, 0),
                    ((velocity[0][0] + 0.5) * w, (velocity[0][1] + 0.5) * h),
                    ((0.5) * w, (0.5) * h),
                    3,
                )
                # pygame.draw.line(
                #     self.surface,
                #     (255, 255, 0),
                #     ((velocity[0][0] + 0.5) * w, (velocity[0][1] + 0.5) * h),
                #     (
                #         (self.last_velocity[0][0] + velocity[0][0] + 0.5) * w,
                #         (self.last_velocity[0][1] + velocity[0][1] + 0.5) * h,
                #     ),
                #     3,
                # )
                # pygame.draw.circle(
                #     surface,
                #     (255, 255, 0),
                #     (
                #         (self.last_velocity[0][0] + velocity[0][0] + 0.5) * w,
                #         (self.last_velocity[0][1] + velocity[0][1] + 0.5) * h,
                #     ),
                #     5,
                # )
                self.last_velocity = velocity

            else:
                hand_points = result.hand_landmarks

            self.connections(hand_points, mode[0])
            if hand_points:
                # define colors for different hands

                hand_color = 0
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
                # get every hand detected
                for index, hand in enumerate(hand_points):
                    # each hand has 21 landmarks
                    pygame.draw.circle(
                        surface,
                        (255, 0, 255),
                        (origins[index][0] * w, origins[index][1] * h),
                        5,
                    )
                    for landmark in hand:
                        self.__render_hands_pygame(
                            colors[hand_color],
                            landmark.x,
                            landmark.y,
                            surface,
                            delay_ms,
                            mode[0],
                        )
                    hand_color += 1

    def __render_hands_pygame(self, color, x, y, surface, delay_ms, mode):
        """Renders a single landmark of a hand in pygame and scales the hand.

        Args:
            color (rgb()): color of points in hand
            x (float): x coordinant of a point
            y (float): y coordinant of a point
            surface (pygame.surface): surface to render a hand on
            delay_ms ((float, float)): contains webcam latency and Mediapipe hands model latency
            mode (bool): True to render in normalized mode. False for world coordinates
        """

        w, h = self.surface.get_size()

        if mode:
            x *= self.hand_scale
            y *= self.hand_scale
            x += 0.5
            y += 0.5

        pygame.draw.circle(surface, color, (x * w, y * h), 5)
        delay_cam = self.font.render(
            str(round(delay_ms[0], 1)) + "ms", False, (255, 255, 255)
        )
        delay_AI = self.font.render(
            str(round(delay_ms[1], 1)) + "ms", False, (255, 255, 255)
        )
        surface.blit(delay_cam, (0, 30))
        surface.blit(delay_AI, (0, 60))
