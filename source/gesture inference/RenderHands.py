import pygame


class RenderHands:
    """Given the Mediapipe hands output data, renders the hands in a normilzed view or camera perspective view on a pygame surface"""

    def __init__(self, surface, render_scale=3):
        """Create Render Hand object using a pygame surface and a scaling factor

        Args:
            surface (pygame.surface): pygame surface to render a hand on
            hand_scale (float): multiplier to change the size at which the hand is rendered at
        """
        self.surface = surface
        self.hand_scale = render_scale
        self.font = pygame.font.Font("freesansbold.ttf", 30)
        self.last_velocity = [(0.5, 0.5)]

    def set_render_scale(self, scale:float):
        self.hand_scale = scale

    def connections(self, landmarks, mode):
        """Renders lines between hand joints

        Args:
            landmarks (results): requires the direct output from mediapipe
            mode (bool): render either normalized or perspective
        """

        xy = []

        w, h = self.surface.get_size()

        for index, hand in enumerate(landmarks):
            xy.append([])
            for point in hand:
                if mode:
                    xy[index].append(
                        (
                            (point.x * self.hand_scale + 0.5) * w,
                            (point.y * self.hand_scale + 0.5) * h,
                        )
                    )
                else:
                    xy[index].append(
                        (
                            point.x * w,
                            point.y * h,
                        )
                    )

        for hand in range(len(xy)):
            # thumb
            self.render_line(xy[hand][0], xy[hand][1])
            self.render_line(xy[hand][1], xy[hand][2])
            self.render_line(xy[hand][2], xy[hand][3])
            self.render_line(xy[hand][3], xy[hand][4])
            # index
            self.render_line(xy[hand][0], xy[hand][5])
            self.render_line(xy[hand][5], xy[hand][6])
            self.render_line(xy[hand][6], xy[hand][7])
            self.render_line(xy[hand][7], xy[hand][8])
            # middle
            self.render_line(xy[hand][9], xy[hand][10])
            self.render_line(xy[hand][10], xy[hand][11])
            self.render_line(xy[hand][11], xy[hand][12])
            # ring
            self.render_line(xy[hand][13], xy[hand][14])
            self.render_line(xy[hand][14], xy[hand][15])
            self.render_line(xy[hand][15], xy[hand][16])
            # pinky
            self.render_line(xy[hand][0], xy[hand][17])
            self.render_line(xy[hand][17], xy[hand][18])
            self.render_line(xy[hand][18], xy[hand][19])
            self.render_line(xy[hand][19], xy[hand][20])
            # knuckle
            self.render_line(xy[hand][5], xy[hand][9])
            self.render_line(xy[hand][9], xy[hand][13])
            self.render_line(xy[hand][13], xy[hand][17])

    def render_line(self, start, end):
        """Wrapper function for pygame's render line. Will render a white line with width=5

        Args:
            start (int): line start position
            end (int): line end position
        """
        pygame.draw.line(self.surface, (255, 255, 255), start, end, 5)

    def render_hands(
        self, result, output_image, delay_ms, mode, origins, velocity, pinch
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
        self.surface.fill((0, 0, 0))
        # Render hand landmarks
        # print(delay_ms)
        if pinch != "":
            text = self.font.render(pinch, False, (255, 255, 255))
            self.surface.blit(text, (0, 90))

        w, h = self.surface.get_size()
        if result.handedness != []:
            if mode:
                hand_points = result.hand_world_landmarks
                pygame.draw.circle(self.surface, (255, 0, 255), (0.5 * w, 0.5 * h), 5)
                pygame.draw.line(
                    self.surface,
                    (255, 255, 0),
                    ((velocity[0][0] + 0.5) * w, (velocity[0][1] + 0.5) * h),
                    ((0.5) * w, (0.5) * h),
                    3,
                )
                self.last_velocity = velocity

            else:
                hand_points = result.hand_landmarks

            self.connections(hand_points, mode)
            if hand_points:
                # define colors for different hands

                hand_color = 0
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
                # get every hand detected
                for index, hand in enumerate(hand_points):
                    # each hand has 21 landmarks
                    pygame.draw.circle(
                        self.surface,
                        (255, 0, 255),
                        (origins[index][0] * w, origins[index][1] * h),
                        5,
                    )
                    for landmark in hand:
                        self.__render_hands_pygame(
                            colors[hand_color],
                            landmark.x,
                            landmark.y,
                            delay_ms,
                            mode,
                        )
                    hand_color += 1

    def __render_hands_pygame(self, color, x, y, delay_ms, mode):
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

        pygame.draw.circle(self.surface, color, (x * w, y * h), 5)
        delay_cam = self.font.render(
            str(round(delay_ms[0], 1)) + "ms", False, (255, 255, 255)
        )
        delay_AI = self.font.render(
            str(round(delay_ms[1], 1)) + "ms", False, (255, 255, 255)
        )
        self.surface.blit(delay_cam, (0, 30))
        self.surface.blit(delay_AI, (0, 60))
