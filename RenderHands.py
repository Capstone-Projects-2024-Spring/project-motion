import pygame


class RenderHands:
    """Given the Mediapipe hands output data, renders the hands in a normilzed view or camera perspective view"""

    def __init__(self, surface, hand_scale) -> None:
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
        pygame.draw.line(self.surface, (255, 255, 255), start, end, 5)

    def render_hands(
        self, result, output_image, delay_ms, surface, mode, origins, velocity, pinch
    ):
        """Used as function callback by Mediapipe hands model

        This seems backwards, but:
        result.hand_landmarks are the real world coordinates, in reference to the camera fov
        result.hand_world_landmarks normalized to the origin of the hand

        Args:
            result (Hands): list of hands and each hand's 21 landmarks
            output_image (_type_): _description_
            delay_ms ((float, float)): Webcam latency and AI processing latency
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
                        self.render_hands_pygame(
                            colors[hand_color],
                            landmark.x,
                            landmark.y,
                            surface,
                            delay_ms,
                            mode[0],
                        )
                    hand_color += 1

    def render_hands_pygame(self, color, x, y, surface, delay_ms, mode):

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
