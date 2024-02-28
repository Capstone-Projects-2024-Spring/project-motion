import pygame


class RenderHands:
    """Given the Mediapipe hands output data, renders the hands in a normilzed view or camera perspective view"""

    def __init__(self, surface, hand_scale) -> None:
        self.surface = surface
        self.hand_scale = float(hand_scale)
        self.font = pygame.font.Font("freesansbold.ttf", 30)

    def connections(self, landmarks):

        xy = []

        w, h = self.surface.get_size()

        for index,point in enumerate(landmarks):
            xy.append(
                (
                    (float(point.x) * float(self.hand_scale) + 0.5) * w,
                    (float(point.y) * float(self.hand_scale) + 0.5) * h,
                )
            )
            if index == 20:
                break


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
        self, result
    ):
        
        self.surface.fill((0, 0, 0))

        w, h = self.surface.get_size()

        hand_points = result

        pygame.draw.circle(self.surface, (255, 0, 255), (0.5 * w, 0.5 * h), 5)
        # pygame.draw.line(
        #     self.surface,
        #     (255, 255, 0),
        #     ((velocity[0] + 0.5) * w, (velocity[1] + 0.5) * h),
        #     ((0.5) * w, (0.5) * h),
        #     3,
        # )

        self.connections(hand_points)
        if hand_points:
            # define colors for different hands

            hand_color = 0
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
            # get every hand detected
            for index,landmark in enumerate(hand_points):
                self.render_hands_pygame(
                    colors[hand_color],
                    float(landmark.x),
                    float(landmark.y),
                )
                if index == 20:
                    break

    def render_hands_pygame(self, color, x, y):
        w, h = self.surface.get_size()
        x *= self.hand_scale
        y *= self.hand_scale
        x += 0.5
        y += 0.5
        pygame.draw.circle(self.surface, color, (x * w, y * h), 5)

