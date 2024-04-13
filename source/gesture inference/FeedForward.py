import torch.nn as nn
import torch
import numpy as np
from Console import GestureConsole


class FeedForward(nn.Module):

    def __init__(self, modelName, force_cpu=False):
        # Device configuration
        if force_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, data = torch.load(modelName, map_location=device)

        # model hyperparameters, saved in the model file with its statedict from my train program
        input_size = data[0]
        hidden_size = data[1]
        num_classes = data[2]
        self.labels = data[3]
        self.confidence_vector = []
        self.input_size = input_size
        self.last_origin = [(0, 0)]

        self.console = GestureConsole()
        self.console.print(device)

        # model definition
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.load_state_dict(model)
        self.eval()

    def forward(self, x):
        """Runs a forward pass of the gesture model

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def get_gesture(self, model_input):
        """ One hand input shape should be (1,65)

            Two hand input shape should be (2, 65)
        """
        hands = torch.from_numpy(np.asarray(model_input, dtype="float32"))
        outputs = self(hands)
        probs = torch.nn.functional.softmax(outputs.data, dim=1)

        self.confidence_vector = probs

        # print table
        self.console.table(self.labels, probs.tolist())

        confidence, classes = torch.max(probs, 1)
        return probs.tolist(), classes.numpy().tolist(), confidence.tolist()
    

    def find_velocity_and_location(self, result):
        """Given a Mediapipe result object, calculates the velocity and origin of hands.

        Args:
            result (Mediapipe.hands.result): Direct output object from Mediapipe hands model

        Returns:
            (origins, velocity): A tuple containing an array of tuples representing hand origins, and an array of tuples containing hand velocitys
        """

        normalized_origin_offset = []
        hands_location_on_screen = []
        velocity = []

        for hand in result.hand_world_landmarks:
            # take middle finger knuckle
            normalized_origin_offset.append(hand[9])

        for index, hand in enumerate(result.hand_landmarks):
            originX = hand[9].x - normalized_origin_offset[index].x
            originY = hand[9].y - normalized_origin_offset[index].y
            originZ = hand[9].z - normalized_origin_offset[index].z
            hands_location_on_screen.append((originX, originY, originZ))
            velocityX = self.last_origin[index][0] - hands_location_on_screen[index][0]
            velocityY = self.last_origin[index][1] - hands_location_on_screen[index][1]
            velocity.append((velocityX, velocityY))
            self.last_origin = hands_location_on_screen

        return hands_location_on_screen, velocity

    def gesture_input(self, result, velocity):
        """Converts Mediapipe landmarks and a velocity into a format usable by the gesture recognition model

        Args:
            result (Mediapipe.hands.result): The result object returned by Mediapipe
            velocity ([(float, float)]): An array of tuples containing the velocity of hands

        Returns:
            array: An array of length 65
        """
        model_inputs = []

        for index, hand in enumerate(result.hand_world_landmarks):
            model_inputs.append([])
            for point in hand:
                model_inputs[index].append(point.x)
                model_inputs[index].append(point.y)
                model_inputs[index].append(point.z)
            if velocity != []:
                model_inputs[index].append(velocity[index][0])
                model_inputs[index].append(velocity[index][1])

        return model_inputs
