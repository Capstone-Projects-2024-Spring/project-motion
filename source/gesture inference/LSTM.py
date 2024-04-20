import torch.nn as nn
from torch import load, device, cuda, zeros, from_numpy, max
from numpy import asarray

# from Console import GestureConsole


class LSTM(nn.Module):

    def __init__(self, modelName, force_cpu=False):
        # Device configuration
        if force_cpu:
            self.device = device("cpu")
        else:
            self.device = device("cuda" if cuda.is_available() else "cpu")
        model, data = load(modelName, map_location=self.device)

        # model hyperparameters, saved in the model file with its statedict from my train program
        # [input_size, hidden_size, num_classes, sequence_length, num_layers, true_labels]
        input_size = data[0]
        hidden_size = data[1]
        num_classes = data[2]
        self.sequence_length = data[3]
        num_layers = data[4]
        self.labels = data[5]
        self.confidence_vector = []
        self.input_size = input_size
        self.last_origin = [(0, 0)]



        super(LSTM, self).__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # hidden state
        self.h_0 = zeros(self.num_layers, 1, self.hidden_size)  # .to(self.device)
        # cell state
        self.c_0 = zeros(self.num_layers, 1, self.hidden_size)  # .to(self.device)

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # lstm
        self.fc_1 = nn.Linear(hidden_size * num_layers, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()
        self.to(self.device)
        self.load_state_dict(model)
        self.eval()

    def forward(self, x):

        # Move initial hidden and cell states to device
        h_0 = self.h_0.to(self.device)
        c_0 = self.c_0.to(self.device)

        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # Flatten the hidden state of all layers
        hn = hn.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        # return out, (hn, cn)
        return out

    def get_gesture(self, model_input):
        """One hand input shape should be (1,65)

        Two hand input shape should be (2, 65)
        """
        # newest data appends to the end of the list
        if len(model_input) < self.sequence_length:
            # print(f"input too short len(model_input): {len(model_input)}")
            return None
        elif len(model_input) > self.sequence_length:
            # print(f"input too long len(model_input): {len(model_input)}")
            return None

        hands = from_numpy(asarray([model_input], dtype="float32"))

        outputs = self(hands.to(self.device))
        probs = nn.functional.softmax(outputs.data, dim=1)

        self.confidence_vector = probs

        confidence, classes = max(probs, 1)
        return probs.tolist(), classes, confidence.tolist()

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
