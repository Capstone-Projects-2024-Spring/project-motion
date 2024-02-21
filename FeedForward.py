import torch.nn as nn
import torch
class NeuralNet(nn.Module):

    def __init__(self, modelName):
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, data = torch.load(modelName, map_location=device)
        print(model)
        super(NeuralNet, self).__init__()
        input_size = data[0]
        hidden_size = data[1]
        num_classes = data[2]
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.load_state_dict(model)
        self.eval()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def get_gesture(self, model_input):
        hands = torch.from_numpy(model_input)
        outputs = self(hands)
        probs = torch.nn.functional.softmax(outputs.data, dim=1)
        confidence, classes = torch.max(probs, 1)
        return confidence.numpy(), classes.numpy()
