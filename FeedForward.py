import torch.nn as nn
import torch

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.load_state_dict(torch.load("gestureModel.pth"))
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
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
