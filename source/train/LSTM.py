
import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, modelName, force_cpu=False):
        # Device configuration
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, data = torch.load(modelName, map_location=self.device)

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
        # self.console = GestureConsole()
        # self.console.print(self.device)

        super(LSTM, self).__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # # hidden state
        # self.h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # .to(self.device)
        # # cell state
        # self.c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # .to(self.device)

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

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Move initial hidden and cell states to device
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

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
