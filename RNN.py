import torch.nn as nn
import torch

class LSTM(nn.Module):
    
    def __init__(self, modelName):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, data = torch.load(modelName, map_location=self.device)
        super(LSTM, self).__init__()

        input_size, hidden_size, num_classes, sequence_length, num_layers = data
        self.sequence_length = sequence_length
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        self.load_state_dict(model)
        self.to(self.device)
        self.eval()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next

        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out, (hn, cn)

    def get_gesture(self, model_input):
        hands = torch.from_numpy(model_input).to(self.device)
        outputs, hidden = self(hands)
        probs = torch.nn.functional.softmax(outputs.data, dim=1)
        confidence, classes = torch.max(probs, 1)
        return confidence.cpu().numpy(), classes.cpu().numpy()
