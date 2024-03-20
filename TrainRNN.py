#https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from torchsummary import summary

# Device configuration
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Hyper-parameters
input_size = 65
hidden_size = 100
num_epochs = 10
batch_size = 100
learning_rate = 0.001
sequence_length = 10
filename = "data.csv"

num_classes = 0
with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
    labels = next(csv.reader(dataset_file))
    num_classes = len(labels)

print("building dataset")
class HandDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
        # self.x_frames = torch.from_numpy(xy[:, num_classes:])
        self.y = np.argmax(xy[:, 0:num_classes], axis=1)
        print("Number of classification labels: " + str(num_classes))

        self.y = torch.from_numpy(self.y[sequence_length + 1 :])

        self.x = self.process_data(xy, num_classes, sequence_length)

        print("Shape of x_tensor:", self.x.shape)
        print("Shape of y_tensor:", self.y.shape)

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def process_data(self, xy, num_classes, sequence_length):
        df = pd.DataFrame(xy, columns=[f"feature_{i}" for i in range(80)])

        x_frames = df.iloc[:, num_classes:].values

        x = np.zeros(
            (len(x_frames) - sequence_length - 1, sequence_length, input_size),
            dtype=np.float32,
        )

        print(x.shape)

        for i in range(len(x_frames) - sequence_length - 1):
            x[i] = x_frames[i : i + sequence_length]

        x_tensor = torch.tensor(x)

        return x_tensor


dataset = HandDataset()

# data size is 3868
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2) + 1]
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print("dataset built")


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out, (hn, cn)


import warnings

warnings.filterwarnings("ignore")

num_layers = 1  # number of stacked lstm layers

lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
example = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
example.to(device)
# Use torchinfo to display the model summary
summary(example)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (hands, labels) in enumerate(train_loader):
        # [batch_size, sequence_length, 65]
        hands_sequence = hands.to(device)
        labels = labels.to(device)

        # Forward pass
        prediction, hidden = lstm(hands_sequence)

        # calculate the loss
        loss = criterion(prediction, labels)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for hands, labels in test_loader:
        hands = hands.to(device)
        labels = labels.to(device)
        outputs, hidden = lstm(hands)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(
        f"Accuracy of the network on {int(dataset.__len__()*0.2)+1} test samples: {round(acc,3)} %"
    )

torch.save(
    (lstm.state_dict(), [input_size, hidden_size, num_classes, sequence_length, num_layers]),
    "gestureModel.pth",
)
