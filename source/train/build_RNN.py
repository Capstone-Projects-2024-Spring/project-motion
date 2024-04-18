# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
from torchsummary import summary
import os
from hand_dataset import HandDataset
from torch.utils.data import WeightedRandomSampler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


test_data_filename = None
data_filename = "training_data/tetris.csv"
model_name = "output_model/tetris.pth"
num_epochs = 5
batch_size = 100
learning_rate = 0.001
WEIGHTED_SAMPLE = False
ROTATE_DATA_SET = False
ROTATE_DEGREES = 15
ANIMATE = False

# Device configuration
# https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

# Hyper-parameters
input_size = 65
hidden_size = 50
sequence_length = 7
num_layers = 2  # number of stacked lstm layers

num_classes = 0
with open(data_filename, "r", newline="", encoding="utf-8") as dataset_file:
    true_labels = next(csv.reader(dataset_file))
    num_classes = len(true_labels)
print(true_labels)
print("building dataset")

dataset = HandDataset(
    data_filename,
    num_classes,
    sequence_length,
    input_size,
    rotate=ROTATE_DATA_SET,
    degrees=ROTATE_DEGREES,
)

print("dataset built")

[
    print("{:<15}".format(true_labels[index]) + f"count:{count}")
    for index, count in enumerate(dataset.label_counts)
]
try:
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2)]
    )
except:
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2) + 1]
    )

if WEIGHTED_SAMPLE:
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights, num_samples=len(dataset), replacement=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )
else:
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


if ANIMATE:
    import animate

    animate.play(train_loader)


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
        self.fc_1 = nn.Linear(hidden_size * num_layers, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
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


lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)

# Use torchinfo to display the model summary
summary(lstm)

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
        prediction = lstm(hands_sequence)
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

torch.save(
    (
        lstm.state_dict(),
        [
            input_size,
            hidden_size,
            num_classes,
            sequence_length,
            num_layers,
            true_labels,
        ],
    ),
    model_name,
)
print(f"model saved as {model_name}")

import test_model

test_model.test(test_loader, device, lstm, true_labels)
