# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from torchsummary import summary
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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
hidden_size = 100
num_epochs = 20
batch_size = 100
learning_rate = 0.001
sequence_length = 7
filename = "wave.csv"

num_classes = 0
with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
    true_labels = next(csv.reader(dataset_file))
    num_classes = len(true_labels)
print(true_labels)
print("building dataset")


class HandDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
        # self.x_frames = torch.from_numpy(xy[:, num_classes:])
        self.y = np.argmax(xy[:, 0:num_classes], axis=1)


        #procress raw data
        self.x = self.process_data(xy, num_classes, sequence_length)
        self.y = torch.from_numpy(self.y[sequence_length + 1 :])
        
        print("Number of classification labels: " + str(num_classes))
        print("Original number of datapoints: " + str(len(self.x)))
        print("Sequence length: " + str(sequence_length))

        #increase the dataset by adding noise to each sample
        for i in range(0):
            self.x = torch.cat((self.x, self.process_data(xy, num_classes, sequence_length, noise_level=0.0003)), dim=0)
            self.y = torch.cat((self.y, self.y), dim=0)

        print("Shape of x_tensor:", self.x.shape)
        print("Shape of y_tensor:", self.y.shape)

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def process_data(self, xy, num_classes, sequence_length, noise_level=0):
        df = pd.DataFrame(xy, columns=[f"feature_{i}" for i in range(len(true_labels) + input_size)])

        x_frames = df.iloc[:, num_classes:].values

        x = np.zeros(
            (len(x_frames) - sequence_length - 1, sequence_length, input_size),
            dtype=np.float32,
        )

        for i in range(len(x_frames) - sequence_length - 1):
            x[i] = x_frames[i : i + sequence_length]

        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=x.shape)
            x_augmented = x + noise
            x_tensor = torch.tensor(x_augmented, dtype = torch.float32)
            return x_tensor

        x_tensor = torch.tensor(x)
        return x_tensor


dataset = HandDataset()

# data size is 3868
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2) + 1]
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


from mpl_toolkits.mplot3d import Axes3D

examples = iter(test_loader)
example_data, example_targets = next(examples)

import matplotlib.animation as animation

examples = iter(test_loader)
example_data, example_targets = next(examples)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def animate(i):
    ax.clear()
    xlist = []
    ylist = []
    zlist = []
    hand = example_data[0][i]
    for index in range(len(hand) // 3):
        x = hand[index * 3].item()
        y = hand[index * 3 + 1].item()
        z = hand[index * 3 + 2].item()
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        ax.scatter(x, y, z)

    # thumb
    ax.plot3D([xlist[0], xlist[1]], [ylist[0], ylist[1]], [zlist[0], zlist[1]],"gray")
    ax.plot3D([xlist[1], xlist[2]], [ylist[1], ylist[2]], [zlist[1], zlist[2]],"gray")
    ax.plot3D([xlist[2], xlist[3]], [ylist[2], ylist[3]], [zlist[2], zlist[3]],"gray")
    ax.plot3D([xlist[3], xlist[4]], [ylist[3], ylist[4]], [zlist[3], zlist[4]],"gray")
    # # index
    ax.plot3D([xlist[0], xlist[5]], [ylist[0], ylist[5]], [zlist[0], zlist[5]],"gray")
    ax.plot3D([xlist[5], xlist[6]], [ylist[5], ylist[6]], [zlist[5], zlist[6]],"gray")
    ax.plot3D([xlist[6], xlist[7]], [ylist[6], ylist[7]], [zlist[6], zlist[7]],"gray")
    ax.plot3D([xlist[7], xlist[8]], [ylist[7], ylist[8]], [zlist[7], zlist[8]],"gray")
    # # middle
    ax.plot3D([xlist[9], xlist[10]], [ylist[9], ylist[10]], [zlist[9], zlist[10]],"gray")
    ax.plot3D([xlist[10], xlist[11]], [ylist[10], ylist[11]], [zlist[10], zlist[11]],"gray")
    ax.plot3D([xlist[11], xlist[12]], [ylist[11], ylist[12]], [zlist[11], zlist[12]],"gray")
    # # ring
    ax.plot3D([xlist[13], xlist[14]], [ylist[13], ylist[14]], [zlist[13], zlist[14]],"gray")
    ax.plot3D([xlist[14], xlist[15]], [ylist[14], ylist[15]], [zlist[14], zlist[15]],"gray")
    ax.plot3D([xlist[15], xlist[16]], [ylist[15], ylist[16]], [zlist[15], zlist[16]],"gray")
    # # pinky
    ax.plot3D([xlist[0], xlist[17]], [ylist[0], ylist[17]], [zlist[0], zlist[17]],"gray")
    ax.plot3D([xlist[17], xlist[18]], [ylist[17], ylist[18]], [zlist[17], zlist[18]],"gray")
    ax.plot3D([xlist[18], xlist[19]], [ylist[18], ylist[19]], [zlist[18], zlist[19]],"gray")
    ax.plot3D([xlist[19], xlist[20]], [ylist[19], ylist[20]], [zlist[19], zlist[20]],"gray")
    # # knuckle
    ax.plot3D([xlist[5], xlist[9]], [ylist[5], ylist[9]], [zlist[5], zlist[9]],"gray")
    ax.plot3D([xlist[9], xlist[13]], [ylist[9], ylist[13]], [zlist[9], zlist[13]],"gray")
    ax.plot3D([xlist[13], xlist[17]], [ylist[13], ylist[17]], [zlist[13], zlist[17]],"gray")

ani = animation.FuncAnimation(
    fig, animate, frames=len(example_data[0]), interval=100, repeat=True
)
plt.show()

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
        #return out, (hn, cn)
        return out


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


# Test the model
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for hands, labels in test_loader:
        hands = hands.to(device)
        labels = labels.to(device)
        outputs = lstm(hands)
        output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in true_labels],
                     columns = [i for i in true_labels])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)

plt.savefig('outputLSTM.png')
plt.show()
torch.save(
    (
        lstm.state_dict(),
        [input_size, hidden_size, num_classes, sequence_length, num_layers, true_labels],
    ),
    "wave.pth",
)
