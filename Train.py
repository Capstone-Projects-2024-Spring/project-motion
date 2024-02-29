# https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv

# Device configuration

"""
The purpose of device configuration is to determine the device on which the PyTorch tensors and operations should be executed.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters

"""
input_size: Size of the input features for the neural network.

hidden_size: Number of neurons in the hidden layer of the neural network.

num_epochs: Number of times the entire dataset is passed forward and backward through the neural network during training.

batch_size: Number of samples used in each iteration of training.

learning_rate: A factor that determines the size of the steps taken during optimization.

filename: Path to the CSV file containing the dataset.

"""

input_size = 65
hidden_size = 100
num_epochs = 20
batch_size = 15
learning_rate = 0.001
filename = "wave_data.csv"

num_classes = 0
with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
    labels = next(csv.reader(dataset_file))
    num_classes = len(labels)
    

"""
Custom Dataset Class for Hand Classification Problem.

This class, 'HandDataset', is designed to handle the loading and processing of
data for a hand classification problem. It inherits from the PyTorch Dataset class.

Parameters:
    - filename (str): Path to the CSV file containing the dataset.
    - num_classes (int): Number of classes in the classification problem.

Attributes:
    - x (torch.Tensor): Input features (torch tensor format).
    - y (torch.Tensor): Target labels (torch tensor format).
    - n_samples (int): Number of samples in the dataset.

Methods:
    - __init__(self): Initializes the dataset by loading data from the CSV file, setting up input and output tensors.
    - __getitem__(self, index): Returns a specific sample (input, label) at the given index.
    - __len__(self): Returns the total number of samples in the dataset.
"""

class HandDataset(Dataset):
    def __init__(self):

        """
        Initializes the dataset by loading data from the CSV file and setting up input and output tensors.

        Pre-conditions:
            - 'filename' is a valid path to the CSV file containing the dataset.
            - 'num_classes' is the correct number of classes in the classification problem.

        Post-conditions:
            - 'self.x' and 'self.y' are initialized with input features and target labels.
            - 'self.n_samples' is set to the total number of samples in the dataset.
        """
        
        xy = np.loadtxt(
            filename, delimiter=",", dtype=np.float32, skiprows=1
        )
        self.x = torch.from_numpy(xy[:, num_classes:])
        self.y = torch.from_numpy(np.argmax(xy[:, 0:num_classes], axis=1))
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):

        """
        Returns a specific sample (input, label) at the given index.

        Parameters:
            - index (int): Index of the desired sample.

        Return:
            - Tuple of torch tensors representing input and label.

        Pre-conditions:
            - 'index' is a valid index within the range of the dataset.

        Post-conditions:
            - Returns the input and label tensors for the specified index.
        """
        
        return self.x[index], self.y[index]

    def __len__(self):

        """
        Returns the total number of samples in the dataset.

        Return:
            - int: Total number of samples in the dataset.

        Pre-conditions:
            - The dataset has been initialized.

        Post-conditions:
            - Returns the total number of samples in the dataset.
        """
        
        return self.n_samples


dataset = HandDataset()

# data size is 3868
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(dataset.__len__()*0.8), int(dataset.__len__()*0.2)+1])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# examples = iter(test_loader)
# example_data, example_targets = next(examples)

# for i in range(4):
#     plt.subplot(2, 3, i + 1)
#     for index, point in enumerate(example_data[i]):
#         plt.scatter(
#             example_data[i][index * 3 + 0].numpy(),
#             example_data[i][index * 3 + 1].numpy(),
#         )
#         if index == 20:
#             plt.plot(example_data[i][63], example_data[i][64])
#             break
# plt.show()


# Fully connected neural network with one hidden layer

"""
Neural Network Model Class for Hand Classification.

This class, 'NeuralNet', defines the architecture of a neural network for the hand classification problem.

Parameters:
    - input_size (int): Size of the input features.
    - hidden_size (int): Number of neurons in the hidden layer.
    - num_classes (int): Number of output classes.

Attributes:
    - input_size (int): Size of the input features.
    - l1 (nn.Linear): First linear layer mapping input to hidden layer.
    - relu (nn.ReLU): Rectified Linear Unit activation function.
    - l2 (nn.Linear): Second linear layer mapping hidden layer to output.
"""

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        """
        Initializes the neural network with specified input size, hidden size, and number of classes.

        Parameters:
            - input_size (int): Size of the input features.
            - hidden_size (int): Number of neurons in the hidden layer.
            - num_classes (int): Number of output classes.

        Pre-conditions:
            - 'input_size', 'hidden_size', and 'num_classes' are valid integers.

        Post-conditions:
            - Neural network architecture is defined with input, hidden, and output layers.
        """
        
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        """
        Performs forward pass through the neural network.

        Parameters:
            - x (torch.Tensor): Input features.

        Return:
            - torch.Tensor: Output of the neural network.

        Pre-conditions:
            - 'x' is a valid torch tensor representing input features.

        Post-conditions:
            - Returns the output of the neural network after the forward pass.
        """
        
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (hands, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        hands = hands.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(hands)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
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
        outputs = model(hands)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network on {int(dataset.__len__()*0.2)+1} test samples: {round(acc,3)} %")

torch.save((model.state_dict(),[input_size, hidden_size, num_classes]), "waveModel.pth")
