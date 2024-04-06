# https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 65
hidden_size = 150
num_epochs = 50
batch_size = 100
learning_rate = 0.001
filename = "minecraft.csv"
labels_list = None

num_classes = 0
with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
    labels_list = next(csv.reader(dataset_file))
    num_classes = len(labels_list)

print("labels: "+str(labels_list))
    

class HandDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(
            filename, delimiter=",", dtype=np.float32, skiprows=1
        )
        self.x = torch.from_numpy(xy[:, num_classes:])
        self.y = torch.from_numpy(np.argmax(xy[:, 0:num_classes], axis=1))
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = HandDataset()

# data size is 3868
train_dataset, test_dataset = (dataset, dataset)

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
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
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
    print(f"Accuracy of the network on {int(dataset.__len__())+1} training dataset: {round(acc,3)} %")

torch.save((model.state_dict(),[input_size, hidden_size, num_classes, labels_list]), "minecraft.pth")