# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
# https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
import torch
import torch.nn as nn
import torch.nn.common_types
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from torchsummary import summary
import os
from hand_dataset import HandDataset
from torch.utils.data import WeightedRandomSampler
from LSTM import LSTM
from copy import copy

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


test_data_filename = "test_dataset/minecraft.csv"
train_data_filename = "training_data/minecraft.csv"
graph_title = "Minecraft Finetuning"
input_model_name = "models/finetunedV12.pth"
output_model_name = "output_model/finetunedV13.pth"
batch_size = 100
num_epochs = 5
learning_rate = 0.0001
WEIGHTED_SAMPLE = True
ROTATE_DATA_SET = False
ROTATE_DEGREES = 30
ANIMATE = False


# Device configuration
# https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)
print()

# # Additional Info when using cuda
# if device.type == "cuda":
#     print(torch.cuda.get_device_name(0))
#     print("Memory Usage:")
#     print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
#     print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

lstm = LSTM(input_model_name, force_cpu=True)


# brain surgery
# for param in lstm.parameters():
#     param.requires_grad = False
# for param in lstm.fc_2.parameters():
#     param.requires_grad = True
# for param in lstm.fc_1.parameters():
#     param.requires_grad = True
# with torch.no_grad():
#     lstm.fc_2.bias = torch.nn.Parameter(torch.cat((lstm.fc_2.bias, torch.zeros(2)), 0))
#     lstm.fc_2.weight = torch.nn.Parameter(
#         torch.cat((lstm.fc_2.weight, torch.zeros(2, 128)), 0)
#     )


# Use torchinfo to display the model summary
summary(lstm)
# Hyper-parameters
input_size = lstm.input_size
hidden_size = lstm.hidden_size
sequence_length = lstm.sequence_length
num_layers = lstm.num_layers  # number of stacked lstm layers

num_classes = 0
with open(train_data_filename, "r", newline="", encoding="utf-8") as dataset_file:
    true_labels = next(csv.reader(dataset_file))
    num_classes = len(true_labels)
print(true_labels)

lstm.labels = true_labels
num_classes = num_classes

dataset = HandDataset(
    train_data_filename,
    num_classes,
    sequence_length,
    input_size,
    rotate=ROTATE_DATA_SET,
    degrees=ROTATE_DEGREES,
)
test_dataset = HandDataset(
    test_data_filename,
    num_classes,
    sequence_length,
    input_size,
    rotate=False,
    degrees=ROTATE_DEGREES,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print("dataset built")
[
    print("{:<15}".format(true_labels[index]) + f"count:{count}")
    for index, count in enumerate(dataset.label_counts)
]


if WEIGHTED_SAMPLE:

    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights, num_samples=len(dataset) - 1, replacement=True
    )
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
else:
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

if ANIMATE:
    import animate

    animate.play(train_loader)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# Train the model
lstm.train()
lstm.train(mode=True)
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

        if ROTATE_DATA_SET and i > n_total_steps / 4:
            break

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )

print("saving model")
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
    output_model_name,
)
print(f"model saved as {output_model_name}")

import test_model as test_model

test_model.test(test_loader, device, lstm, true_labels, graph_title=graph_title)
