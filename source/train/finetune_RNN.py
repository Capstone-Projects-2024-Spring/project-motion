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
from hand_dataset import HandDataset
from torch.utils.data import WeightedRandomSampler
from LSTM import LSTM

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


test_data_filename = "test_dataset/minecraft_motion.csv"
train_data_filename = "training_data/finetuned.csv"
input_model_name = "models/minecraftV2.12.pth"
output_model_name = "output_model/minecraftV2.13.pth"
batch_size = 100
num_epochs = 1
learning_rate = 0.0005
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

# # Additional Info when using cuda
# if device.type == "cuda":
#     print(torch.cuda.get_device_name(0))
#     print("Memory Usage:")
#     print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
#     print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

lstm = LSTM(input_model_name)
lstm.train()
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
dataset = HandDataset(
    train_data_filename,
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

if WEIGHTED_SAMPLE:
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights, num_samples=len(dataset), replacement=True
    )
    # its difficult to random split and do weighted random sample so test and train are the same data
    train_loader = DataLoader(sampler=sampler, dataset=dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
else:

    if len(dataset) % 2 == 0:
        plus1 = 0
    else:
        plus1 = 1
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2) + 1]
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    if test_data_filename:

        test_dataset = HandDataset(
            test_data_filename,
            num_classes,
            sequence_length,
            input_size,
            rotate=False,
            degrees=ROTATE_DEGREES,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )


if ANIMATE:
    import animate

    animate.play(train_loader)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# Train the model
lstm.train()
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

import test_RNN

if WEIGHTED_SAMPLE:
    if test_data_filename:
        test_dataset = HandDataset(
            test_data_filename, num_classes, sequence_length, input_size, rotate=False
        )
        test_loader = test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
    test_RNN.test(test_loader, device, lstm, true_labels)
else:
    test_RNN.test(test_loader, device, lstm, true_labels)
