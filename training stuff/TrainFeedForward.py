# https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 65
hidden_size = 100
num_epochs = 30
batch_size = 200
learning_rate = 0.001
filename = "training stuff/data.csv"
true_labels = []

num_classes = 0
with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
    true_labels = next(csv.reader(dataset_file))
    num_classes = len(true_labels)
    
print(true_labels)

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

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )


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
        outputs = model(hands)
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

plt.savefig('outputFF.png')
plt.show()
torch.save((model.state_dict(),[input_size, hidden_size, num_classes]), "waveModel.pth")