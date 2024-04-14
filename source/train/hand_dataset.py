# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import math

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# # Hyper-parameters
# input_size = 65
# hidden_size = 50
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.001
# sequence_length = 10
# num_layers = 2  # number of stacked lstm layers
# filename = "data.csv"

# num_classes = 0
# with open(filename, "r", newline="", encoding="utf-8") as dataset_file:
#     true_labels = next(csv.reader(dataset_file))
#     num_classes = len(true_labels)
# print(true_labels)
# print("building dataset")


class HandDataset(Dataset):
    def __init__(self, filename, num_classes, sequence_length, input_size, rotate=True):
        self.num_classes = num_classes
        self.input_size = input_size
        self.xy = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
        self.y = np.argmax(self.xy[:, 0:num_classes], axis=1)
        self.y = torch.from_numpy(self.y[sequence_length + 1 :])
        print("Number of classification labels: " + str(num_classes))
        print("Original number of datapoints: " + str(len(self.y)))
        print("Sequence length: " + str(sequence_length))

        df = pd.DataFrame(
            self.xy, columns=[f"feature_{i}" for i in range(num_classes + input_size)]
        )
        x = df.iloc[:, num_classes:].values
        self.x = self.process(x, num_classes, sequence_length)

        degrees = 15
        step = 15
        # #apply a rotaion in 
        if rotate:
            for x in range(-degrees,degrees+1,step):
                for y in range(-degrees,degrees+1,step):
                    for z in range(-degrees,degrees+1,step):
                        print(f"Applying rotation and noise: ({x},{y},{z})")
                        rotated = self.rotate(self.xy, x, y, z)
                        #add some noise after rotating
                        self.x_rotated = self.process(rotated, num_classes, sequence_length, noise_level=0.0003)
                        self.x = torch.cat((self.x, self.x_rotated), dim=0)
                        self.y = torch.cat(
                                (
                                    self.y,
                                    torch.from_numpy(
                                        np.argmax(self.xy[:, 0:num_classes], axis=1)[sequence_length + 1 :]
                                    ),
                                ),
                                dim=0,
                            )
                    
        
        # Count the number of occurrences of each label
        self.label_counts = np.bincount(self.y.numpy(), minlength=num_classes)
        
                # Calculate weights for each class
        class_count = np.bincount(self.y.numpy(), minlength=len(self.x))
        weights = 1.0 / class_count
        self.sample_weights = weights[self.y.numpy()]
        self.n_samples = len(self.x)

        print(f"Number of samples: {len(self.x)}")

    def rotate(
        self, xy, angle_x_degrees, angle_y_degrees, angle_z_degrees
    ) -> pd.DataFrame:

        self.coords_vel = torch.from_numpy(xy[:, self.num_classes:])
        # self.coords_vel *= 10
        self.coords = self.coords_vel[:, :63]

        xyz = ["x", "y", "z"]
        xyz_df = pd.DataFrame(self.coords, columns=[f"{xyz[i%3]}" for i in range(63)])

        # Create MultiIndex
        arrays = [np.tile(xyz, 21), np.arange(21).repeat(3)]
        multi_index = pd.MultiIndex.from_arrays(arrays, names=("Coordinate", "Index"))

        # Set MultiIndex
        xyz_df.columns = multi_index

        # Add the columns back to rotated_df
        vel = self.coords_vel[:, 63:]

        vel_df = pd.DataFrame(
            vel.numpy(), columns=[f"vel_{i}" for i in range(vel.shape[1])]
        )

        angle_x_radians = np.radians(angle_x_degrees)
        angle_y_radians = np.radians(angle_y_degrees)
        angle_z_radians = np.radians(angle_z_degrees)

        cos_x = np.cos(angle_x_radians)
        sin_x = np.sin(angle_x_radians)

        cos_y = np.cos(angle_y_radians)
        sin_y = np.sin(angle_y_radians)

        cos_z = np.cos(angle_z_radians)
        sin_z = np.sin(angle_z_radians)

        rotation_matrix_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

        rotation_matrix_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

        rotation_matrix_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

        # Reshape the DataFrame to (num_points, num_coords, 3)
        reshaped_df = xyz_df.values.reshape(-1, 3, 3)

        # Rotate each point
        rotated_points = np.matmul(
            np.matmul(np.matmul(reshaped_df, rotation_matrix_x), rotation_matrix_y),
            rotation_matrix_z,
        )

        # Reshape back to original shape
        rotated_df = pd.DataFrame(rotated_points.reshape(xyz_df.shape))
        rotated_df.columns = xyz_df.columns

        rotated_df = pd.concat([rotated_df, vel_df], axis=1)

        return rotated_df

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def process(self, x_frames, num_classes, sequence_length, noise_level=0):

        x = np.zeros(
            (len(x_frames) - sequence_length - 1, sequence_length, self.input_size),
            dtype=np.float32,
        )

        for i in range(len(x_frames) - sequence_length - 1):
            x[i] = x_frames[i : i + sequence_length]

        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=x.shape)
            
            x_augmented = x + noise
            x_tensor = torch.tensor(x_augmented, dtype=torch.float32)
            return x_tensor

        x_tensor = torch.tensor(x)
        return x_tensor

# # input_size = 65
# # hidden_size = 50
# # num_epochs = 10
# batch_size = 100
# # learning_rate = 0.001
# # sequence_length = 10
# # num_layers = 2  # number of stacked lstm layers
# # filename = "data.csv"

#dataset = HandDataset("testdata.csv",2,10,65)
# print(f"dataset.label_counts={dataset.label_counts}")
# print(f"dataset.__len__()={dataset.__len__()}")
# # data size is 3868
# # train_dataset, test_dataset = torch.utils.data.random_split(
# #     dataset, [int(dataset.__len__() * 0.8), int(dataset.__len__() * 0.2) + 1]
# # )
# train_dataset = torch.utils.data.Subset(dataset, range(1))
# test_dataset = torch.utils.data.Subset(dataset, range(1))

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# import matplotlib.animation as animation

# examples = iter(test_loader)
# example_data, example_targets = next(examples)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")


# def animate(i):
#     ax.clear()
#     xlist = []
#     ylist = []
#     zlist = []
#     hand = example_data[0][i]
#     for index in range(len(hand) // 3):
#         x = hand[index * 3].item()
#         y = hand[index * 3 + 1].item()
#         z = hand[index * 3 + 2].item()
#         xlist.append(x)
#         ylist.append(y)
#         zlist.append(z)
#         ax.scatter(x, y, z)

#     # thumb
#     ax.plot3D([xlist[0], xlist[1]], [ylist[0], ylist[1]], [zlist[0], zlist[1]], "gray")
#     ax.plot3D([xlist[1], xlist[2]], [ylist[1], ylist[2]], [zlist[1], zlist[2]], "gray")
#     ax.plot3D([xlist[2], xlist[3]], [ylist[2], ylist[3]], [zlist[2], zlist[3]], "gray")
#     ax.plot3D([xlist[3], xlist[4]], [ylist[3], ylist[4]], [zlist[3], zlist[4]], "gray")
#     # # index
#     ax.plot3D([xlist[0], xlist[5]], [ylist[0], ylist[5]], [zlist[0], zlist[5]], "gray")
#     ax.plot3D([xlist[5], xlist[6]], [ylist[5], ylist[6]], [zlist[5], zlist[6]], "gray")
#     ax.plot3D([xlist[6], xlist[7]], [ylist[6], ylist[7]], [zlist[6], zlist[7]], "gray")
#     ax.plot3D([xlist[7], xlist[8]], [ylist[7], ylist[8]], [zlist[7], zlist[8]], "gray")
#     # # middle
#     ax.plot3D(
#         [xlist[9], xlist[10]], [ylist[9], ylist[10]], [zlist[9], zlist[10]], "gray"
#     )
#     ax.plot3D(
#         [xlist[10], xlist[11]], [ylist[10], ylist[11]], [zlist[10], zlist[11]], "gray"
#     )
#     ax.plot3D(
#         [xlist[11], xlist[12]], [ylist[11], ylist[12]], [zlist[11], zlist[12]], "gray"
#     )
#     # # ring
#     ax.plot3D(
#         [xlist[13], xlist[14]], [ylist[13], ylist[14]], [zlist[13], zlist[14]], "gray"
#     )
#     ax.plot3D(
#         [xlist[14], xlist[15]], [ylist[14], ylist[15]], [zlist[14], zlist[15]], "gray"
#     )
#     ax.plot3D(
#         [xlist[15], xlist[16]], [ylist[15], ylist[16]], [zlist[15], zlist[16]], "gray"
#     )
#     # # pinky
#     ax.plot3D(
#         [xlist[0], xlist[17]], [ylist[0], ylist[17]], [zlist[0], zlist[17]], "gray"
#     )
#     ax.plot3D(
#         [xlist[17], xlist[18]], [ylist[17], ylist[18]], [zlist[17], zlist[18]], "gray"
#     )
#     ax.plot3D(
#         [xlist[18], xlist[19]], [ylist[18], ylist[19]], [zlist[18], zlist[19]], "gray"
#     )
#     ax.plot3D(
#         [xlist[19], xlist[20]], [ylist[19], ylist[20]], [zlist[19], zlist[20]], "gray"
#     )
#     # # knuckle
#     ax.plot3D([xlist[5], xlist[9]], [ylist[5], ylist[9]], [zlist[5], zlist[9]], "gray")
#     ax.plot3D(
#         [xlist[9], xlist[13]], [ylist[9], ylist[13]], [zlist[9], zlist[13]], "gray"
#     )
#     ax.plot3D(
#         [xlist[13], xlist[17]], [ylist[13], ylist[17]], [zlist[13], zlist[17]], "gray"
#     )


# ani = animation.FuncAnimation(
#     fig, animate, frames=len(example_data[0]), interval=100, repeat=True
# )
# plt.show()

# print("dataset built")
