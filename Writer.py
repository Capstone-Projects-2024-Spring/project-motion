import csv
import os.path


class Writer:

    def __init__(self, gesture_list) -> None:
        self.gesture_list = gesture_list
        self.data_file = None
        self.writer = None
        #create file and write gesture list
        if not os.path.isfile("training_data.csv"):
            self.data_file = open(
                "training_data.csv", "a", newline="", encoding="utf-8"
            )
            self.writer = csv.writer(self.data_file)
            self.writer.writerow(self.data_file, self.gesture_list)
        else:
            self.data_file = open(
                "training_data.csv", "a", newline="", encoding="utf-8"
            )
            self.writer = csv.writer(self.data_file)


    def write(self, data, velocity, gesture_index):
        if data != []:
            # add one-hot encoded gesture label
            row = []
            for index, gesture in enumerate(self.gesture_list):
                row.append("0")
            
            hand = 0
            # add 21 landmarks
            for landmark in data[hand]:
                row.append('{:.25f}'.format(landmark.x))
                row.append('{:.25f}'.format(landmark.y))
                row.append('{:.25f}'.format(landmark.z))

            # add 2D velocity vector
            
            row.append('{:.25f}'.format(velocity[hand][0]))
            row.append('{:.25f}'.format(velocity[hand][1]))

            self.writer.writerow(row)
