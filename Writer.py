import csv
import time


class Writer:
    """Generates a csv file containing a one hot encoded label of gestures and hand landmark point data"""

    def __init__(self, gesture_list, write_labels=False) -> None:
        """Generates a random name for the csv file and initilizes writing

        Args:
            gesture_list (_type_): List of recognized gestures to write to file. The last index in the gesture list shall be a flag for determining if data needs to be written or not
        """
        self.gesture_list = gesture_list
        self.data_file = None
        self.writer = None
        self.write_labels = write_labels
        # create file and write gesture list

        filename = str(round(time.time(), 0)) + ".csv"

        self.data_file = open(filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.data_file)
        if self.write_labels:
            self.writer.writerow(self.gesture_list)

    def write(self, data, velocity, gesture_vector):
        """Writes the labels, hand data, and velocity to file

        Args:
            data (_type_): Normizlized hand data
            velocity (_type_): velocity of a hand
            gesture_vector (_type_): one hot encoded label to be written to file
        """

        if data != [] and gesture_vector != None:
            # add one-hot encoded gesture label
            row = []

            if self.write_labels:
                for index, gesture in enumerate(gesture_vector):
                    if gesture_vector[index] == "0" or gesture_vector[index] == "1":
                        row.append(gesture_vector[index])

            hand = 0
            # add 21 landmarks
            for landmark in data[hand]:
                # size is float32
                row.append("{:.7f}".format(landmark.x))
                row.append("{:.7f}".format(landmark.y))
                row.append("{:.7f}".format(landmark.z))

            # add 2D velocity vector

            row.append("{:.7f}".format(velocity[hand][0]))
            row.append("{:.7f}".format(velocity[hand][1]))

            self.writer.writerow(row)
