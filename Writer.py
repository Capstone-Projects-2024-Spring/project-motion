import csv
import time


class Writer:

    def __init__(self, gesture_list) -> None:
        self.gesture_list = gesture_list
        self.data_file = None
        self.writer = None
        # create file and write gesture list

        filename = str(round(time.time(),0)) + ".csv"

        self.data_file = open(filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.data_file)
        self.writer.writerow(self.gesture_list)


    def write(self, data, velocity, gesture_vector):

        if data != [] and gesture_vector != None:
            # add one-hot encoded gesture label
            row = []
            for index, gesture in enumerate(gesture_vector):
                if gesture_vector[index] == "0" or gesture_vector[index] == "1":
                    row.append(gesture_vector[index])

            hand = 0
            # add 21 landmarks
            for landmark in data[hand]:
                row.append("{:.25f}".format(landmark.x))
                row.append("{:.25f}".format(landmark.y))
                row.append("{:.25f}".format(landmark.z))

            # add 2D velocity vector

            row.append("{:.25f}".format(velocity[hand][0]))
            row.append("{:.25f}".format(velocity[hand][1]))

            self.writer.writerow(row)
