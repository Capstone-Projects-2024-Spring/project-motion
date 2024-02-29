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

        self.filename = str(round(time.time(), 0)) + ".csv"
        open(self.filename, "x")
        self.data_file = open(self.filename, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.data_file)
        if self.write_labels:
            self.writer.writerow(self.gesture_list)

    def write(self, data, gesture_vector):

        velocity = (data[21], data[22])

        if data != [] and gesture_vector != None:
            # add one-hot encoded gesture label
            row = []

            if self.write_labels:
                for index, gesture in enumerate(gesture_vector):
                    if gesture_vector[index] == "0" or gesture_vector[index] == "1":
                        row.append(gesture_vector[index])
            # add 21 landmarks
            for index, landmark in enumerate(data):
                # size is float32
                row.append("{:.7f}".format(landmark.x))
                row.append("{:.7f}".format(landmark.y))
                row.append("{:.7f}".format(landmark.z))
                if index == 20:
                    break

            # add 2D velocity vector

            row.append("{:.7f}".format(velocity[0]))
            row.append("{:.7f}".format(velocity[1]))
            self.writer.writerow(row)
            self.data_file.flush()

    def remove_rows(self, num_of_rows_to_remove, current_frame):
        self.data_file.close()
        with open(self.filename, "r", newline="", encoding="utf-8") as file:
            self.data_file = file
            data = list(csv.reader(self.data_file))
            del data[0]
            for index in range(num_of_rows_to_remove):
                del data[current_frame - index-1]

        # Write the updated data back to the CSV file
        self.data_file = open(self.filename, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.data_file)
        self.writer.writerow(self.gesture_list)
        self.writer.writerows(data)
