import csv
from Landmark import Landmark

class Player:

    def __init__(self, data_file=None, gesture_list=[]) -> None:
        self.data_file_name = data_file
        self.data_file = data_file
        self.reader = None
        self.gesture_list = gesture_list
        # create file and write gesture list

        self.frame_count = 0
        self.data_file = open(data_file, "r")
        self.reader = csv.reader(self.data_file)

    def read(self):
        self.frame_count+=1

        try:
            data = (next(self.reader))
            labels = []

            for index, label in enumerate(self.gesture_list):
                labels.append(data[index])
            
            gesture_label = "no gesture"

            for index, label in enumerate(labels):
                if label == "1":
                    gesture_label = self.gesture_list[index]

            landmarks = []
            for point in range(21):
                index = point*3 + 2
                landmarks.append(Landmark(float(data[index]),float(data[index+1]),float(data[index+2])))
            landmarks.append(float(data[63]))
            landmarks.append(float(data[64]))
            return (landmarks, gesture_label)
        except:
            return False
    

