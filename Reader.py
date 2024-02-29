import csv
from Landmark import Landmark

class Reader:
    """Generates a csv file containing a one hot encoded label of gestures and hand landmark point data"""

    def __init__(self, data_file=None) -> None:
        """Generates a random name for the csv file and initilizes writing

        Args:
            gesture_list (_type_): List of recognized gestures to write to file. The last index in the gesture list shall be a flag for determining if data needs to be written or not
        """
        self.data_file_name = data_file
        self.data_file = data_file
        self.reader = None
        # create file and write gesture list

        self.frame_count = 0
        self.data_file = open(data_file, "r")
        self.reader = csv.reader(self.data_file)

    def read(self):
        self.frame_count+=1
        data = (next(self.reader))
        landmarks = []
        for point in range(21):
            index = point*3
            landmarks.append(Landmark(float(data[index]),float(data[index+1]),float(data[index+2])))
        landmarks.append(float(data[63]))
        landmarks.append(float(data[64]))
        return (landmarks)
    
    def go_back(self, num_go_back):
        self.data_file.close()
        self.frame_count-=num_go_back
        self.data_file = open(self.data_file_name, "r")
        self.reader = csv.reader(self.data_file)
        
        for index in range(self.frame_count):
            data = (next(self.reader))

        landmarks = []
        for point in range(21):
            index = point*3
            landmarks.append(Landmark(float(data[index]),float(data[index+1]),float(data[index+2])))
        landmarks.append(float(data[63]))
        landmarks.append(float(data[64]))
        return (landmarks)


