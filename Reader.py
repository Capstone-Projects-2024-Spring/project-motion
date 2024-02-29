import csv
from Landmark import Landmark

class Reader:

    def __init__(self, data_file=None) -> None:
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
    
    def go_to(self, num_frame):
        self.data_file.close()
        self.frame_count=num_frame
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


