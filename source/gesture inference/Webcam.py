# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py

import cv2
class Webcam:
    """
    Class that continuously gets frames and extracts hand data
    with a dedicated thread and Mediapipe
    """
    def __init__(
        self,
        webcam_id=0,
    ):
        self.webcam_id = webcam_id
        self.working_ports, self.menu_selector = self.list_ports()

    def start(self, port):
        # OpenCV setup
        self.stream = cv2.VideoCapture(port)
        # motion JPG format
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        (self.grabbed, self.frame) = self.stream.read()
        self.frame = cv2.flip(self.frame, 1)

    def stop(self):
        self.stream.release()
        cv2.destroyAllWindows()

    def read(self, flip=True):
        (self.grabbed, self.frame) = self.stream.read()
        if flip:
            self.frame = cv2.flip(self.frame, 1)
        return (self.grabbed, self.frame)
    
    #https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
    def list_ports(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        """
        non_working_ports = []
        dev_port = 0
        working_ports = []
        menu_selector = []
        while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
            else:
                is_reading, img = camera.read()
                w = int(camera.get(3))
                h = int(camera.get(4))
                if is_reading:
                    print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                    menu_selector.append((f"{dev_port+1}: {h}x{w}", dev_port))
                    working_ports.append(dev_port)
            dev_port +=1
        #return available_ports,working_ports,non_working_ports
        return working_ports, menu_selector