## Release Version
Features implemented
 - Mouse controls (by Jayming Liu)
 - Pinch to click/select (by Jayming Liu)
 - Multiple hands (by Jayming Liu)
 - Static Gesture recognition (by Jayming Liu)
 - Action Gesture Recognition (by Jayming Liu)
 - NVDIA acceleration and multithreading (by Jayming Liu)
 - Hand Motion Smoothing Interpolation (by Jayming Liu)
 - Simple Games (by Giavonna M Maio, Zaina A Walker-Bey, Arafat Rahaman, Theodore Glenn, Becca Makenova)
 - Complex Game Integration (by Jayming Liu and Arafat Rahaman)

Known bugs
 - possible race condition when using mouse and keyboard emulation
 - improper detection of webcam on some systems
 - labeler program will crash when going out of bonds of the csv file
 - testing will fail if not all classes present in the test data
 - some batch sizes during training will not print training progress
 - confusion matrix class descriptions don't scale correctly with size of matrix

## How to use and build
### To build
 - Windows or Mac
 - Python: version 3.8 - 3.11
 - install all libraries listed in requirements section
 - target "source/gesutre inference/inference.py" including the models folder and each game's folder using pyinstaller or other python to exe program to build an executable

### Overview of usage

The source code contains four folders, "data labeler", "data recorder", "gesture inference" and "train". Each folder contains code that does its namesake. To run the inference program, run the python script at "gesture inference/inference.py". This opens a pygame window and brings up menu options when pressing "esc". Under the settings page, keyboard, mouse, and gesture model options can be configured. 

**Training a new model**<br>
This source code can create two types of models, a static gesture recognition model which uses a single frame as input into a feed forward network, and an action recognition model which takes in a sequence of frames as input into an LSTM network (this can also be used for static gesture regocnition though not as effectively). Typically, to collect training data for a static gesture model, the automatic labeling process can be followed, and action gestures will follow the manual labeling process. 

**Automatic label data collection**<br>
The first step to training a model is to collect data using the python script at "data recorder/recordData.py". The script has two options that must be manually set before running. These are the list "gesture_list" on line 39, and the paramenter "write_labels" on line 49. The gesture list is the set of classes to be classified, and is a list of strings. To automatically record the classification of data during collection, define gesture_list to some list of strings and set write_labels=True. Then when running the script, use the gui and select a gesture to record data for under the dropdown selector. These gestures will be the list that was defined at "gesture_list". Pressing space will create a csv file, then pressing again will either start or stop collection of data when a hand is detected in the webcam's vision. Use this generated csv file for the next training step.

**Manual label data collection**<br>
To manually label data, use the python script at "data recorder/recordData.py", define gesture_list, but this time set the paramenter "write_labels" on line 49 to "False". When running the script, the gesture selector dropdown can be ignored, and data can be recorded to generate a csv file. Then once data collection is finished, rename the generated csv to something convient, and add it to the folder "data labeler". In the python script at "data labeler/labeler.py", change the variable "filename" on line 37 to the name of the generated csv. Assign the variable "gesture_list" on line 40 to a list of strings, in most cases this will be the same list from "recordData.py". Run the script, which will generate another CSV file, this time with the prefix "LABELED" plus the original csv file name. Pressing the keys on the keyboard "1" to "0" and "q" to "p" will select a gesture from "gesture_list". If this is not enough keys for all gestures, the dropdown selector will contain all gestures. Once a gesture is selected, pressing space will label the current frame will the selected gesture, and move onto the next frame. To go back and relabel the previous frame, the left arrow key can be pressed. Pressing the up or down arrows will label multiple frames at a time. To omit data from training, press the right arrow key. To playback the labeled data and its labeled gesture per frame, press "p". When all frames have been processed, use the "LABELED" csv for the next training step, not the original csv file. 

**Training a model**<br>
A new model can be built and trained using either the scripts "TrainFF.py", which will define a feed forward model, or "build_RNN.py" which will define an LSTM model. Define the appropiate hyperparameters for each model type. For an LSTM model, when setting ROTATE_DATA_SET to "True", ensure num_epochs is set to 1 to avoid overfitting, since ROTATE_DATA_SET will duplicate the dataset. Once training has finished, a confusion matrix and .pth will be generated. The .pth file contains the model weights and hyperparameters nessesary for inference, and can simpily be copied into the appropiate folder under "gesture inference/models" to be used for inference.

**Running inference**<br>
Running the script at "gesture inference/inference.py" will initiate running inference of the model path at line 35. The currently used model can be changed under the settings menu page. Pressing F1 cycles through webcam settings. Pressing F2 toggles debug text. Pressing F3 changes how hands are rendered. Pressing "m" will toggle mouse control. The csv file "keybinds.csv" can be modifed to create new key binds for each gesture class. 


## Keywords

Mediapipe feedforward or fully connected network and LSTM or RNN model machine learning artifical intelligence computer vision static and action gesture recogniction and classificaiton system data analysis time sequence multivariant prediction multithreading with pygame projects. Graphical user interface and simulated user input automation. Uses popular AI libraries such as pytorch, numpy, matplotlib, pandas, scikit, sklearn-learn. 


## Conceptual Design

A thread dedicated for gesture recognition captures webcam frames using openCV and runs them through the gesture dection pipeline roughly 30 times a second. This starts with Mediapipe's palm detection model and regression model to extract 21 3-dimensional hand landmarks. This data is used in a another thread which runs roughly 120 times a second to smoothly control a mouse. Another thread for pygame renders the landmarks and runs games 60 times a second. The 21 landmarks plus hand velocity are classified by either a feed forward or LSTM custom pytorch model to obtain an output gesture. Custom keybinds can be configured to each gesture through a csv file, and custom gesture models can be trained or downloaded. 

Training models involves capturing the desired gestures in a csv file, which can be assigned a fixed gesture class during data capture to quickly build a robust dataset, or maunally labeling the data. The csv file can be used during model training for both feed forward and LSTM model types. After training, the model is saved to a .pth file, which contains the model weights and hyperparameters so it can be loaded for inference. 

## Sources

Code snippets taken inspiration from, modified, or copied<br>
https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py<br>
https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoShow.py<br>
https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python<br>
https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130<br>
https://stackoverflow.com/questions/71077706/redirect-print-and-or-logging-to-panel<br>

For more on hand gesture recogniction:<br>
https://paperswithcode.com/task/hand-gesture-recognition<br>
https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.1028907/full

## Requirements
matplotlib==3.8.4 <br/>
mediapipe==0.10.9 <br/>
numpy==1.26.4 <br/>
opencv_python==4.9.0.80 <br/>
pandas==2.2.2 <br/>
PyAutoGUI==0.9.54 <br/>
pydirectinput_rgx==2.1.1 **windows only*<br/>
pygame==2.5.2 <br/>
pygame_menu==4.4.3 <br/>
rich==13.7.1 <br/>
scikit_learn==1.4.2 <br/>
seaborn==0.13.2 <br/>
torch==2.2.0+cu121 <br/>
torch_summary==1.4.5 <br/>
