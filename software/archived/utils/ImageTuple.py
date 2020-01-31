import numpy as np
class ImageTuple():
    #Creates an ImageTuple object that has title of image, frame number of image and actual numpy image array. Used to facilitate debayering/serializing process.
    def __init__(self,cameraNum, time, img):
    # def __init__(self,cameraNum,frameNum, time, img):
        # if not isinstance(cameraNum, long):
        #     raise TypeError("Camera Number must be a number")
        # if not isinstance(img, np.ndarray):
        #     raise TypeError("Image must be a Numpy array")
        # self.title = "Camera: " + str(cameraNum) + ", Frame: " + str(frameNum) + ", Time: " + str(time) 
        self.title = "Camera: " + str(cameraNum) + ", Time: " + str(time) 
        self.cameraNum = cameraNum
        self.img = img

