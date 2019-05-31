from threading import Thread 
import cv2
from ximea import xiapi
import reach_master.camera_utilities.ImageTuple as imgTup
import reach_master.camera_utilities.serializeBuffer as serBuf
import time
import datetime
import numpy as np
import serial
from serial.tools import list_ports
import binascii
import struct 
import os 
from collections import deque
 
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
 
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
 
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

class WebcamVideoStream:
    def __init__(self, src=0):
        print('opening camera %s ...' %(0))
        self.cam = xiapi.Camera(dev_id = 0)
        self.cam.open_device()
        self.cam.set_imgdataformat(imgdataformat)
        self.cam.set_exposure(exposure)
        self.cam.set_gain(gain)
        self.cam.set_sensor_feature_value(sensor_feature_value)
        self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
        self.cam.set_gpi_selector(gpi_selector)
        self.cam.set_gpi_mode(gpi_mode)
        self.cam.set_trigger_source(trigger_source)
        self.cam.set_gpo_selector(gpo_selector)
        self.cam.set_gpo_mode(gpo_mode)
        self.cam.set_height(imgHeight)
        self.cam.set_width(imgWidth)
        self.cam.set_offsetX(offsetX)
        self.cam.set_offsetY(offsetY)
        self.cam.enable_recent_frame()
        self.cam.start_acquisition()
        self.img = xiapi.Image()
        self.cam.get_image(self.img,timeout = 2000)
        self.npImg = self.img.get_image_data_numpy() 
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
        self.cam.get_image(self.img,timeout = 2000)
        self.npImg = self.img.get_image_data_numpy()
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

#cam settings
numCams = 3
imgdataformat = "XI_RAW8"                   #raw camera output format (note: use XI_RAW8 for full fps) 
fps = 200
exposure = 3000                             #exposure time (microseconds) (note: determines minimum trigger period) 
gain = 0.0                                  #gain: sensitivity of camera 
sensor_feature_value = 1
gpi_selector = "XI_GPI_PORT1" 
gpi_mode =  "XI_GPI_OFF"#"XI_GPI_TRIGGER"                
trigger_source = "XI_TRG_OFF"       
gpo_selector = "XI_GPO_PORT1"
gpo_mode = "XI_GPO_EXPOSURE_ACTIVE"
imgWidth = 1280 
imgHeight = 1024
offsetX = 0
offsetY = 0
numFrames = 1000

#non-threaded
print("[INFO] sampling frames from webcam...")
camList = []
for i in range(numCams):
    print('opening camera %s ...' %(i))
    cam = xiapi.Camera(dev_id = i)
    cam.open_device()
    cam.set_imgdataformat(imgdataformat)
    cam.set_exposure(exposure)
    cam.set_gain(gain)
    cam.set_sensor_feature_value(sensor_feature_value)
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    cam.set_gpi_selector(gpi_selector)
    cam.set_gpi_mode(gpi_mode)
    cam.set_trigger_source(trigger_source)
    cam.set_gpo_selector(gpo_selector)
    cam.set_gpo_mode(gpo_mode)
    cam.set_height(imgHeight)
    cam.set_width(imgWidth)
    cam.set_offsetX(offsetX)
    cam.set_offsetY(offsetY)
    cam.enable_recent_frame()
    camList.append(cam)
    camList[i].start_acquisition()
    img = xiapi.Image()
    camList[i].get_image(img,timeout = 2000)
    npImg = img.get_image_data_numpy()
fps = FPS().start()
while fps._numFrames < numFrames:
    for i in range(numCams):
        camList[i].get_image(img,timeout = 2000)
        npImg = img.get_image_data_numpy()
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
for i in range(numCams):
    print('Stopping acquisition for camera %d ...' %i)
    camList[i].stop_acquisition()
    camList[i].close_device()

#threaded
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
while fps._numFrames < numFrames:
    vs.cam.get_image(vs.img,timeout = 2000)
    vs.npImg = vs.img.get_image_data_numpy()
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print('Stopping acquisition for camera %d ...' %0)
vs.cam.stop_acquisition()
vs.cam.close_device()
vs.stop()