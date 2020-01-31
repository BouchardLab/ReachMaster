from ximea import xiapi
from collections import deque
from cam_func.cameraSetup import cameraDev as cam_dev
from cam_func.camera_start_acq import *
import cam_Buffer.ImageTuple as imgTup
import cv2
import time
import datetime
import numpy as np

#camera parameters
init_time = time.time() #time since start of program
num_cameras = 2
time_out = 5000 #time (msec) w/no no ECU triggers before program terminates  #TO-DO: adjust this for intertrial intervals when ECU stops triggering
buffer_full = False 
cam_set_dict = {'gpi_selector': "XI_GPI_PORT1", 
				'gpi_mode': "XI_GPI_TRIGGER", 
				'trigger_source': "XI_TRG_EDGE_RISING", 
				'gpo_selector': "XI_GPO_PORT1",
				'gpo_mode': "XI_GPO_EXPOSURE_ACTIVE",
				'imgdf': 'XI_RAW8', #raw camera output format (note: use XI_RAW8 for full fps) 
				'exp_per': 4000, #exposure time (microseconds) (note: determines minimum trigger period) 
				'gain_val': 5.0, #gain: sensitivity of camera 
				'sensor_feat': 1} #set to 1 for faster FPS
queue_time = 50  #Image Buffer queue length (seconds)

#initialize cameras with user defined settings above
initialize_cam_list(num_cameras,cam_set_dict,init_time)

#creates separate image object for EACH camera that can be referred to during acquisition 
#also starts acquisition mode for all cameras
img_dict = prepare_stream(num_cameras)

#dictionary settings for pixels used for reach detection
#image dimensions: 1280 x 1024
#TO-DO: add threshold parameter as well as other hardcoded values noted in the rest of the code
#Also consider merging these two dictionaries into a single object 
col_dict = {'avg_time': 1, #time (sec) we'll record baseline images for
			'row1':150 ,  #row start index for desired pixels
			'row2': 501,  #row stop index 
			'col': 300,	  #column index
			'num_std': 10} #number of standard deviations for threshold calculation 
stream_dict = {'init_time': init_time, 
			   'qt': queue_time, #image buffer duration (seconds)
			   'show': 0, #display images or not (note: setting to 1 does not achieve full fps)
			   'path': '/home/pns/Desktop/project', #path that we save data files into 
			   'serial_path': '/dev/ttyACM0', #device name of Arduino
			   'serial_baud': 115200, #must be same on Arduino
			   'timeout': time_out,
			   'label_images': False} #time (millisec) w/no new data before program terminates

#acquire images triggered/synced by ECU, detect reaches and communicate w/arduino
cam_stream(img_dict,col_dict,stream_dict)