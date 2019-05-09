from ximea import xiapi
import cam_Buffer.ImageTuple as imgTup
from cam_func.cameraSetup import cameraDev as cam_dev
from collections import deque
import cam_Buffer.serializeBuffer as serBuf
import cv2
import time
import datetime
import numpy as np
import serial
import struct


#Global Variables
font = cv2.FONT_HERSHEY_SIMPLEX
camera_list = []

def initialize_cam_list(num_cameras,cam_set_dict,init_time):
	global camera_list 
	camera_list = cam_dev(num_cameras,cam_set_dict,init_time).cameraList

def get_img_width():
	width = 0
	if camera_list:
		width = camera_list[0].get_width()
	return width

def get_img_height():
	height = 0
	if camera_list:
		height = camera_list[0].get_height()
	return height

def grab_image(num_cameras, img_dict,timeout = 3000):
	ret = 1
	try:
		for i in range(num_cameras):
			camera_list[i].get_image(img_dict[i], timeout = timeout)
	except xiapi.Xi_error as err:
		ret = 0
	return ret

#Starts data acquisition for each camera
def prepare_stream(num_cameras):
	img_dict = {}
	for i in range(num_cameras):
		img_dict[i] = xiapi.Image()
	for i in range(num_cameras):
		camera_list[i].start_acquisition()
		print('Data acquisition for camera %d ready!' %(i))
	return img_dict

def stop_cam_acq(num_cameras, img_dict):
	for i in range(num_cameras):
		#stop data acquisition
		print('Stopping acquisition for camera %d ...' %i)
		camera_list[i].stop_acquisition()
		print("Total Frames for camera %d: %d" % (i,img_dict[i].nframe))
		camera_list[i].set_gpo_mode("XI_GPO_OFF")
		#stop communication
		camera_list[i].close_device()

#computes means/thresholds for requested pixels used for reach detection
def acquireBaseCol(curr_cam, avg_time,row1, row2, col,num_std):

	#initialize vars	
	counter = 0 #counts num images aquired
	sampNum = avg_time*int(curr_cam.get_framerate()) #total images to acquire
	baseMatrix = np.zeros(shape = (row2-row1, sampNum)) #stores acquired pixel values
	img = xiapi.Image() #image object used to acquire images

	#acquire pixel data from desired number of images
	try:
		print('Calculating BaselineCol...')
		while (counter < sampNum):
			curr_cam.get_image(img,timeout = 5000)
			baseMatrix[:,counter] = img.get_image_data_numpy()[row1:row2,col]
			counter +=1
		print('Finished Calculating BaselineCol')
	except KeyboardInterrupt:
		cv2.destroyAllWindows()

	#compute pixel means and thresholds
	meanVector = np.mean(baseMatrix, axis = 1)
	k = np.zeros(shape = (sampNum,1))
	for i in range(0,sampNum):
		diffVec = baseMatrix[:,i] - meanVector
		k[i,:] = np.sum(np.square(diffVec))
	k_hat = np.mean(k)
	k_std = np.std(k)
	threshold = k_hat + num_std*k_std

	#give as option to user where noted in parent script
	return (meanVector, threshold)

def calc_serial(serial_times):
	totalSerial = 0
	if serial_times:
		for ser_time in serial_times:
			totalSerial += ser_time
	return totalSerial

def label_images(cam_id,img,data_arr,intFrame):
	camera_id = 'Camera: ' + str(cam_id)
	frameNum = 'FrameNum : ' + str(img.nframe)
	ts = 'TimeStamp(s) : ' + str(img.tsSec)
	frameRate = 'FPS Setting: ' + str(camera_list[cam_id].get_framerate())
	currTime = "Current Time: " + str(datetime.datetime.now())
	avgFPS = 'Avg FPS(s) : {:5.1f}'.format(intFrame)
	cv2.putText(
		data_arr[cam_id], frameNum, (10,20), font, 0.5, (255, 255, 255), 1
	)
	cv2.putText(
		data_arr[cam_id], ts, (10,40), font, 0.5, (255, 255, 255), 1
		)
	cv2.putText(
		data_arr[cam_id], frameRate, (10,60), font, 0.5, (255, 255, 255), 1
		)
	cv2.putText(
		data_arr[cam_id], currTime, (10,80), font, 0.5, (255, 255, 255), 1
		)
	cv2.putText(
		data_arr[cam_id], avgFPS, (10,100), font, 0.5, (255, 255, 255), 1
		)
	cv2.putText(
		data_arr[cam_id], camera_id, (10,120), font, 0.5, (255, 255, 255), 1
	)

def cam_stream(img_dict,col_dict, stream_dict):
	num_cameras = len(camera_list) #NOTE: consider avoiding use of global variables and just pass the camera list to cam_stream (improves readability)

	buffer_full = False 
	image_buffer = deque() #deque object that is buffer for both cameras. 
	serial_times = deque()  #deque of all serial_times. deque is a high-performance list object w/fast appends/pops on either end

	#unpack stream_dict
	init_time = stream_dict['init_time']
	queue_time = stream_dict['qt']
	show = stream_dict['show']
	path = stream_dict['path']
	camerapath = path + "/camera/data"
	sensorpath = path + "/sensor_data/"
	serial_path = stream_dict['serial_path'] #device name of arduino that we're communicating with via Serial port
	serial_baud = stream_dict['serial_baud'] #Baudrate for arduino. Make sure it is the same on Arduino
	timeout = stream_dict['timeout'] 
	label_images = stream_dict['label_images']
	#get mean and threshold for desired pixels/camera
	#TO-DO: consider arbitrary pixel arrangements and multiple cameras
	meanBaseVector,threshold = acquireBaseCol(camera_list[0], col_dict['avg_time'], col_dict['row1'], col_dict['row2'], col_dict['col'], col_dict['num_std'])

	#some housekeeping timers
	t0 = time.time()
	startTime = t0
	prev_frame = 0
	intFrame = camera_list[0].get_framerate()

	#open sensor data file for writing into
	#sensorfile = sensorpath + str(datetime.datetime.now());    
	#output_file = open(sensorfile, "w+");
	#header = "time countTrials serPNS robotOutState ecuPinState inRewardWin rewardPinState countRewards lickVoltage"
	#output_file.write(header + "\n");

	#create serial communication object for arduino
	#ard_ser = serial.Serial(serial_path, serial_baud, timeout = timeout)
	#time.sleep(2) #wait for arduino to wake up
	#ard_ser.flushInput()
	#ard_ser.flushOutput()
	ser_timeout = False;
	#newline is a list that consists of initial dummy parameters
	# 0 : count_trials = 0
	# 1 : serPNS = '0'
	# 2 : robotOutState = 1
	# 3 : ecuPinState  = 0
	# 4 : inRewardWin = 0
	# 5 : rewardPinState = 1
	# 6 : countRewards = 0
	# 7 : lickVoltage = 0
	newline = '0 0 1 0 0 1 0 0'.split()
	sendArd = '0' #character code for commands to arduino

	print('Starting data acquisition for all cameras')
	while not(ser_timeout):

		try:
			#get images from cameras
			for i in range(num_cameras):
				camera_list[i].get_image(img_dict[i], timeout = timeout) 
			recentTime = time.time()
			#create numpy arr with data from camera. Dimensions of the arr are 
			#determined by imgdataformat
			data_arr = [img_dict[i].get_image_data_numpy() for i in range(num_cameras)]
			#Recalculate Avg FPS every second (Only useful in free run mode not trigger mode)
			# if (time.time() - startTime) > 1:
			# 	intFrame = (img_dict[0].nframe - prev_frame)
			# 	prev_frame = img_dict[0].nframe
			# 	startTime = time.time()

			#show acquired image with frameNum, Timestamp, FPS Setting, current time, and avgFPS since 
			#the beginning of acquisition
			for j in range(num_cameras):
				if label_images:
					label_images(j,img_dict[j],data_arr,intFrame)

				if j == 0 and not(newline[1]=='r') and newline[2] == '0' and newline[3] == '0':
					#reach has not yet been detected, robot is in position, ecu is triggering, must detect reach
					diffVec = data_arr[j][150:501, 300] - meanBaseVector #again, these pixel values shouldn't be hardcoded
					k_real = np.sum(np.square(diffVec))
					if (k_real > threshold):
						sendArd = 'r' #tells arduino reach has been detected

				#imshow is slow so buffer will miss images if show != 0, only shows what we have for first camera 
				if (show == 1):
					cv2.imshow('XiCAM %s' % camera_list[0].get_device_name(), data_arr[0])
					cv2.waitKey(1) #necessary to make the video window the right size 

				if len(image_buffer) == num_cameras*queue_time*int(camera_list[j].get_framerate()):
					#image buffer is full, replace old images in FIFO manner
					if not(buffer_full):
						buffer_full = True 
						print("Buffer is full. Ready for Serialization!")
					removed = image_buffer.popleft()
				#Add new tuple of image frame, date time, and numpy arr of image
				image_buffer.append(imgTup.ImageTuple(j, img_dict[j].nframe, datetime.datetime.now(), data_arr[j]))

			#send/receive stuff to/from arduino
			#ard_ser.write(sendArd) #send arduino most recent character command
			#newline = ard_ser.readline().decode("ascii") #arduino sends a line of data back
			#print(newline)
			#if newline == '':					
				#ser_timeout = True #serial object has timed out, terminate program
			#output_file.write(str(int(round(time.time()*1000))) + " " + newline)
			#newline = newline.split() #parses commands from arduino

			# if newline[3] == '1' and not(sendArd=='s'): 
			# 	#arduino said to serialize image buffer and save to path
			# 	serial_time = serBuf.serialize(image_buffer,camerapath)
			# 	sendArd = 's' #tells arduino image buffer has been saved
			# 	serial_times.append(serial_time)
			# 	#flush out buffers
			# 	image_buffer = deque()
			# 	#ard_ser.reset_input_buffer()
			# 	buffer_full = False
			# 	if (show == 1):
			# 		cv2.destroyAllWindows()
			
		except KeyboardInterrupt:
			#ard_ser.close();
			serial_time = serBuf.serialize(image_buffer,camerapath,newline) #uncomment to save data 
			serial_times.append(serial_time)
			#Flush out the buffer
			image_buffer = deque()
			buffer_full = False
			if (show == 1):
				cv2.destroyAllWindows()
			#exit out of the loop
			ser_timeout = True
		except xiapi.Xi_error as err:
			if err.status == 10:
				print("Triggers not detected.")
			print (err)
			break
	stop_cam_acq(num_cameras,img_dict)
	totalSerial = calc_serial(serial_times)
	print("Total Serialization Time: %s " %str(totalSerial))
	print("Lag between start of program and start of acquisition: %s" % str(t0 - init_time))
	print("Total Acquisition Time: %s " % str(recentTime - t0))
	print("Total Time: %s " % str(recentTime - init_time))
	print("Done.")
