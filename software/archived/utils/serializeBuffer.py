import time
import datetime
import h5py
import os 

#Function for serializing the imageBuffer
def serialize(imageBuffer,path,newline):
	#SERIALIZE WITH HDF5
	#Create Unique Trial names
	trial_fn = 'trial: '+ str(newline[0])#+', '+newline[1] #str(datetime.datetime.now())#time.strftime("%Y%m%d-%H%M%S")
	#Checks to see if path directory exists
	if not os.path.isdir(path):
		os.makedirs(path)
	hdfStart = time.time()
	h = h5py.File(os.path.join(path, trial_fn), 'w', libver='latest')
	for i in imageBuffer:
		h.create_dataset(i.title, data=i.img)
	hdfEnd = time.time()
	serialTime = hdfEnd - hdfStart
	# print("Time it took to serialize hdf5: %f ms" % ((serialTime)*1000))
	# print("Image Buffer Length: %d" % len(imageBuffer))
	# print("Image Buffer contains frames from: %d to %d" %(imageBuffer[0].frameNum,imageBuffer[len(imageBuffer) -1].frameNum))
	return serialTime
