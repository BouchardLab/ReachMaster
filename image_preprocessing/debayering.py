import os
import h5py 
import cv2

saved_dir = '/home/pns/rat_reaching/scripts/calibration_images/debayered/'
path = '/home/pns/rat_reaching/scripts/calibration_images'
trial_fn = 'trial: 1'

def fnConverter(camids, t):
# def fnConverter(camids,frameNum, t):
    camid = camids.split(" ")[1]
    # frame = frameNum.split(" ")[2]
    time = t.split(" ")[2]
    return "camera: %s, t: %s" % (camid, time)
    # return "camera: %s, frame: %s,  t: %s" % (camid, frame, time)

def debayerSave(filename):
	nestedDir = saved_dir + os.path.splitext(filename)[0]
	if not os.path.isdir(saved_dir):
	    os.makedirs(saved_dir)
	if not os.path.isdir(nestedDir):
	    os.makedirs(nestedDir)
	f = h5py.File(os.path.join(path, filename), 'r')
	for key in f.keys():
	    image = f[key]
	    debayer = cv2.cvtColor(image.value,cv2.COLOR_BAYER_BG2BGR)
	    convertedFn = fnConverter(str(key).split(",")[0],str(key).split(",")[1])
	    # convertedFn = fnConverter(str(key).split(",")[0],str(key).split(",")[1],str(key).split(",")[2])
	    cv2.imwrite('%s/%s.png' % (nestedDir, convertedFn), debayer)
	f.close()

debayerSave(trial_fn)