import os
import h5py 
import cv2
# import ffmpeg

saved_dir = '/media/pns/0e3152c3-1f53-4c52-b611-400556966cd8/data/RM7/05292019/camera/data/2019-05-29 12:44:47.737740/debayered/'
path = '/media/pns/0e3152c3-1f53-4c52-b611-400556966cd8/data/RM7/05292019/camera/data/2019-05-29 12:44:47.737740'
trial_fn = 'trial: 5'

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
	    img = f[key].value
	    debayered = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
	    convertedFn = fnConverter(str(key).split(",")[0],str(key).split(",")[1])
	    # convertedFn = fnConverter(str(key).split(",")[0],str(key).split(",")[1],str(key).split(",")[2])
	    cv2.imwrite('%s/%s.png' % (nestedDir, convertedFn), debayered)
	f.close()

debayerSave(trial_fn)