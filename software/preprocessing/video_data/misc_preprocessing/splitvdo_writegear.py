import cv2
import numpy as np
from vidgear.gears import WriteGear

intput_filename = '928.mp4'
no_of_cam = 3
crf = '25'
pix_Format = 'yuv420p'


def conver2bgr(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
    return frame

def enhanceImage(frame):
	cols, rows, ch = frame.shape
	brightness = np.sum(frame) / (ch * 255 * cols * rows)
	minimum_brightness = 0.2
	alpha = brightness / minimum_brightness
	ratio = brightness / minimum_brightness
	frame = cv2.convertScaleAbs(frame, alpha = 1, beta = 255 * (minimum_brightness - brightness))
	return frame

cap = cv2.VideoCapture(intput_filename)
if (cap.isOpened()== False): 
	print("Error opening video file")

fps = int(cap.get(5))
width = int(cap.get(3)/no_of_cam)
height = int(cap.get(4))

output_params = {'-c:v':'h264', '-crf':crf, '-input_framerate':fps, '-pix_fmt':pix_Format, \
		 '-preset':'fast', '-tune':'zerolatency', '-output_dimensions':(width,height)}

print('Start converting...      ', end='', flush=True)

writers = []
for i in range(no_of_cam):
	output_filename = intput_filename.split('.')[0] + '_cam' + str(i+1) + '.mp4'
	writers.append(WriteGear(output_filename = output_filename, compression_mode = True, \
		logging = False, **output_params))

while(cap.isOpened()): 
	ret, frame = cap.read()
	if ret == True:
		for i, w in enumerate(writers):
			index = range(width * i, width * i + width)
			frame_ = frame[:, index]			
			frame_ = conver2bgr(frame_)
			frame_ = enhanceImage(frame_)
			w.write(frame_)
	else: 
		break

for w in writers:
	w.close()

cap.release()
cv2.destroyAllWindows()

print('[DONE]')



