import cv2
from vidgear.gears import WriteGear

intput_filename = '917.mp4'
output_filename = 'converted_917_.mp4'
crf = '25'
pix_Format = 'yuv420p'	

def conver2bgr(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
	return frame

cap = cv2.VideoCapture(intput_filename)
if (cap.isOpened()== False): 
	print("Error opening video stream or file")

fps = int(cap.get(5))
width = int(cap.get(3))
height = int(cap.get(4))

print('Start convering!')

output_params = {'-c:v':'h264', '-crf':crf, '-input_framerate':fps, '-pix_fmt':pix_Format, \
		 '-preset':'fast', '-tune':'zerolatency', '-output_dimensions':(width,height)}
writer = WriteGear(output_filename = output_filename, compression_mode = True, \
	logging = False, **output_params)

while(cap.isOpened()): 
	ret, frame = cap.read()
	if ret == True:
		frame = conver2bgr(frame)
		writer.write(frame)
	else: 
		break

writer.close()
cv2.destroyAllWindows()

print('Finish convering!')



