import os
import numpy as np
import cv2

path = '/home/pns/Downloads/'
fnIn1 = 'example2_labeled'

cap1 = cv2.VideoCapture(path+fnIn1+'.mp4')

fps = cap1.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_no = 367
cap1.set(1,frame_no)

ret, frame = cap1.read()

# frame = np.hstack((frame1,frame2,frame3))

cols, rows, ch = frame.shape
brightness = np.sum(frame) / (ch * 255 * cols * rows)
minimum_brightness = 0.2
alpha = brightness / minimum_brightness
ratio = brightness / minimum_brightness
frame = cv2.convertScaleAbs(frame, alpha = 1, beta = 255 * (minimum_brightness - brightness))
cv2.imwrite("C3losegrip2.png",frame)

cv2.imshow('Frame',frame)

# Press Q on keyboard to  exit
while not (cv2.waitKey(25) & 0xFF == ord('q')):
	a = 0

cap1.release()
cv2.destroyAllWindows()
