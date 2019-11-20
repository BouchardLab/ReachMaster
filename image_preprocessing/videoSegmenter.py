import os
import numpy as np
import cv2
from vidgear.gears import WriteGear

path = '/media/pns/0e3152c3-1f53-4c52-b611-400556966cd8/PNS_data/RM15/09272019/S1/videos/'
fnIn = '2019-09-27 10:22:24.811332'
fnOut = path+'exampleS1'+'.mp4'
cap = cv2.VideoCapture(path+fnIn+'.mp4')

output_params = {"-vcodec":"libx264","-crf": 0,"-output_dimensions": (688,688)}
video = WriteGear(output_filename = fnOut,compression_mode = True,logging=False,**output_params)

frame1 = 2313*25
frame2 = 2321*25

for f in np.arange(frame1,frame2):
  cap.set(1,f)
  ret, frame = cap.read()
  frame = cv2.cvtColor(frame[:,688:(2*688)], cv2.COLOR_BGR2GRAY)
  frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)
  video.write(frame)

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
video.close()