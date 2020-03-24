import os
import numpy as np
import cv2
from vidgear.gears import WriteGear

path = '/media/pns/0e3152c3-1f53-4c52-b611-400556966cd8/PNS_data/RM15/09272019/S4/videos/'
fnIn = '2019-09-27 16:27:55.484554'
fnOut = path+fnIn+'_deb.mp4'
cap = cv2.VideoCapture(path+fnIn+'.mp4')

output_params = {"-vcodec":"libx264","-crf": 21,"-output_dimensions": (3*1280,1024)}
video = WriteGear(output_filename = fnOut,compression_mode = True,logging=False,**output_params)

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)
    # Display the resulting frame
    # cv2.imshow('Frame',frame)  
    video.write(frame)   
 
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #   break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
video.close()