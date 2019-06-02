import cv2

vidcap = cv2.VideoCapture('/home/pns/rat_reaching/scripts/camera/data/2019-06-02 13:12:05.756840/trial: 1.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  # print('Read a new frame: ', success)
  count += 1
print(count)