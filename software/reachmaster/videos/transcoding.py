import subprocess as sp
import time
import os

DEVNULL = open(os.devnull,'wb')

cmd_code0 = 'ffmpeg -y -vsync 0 -hwaccel cuvid -c:v h264_cuvid -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output_nv.mp4'
cmd_code1 = 'ffmpeg -y -vsync 0 -c:v h264 -i input.mp4 -c:a copy -c:v h264 -b:v 5M output.mp4'

start_time = time.time()
p0 = sp.Popen(cmd_code0, stdin=sp.PIPE, stdout=DEVNULL, stderr=DEVNULL)
print('Running using GPU...')
p0.wait()   
print('Done!')
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
p1 = sp.Popen(cmd_code0, stdin=sp.PIPE, stdout=DEVNULL, stderr=DEVNULL)
print('Running without GPU...')
p1.wait()   
print('Done!')
print("--- %s seconds ---" % (time.time() - start_time))