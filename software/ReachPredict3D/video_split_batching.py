""" This module provides functions to split un-split recorded experimental videos
using ffmpeg and vidgear options

"""
import cv2
from vidgear.gears import WriteGear
import pdb
import numpy as np


def conver2bgr(frame):
        """Function to convert image to bgr color scheme

        Attributes

        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        return frame


def enhanceImage(frame):
        cols, rows, ch = frame.shape
        brightness = np.sum(frame) / (ch * 255 * cols * rows)
        minimum_brightness = 0.2
        alpha = brightness / minimum_brightness
        ratio = brightness / minimum_brightness
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=255 * (minimum_brightness - brightness))
        return frame


def mainrun_split(self,input):
        intput_filename = input[0]
        no_of_cam = 3
        crf = '2'
        pix_Format = 'yuv420p'
        cap = cv2.VideoCapture(str(intput_filename))
        print('opening filename' + str(intput_filename))
        if cap.isOpened():
            print("Error opening video file")
            pdb.set_trace()
        fps = int(cap.get(5))
        width = int(cap.get(3) / no_of_cam)
        height = int(cap.get(4))
        nframes = int(cap.get(7))
        output_params = {'-c:v': 'h264', '-crf': crf, '-input_framerate': fps, '-pix_fmt': pix_Format, \
                         '-preset': 'fast', '-tune': 'zerolatency', '-output_dimensions': (width, height)}
        print('Start converting...      ', end='', flush=True)
        writers = []
        for i in range(no_of_cam):
            output_filename = intput_filename.split('.')[0] + '_cam' + str(i + 1) + '.mp4'
            writers.append(WriteGear(output_filename=output_filename, compression_mode=True, \
                                     logging=False, **output_params))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                for i, w in enumerate(writers):
                    index = range(width * i, width * i + width)
                    frame_ = frame[:, index]
                    frame_ = conver2bgr(frame_)
                    frame_ = enhanceImage(frame_)
                    w.write(frame_)
        for w in writers:
            w.close()
        cap.release()
        cv2.destroyAllWindows()
        print('[DONE]')
