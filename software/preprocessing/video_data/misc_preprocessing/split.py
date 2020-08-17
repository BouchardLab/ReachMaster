import cv2
from vidgear.gears import WriteGear
import numpy as np

import glob



def get_video_files(path_list, glob_path):
    """

    Parameters
    ----------
    path_list : empty list
    glob_path : recursive path used for finding un-split .mp4 files

    Returns
    -------
    path_list : list of paths to all .mp4 files inside directories
    """
    path_list.append(glob.glob(glob_path, recursive=True))
    return path_list


def batch_vid_split(glob_path):
    """
    Function to split videos using VidGear.
    Parameters
    ----------
    glob_path : string, recursive path used to iterate over sub-directories and find .mp4 files

    Returns
    -------

    """
    print("Starting..")
    pathList = []
    pathList = get_video_files(glob_path, pathList)
    for file in pathList[0]:
        # check for _cam in file name, if its there, move on to next item
        if "cam" in file:
            print('File has already been added')
            continue
        if "deb" in file:
            print('debayered/depreciated file..')
            continue
        else:
            filename = file.rsplit('.mp4', 1)[0]
            print(filename)
            # check other file names and see if there are files w/ cam extension already
            path = file.rsplit('/', 1)[0]
            print(path)
            for files in glob.glob(path):
                if "cam" in files:
                    print('File has already been added..')
                    continue
                elif "calibration" in path:
                    print('Calibration file, doesnt need to be split..')
                    continue
                else:
                    print(file + ' is being added..')
                    mainrun(file)


def conver2bgr(frame):
    """
    Function to convert frames to bgr color scheme
    Parameters
    ----------
    frame : video frame

    Returns
    -------
    frame : converted video frame
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
    return frame


def enhanceImage(frame):
    """
    Function to enhance array image.
    Parameters
    ----------
    frame : array, frame image

    Returns
    -------
    frame : array, frame image
    """
    cols, rows, ch = frame.shape
    brightness = np.sum(frame) / (ch * 255 * cols * rows)
    minimum_brightness = 0.2
    alpha = brightness / minimum_brightness
    ratio = brightness / minimum_brightness
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=255 * (minimum_brightness - brightness))
    return frame


def mainrun(input):
    """
    Main video splitting function w/ VideoGear (uses ffmpeg conversions)
    Parameters
    ----------
    input : input filename or filepath

    Returns
    -------

    """
    intput_filename = input
    no_of_cam = 3
    crf = '25'
    pix_Format = 'yuv420p'

    cap = cv2.VideoCapture(intput_filename)
    if (cap.isOpened() == False):
        print("Error opening video file")

    fps = int(cap.get(5))
    width = int(cap.get(3) / no_of_cam)
    height = int(cap.get(4))

    output_params = {'-c:v': 'h264', '-crf': crf, '-input_framerate': fps, '-pix_fmt': pix_Format, \
                     '-preset': 'fast', '-tune': 'zerolatency', '-output_dimensions': (width, height)}

    print('Start converting...      ', end='', flush=True)

    writers = []
    for i in range(no_of_cam):
        output_filename = intput_filename.split('.')[0] + '_cam' + str(i + 1) + '.mp4'
        writers.append(WriteGear(output_filename=output_filename, compression_mode=True, \
                                 logging=False, **output_params))

    while (cap.isOpened()):
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


glob_path = '/home/bnelson/Data/PNS_data/RM9/**/*.mp4'
batch_vid_split(glob_path)
