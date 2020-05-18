""" Utilities to transform cropped experimental video data into uncropped data
    usable for labeling in DLT w/ calibration
    Crop parameters are found from the config file used in a given experiment.
    Also contains functions to pad videos for labeling in DLT.

"""
import cv2
import numpy as np


def get_crop_cfg(config_file):
    offset_x = config_file['CameraSettings']['offsetX']
    offset_y = config_file['CameraSettings']['offsetY']
    img_height = config_file['CameraSettings']['imgHeight']
    img_width = config_file['CameraSettings']['imgWidth']
    crop_cfg = [offset_x, offset_y, img_height, img_width]
    return crop_cfg


def pad_labeled_data(labeled_data, crop_cfg):
    labeled_data[0::2, :] = labeled_data[0::2, :] + crop_cfg[0]  # x offsets
    labeled_data[0::1, :] = labeled_data[0::1, :] + crop_cfg[0]  # y offsets
    return labeled_data


def get_video_frames(video_name):
    cap = cv2.VideoCapture(video_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf, frameCount, frameWidth, frameHeight


def video_pad(file_name, crop_cfg, video=False):
    height = crop_cfg[0]
    width = crop_cfg[1]
    height_crop = crop_cfg[1] / 2
    length_crop = crop_cfg[0] / 2
    if video:
        buf, frameCount, frameWidth, frameHeight = get_video_frames(file_name)
    # read image
    img = cv2.imread(file_name)
    ht, wd, cc = img.shape

    # create new image of desired size and color (blue) for padding
    ww = 300
    hh = 300
    color = (255, 0, 0)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy + ht, xx:xx + wd] = img

    # view result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save result
    cv2.imwrite("lena_centered.jpg", result)
    # position video over blank_image
    # blank_video = np.zeros((frameCount, height, width, 3), np.uint8)

    for i in frameCount:
        blank_video[i, height_crop:height_crop + crop_cfg[2], length_crop:length_crop + crop_cfg[3]] = buf[i, :, :]

    return
