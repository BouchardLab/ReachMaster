""" Utilities to transform cropped experimental video data into uncropped data
    usable for labeling in DLT w/ calibration
    Crop parameters are found from the config file used in a given experiment.
    Also contains functions to pad videos for labeling in DLT.

"""
import cv2
import numpy as np

# hard-code for the moment
video_name = "00_40_30_cam1.mp4"
crop_cfg = [304, 168, 0, 0]


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


def get_video_frames(video_name, crop_cfg):
    cap = cv2.VideoCapture(video_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        buf[fc] = pad_frame(buf[fc])
        # add crop_cfg pass
        fc += 1
    cap.release()
    return buf, frameCount, frameWidth, frameHeight, frameRate


def pad_frame(frame, config):
    frame = cv2.copyMakeBorder(frame, 84, 84, 152, 152, cv2.BORDER_CONSTANT)
    return frame


def video_pad(file_name, crop_cfg, video=True):
    height = crop_cfg[0]
    width = crop_cfg[1]
    height_crop = crop_cfg[1] / 2
    length_crop = crop_cfg[0] / 2
    if video:
        buf, frameCount, frameWidth, frameHeight, frameRate = get_video_frames(file_name, crop_cfg)
        # fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.cv.CV_FOURCC(*'XVID')
    videowriter = cv2.VideoWriter(file_name, fourcc, frameRate, (width + frameWidth, height + frameHeight))
    for f in buf:
        videowriter.write(buf[f])
    videowriter.release()
    cv2.destroyAllWindows()
    return


# code that runs the video padding


video_pad(video_name, crop_cfg)
