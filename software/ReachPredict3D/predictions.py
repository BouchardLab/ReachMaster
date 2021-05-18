"""This module is intended to provide methods to iterate over all video directories
for a given rat, split the videos into individual camera videos if this has not
yet been done, and use a pre-trained DLC network to obtain 2-D predictions of
experimental data for multiple rat positions.
"""

import os
import deeplabcut
import tensorflow as tf
import glob


def find_cam_files(root_dir):
    """Function to find cam files for DeepLabCut to run our predictive network on!
    Attributes
    ----------
    root_dir : path directory
    Returns
    -------
    cam1_array: list of paths to experimental video files from camera1
    cam2_array: list of paths to experimental video files from camera2
    cam3_array: list of paths to experimental video files from camera3

    """
    cam1_array = []
    cam2_array = []
    cam3_array = []
    for file in glob.glob(root_dir, recursive=True):
        path = file.rsplit('/', 1)[0]
        path = path+'/'
        if "shuffle2" in file:
            print("File has been analyzed already!" + file)
            sig_flag = 0
        else:
            sig_flag = 1
        if "cam1" in file: #check and make sure that the files have been split
            if sig_flag == 1:
                cam1_array.append(file)
        elif "cam2" in file:
            if sig_flag == 1:
                cam2_array.append(file)
        elif "cam3" in file:
            if sig_flag == 1:
                cam3_array.append(file)
    return cam1_array, cam2_array, cam3_array


def run_analysis_videos(cam_video_paths, config_path, filtering=False):
    """function to run deeplabcut on a list of videos from a single cam

    Attributes
    -------------
    cam_video_paths: list of video paths from individual cams
    config_path: DLC config path
    filtering: boolean, uses (at the moment) median filtering from DLC

    """
    shuffle = 2 # shuffle of the network to use
    print("Starting to extract files..")
    deeplabcut.analyze_videos(config_path, cam_video_paths, videotype='mp4', shuffle=shuffle, save_as_csv=True)
    if filtering:
        deeplabcut.filterpredictions(config_path, cam_video_paths, videotype='.mp4', shuffle=shuffle)


def run_main(root_dir, config, fset=False):
    """function intended to loop over PNS directory, obtain each camera's files, and send them to DLC.

    Attributes
    ------------
    root_dir: path to PNS root_dir to iterate over
    config: path to DLC config file
    fset: boolean, filtering flag

    """
    pathList = []
    cam1_array, cam2_array, cam3_array = find_cam_files(root_dir, config)
    #print(cam1_array)
    print('Starting Cam 1 DLC analysis')
    #sleep(5)
    print('Number of Videos to analyze :  ' + str(len(cam1_array)))
    run_analysis_videos(cam1_array,config, filtering=fset)
    print('Starting Cam 2 DLC analysis')
    print('Number of Videos to analyze :  ' + str(len(cam2_array)))
    run_analysis_videos(cam2_array,config,filtering=fset)
    print('Starting Cam 3 DLC analysis')
    print('Number of Videos to analyze :  ' + str(len(cam3_array)))
    run_analysis_videos(cam3_array,config,filtering=fset)
    print('Finished extracting 2-D DLC estimates!')


class GetPredictions:
    """Class to run DLC over a given Rat's PNS directory containing experimental video. Resulting
    files are saved inside of the PNS folder, filtering is optional!
    """
    def __init__(self,root_dir,config_dir, GPU_num=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_num
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        run_main(root_dir, config_dir)
