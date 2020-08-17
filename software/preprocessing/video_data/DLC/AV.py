import deeplabcut
import glob
import tensorflow as tf
import os


def initialize_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return


def find_cam_files(root_dir):
    """
    Function to find cam files for 3-D calibration
    Parameters
    ----------
    root_dir : path directory
    Returns
    -------
    [cam1_array, cam2_array, cam3_array]: lists containing paths for individual camera .mp4 video files for experiments
    """
    cam1_array = [] 
    cam2_array = [] 
    cam3_array=[]
    sig_flag=1
    for file in glob.glob(root_dir, recursive=True):
        path = file.rsplit('/',1)[0]
        path=path+'/'
        for files in os.listdir(path):
            #print(files)
            if "resnet_101" in files:
                print("File has been analyzed already!" + files)
                sig_flag=0
        if "cam1" in file:# check and make sure that the files have been split
            if sig_flag==1:
                cam1_array.append(file)
        elif "cam2" in file:
            if sig_flag==1:
                cam2_array.append(file)
        elif "cam3" in file:
            if sig_flag==1:
                cam3_array.append(file)
        else:
            print('Sorry, file not found :: file designation' + file)
        sig_flag=1
    return cam1_array,cam2_array,cam3_array


def run_analysis_videos(cam_video_paths,filtertype='arima', windowlength=20, p_bound=0.9, ARdegree=0,
                        MAdegree=2,alpha=0.1):
    """
    Function to analyze an experimental video using pre-trained deeplabcut neural networks to predict specified
    joint locations from raw video files obtained during an experiment.
    Additionally, DeepLabCut can filter data using (S)ARIMAX models. For more information, please see
    https://github.com/DeepLabCut/DeepLabCut
    Information is saved inside the video file to a .csv file necessary for 3-D reconstruction.
    Parameters
    ----------
    cam_video_paths : list of file paths for each individual camera

    Returns
    -------

    """
    # set up network and get it ready
    config_path = '/clusterfs/bebb/users/bnelson/DLC/CatScan-Brett-2020-07-27/config.yaml'
    shuffle = 5 # shuffle of the network to use
    print("Starting to extract files..")
    # analyze each video path
    deeplabcut.analyze_videos(config_path, cam_video_paths, videotype='mp4', shuffle=shuffle, 
                              save_as_csv=True)
    deeplabcut.filterpredictions(config_path, cam_video_paths, videotype='.mp4',shuffle=shuffle,
                                 filtertype=filtertype, windowlength=windowlength,
                                 p_bound=p_bound, ARdegree=ARdegree, MAdegree=MAdegree,alpha=alpha, save_as_csv=True)
    return


def run_main(root_dir):
    """
    Function to run AV extraction for experiments
    Parameters
    ----------
    root_dir : root dir to find analyzable experimental video files in (must be glob recursive)

    Returns
    -------

    """
    initialize_gpu()
    cam1_array, cam2_array, cam3_array = find_cam_files(root_dir)
    print('Starting Cam 1')
    run_analysis_videos(cam1_array)
    print('Starting Cam 2')
    run_analysis_videos(cam2_array)
    print('Starting Cam 3')
    run_analysis_videos(cam3_array)
    print('Finished extracting!')

# execute script
run_main('/home/bnelson/Data/PNS_data/RM12/**/*mp4') # I need 13 to be ran still