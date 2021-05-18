import os
from datetime import datetime

import cv2
import numpy as np
import pynwb
from dateutil.tz import tzlocal
from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import get_reach_indices, get_reach_times
from software.preprocessing.controller_data.controller_data_parser import import_controller_data
from software.preprocessing.trodes_data import experiment_data_parser as trodes_edp
from software.preprocessing.trodes_data.experiment_data_parser import get_exposure_times


def init_nwb_file(file_name, source_script, experimenter, session_start_time):
    """ Function intended to initialize NWB file
    Attributes
    ---------------
    file_name:
    source_script:
    experimenter:
    session_start_time:

    Returns
    ---------------
    nwb_file:


        """
    subject_id, date, session_id = file_name.split("_")
    nwb_file = pynwb.NWBFile(
        session_description='reaching_without_borders',  # required
        identifier=file_name,  # required
        session_start_time=session_start_time,  # required
        experimenter=experimenter,
        session_id=session_id,
        file_create_date=datetime.now(tzlocal()),
        source_script=source_script
    )
    return nwb_file


def add_metadata(list_of_metadata,nwb_file_):
    """
    Function to add metadata to block NWB file
    Attributes
    -----------------
    list_of_metadata : List of metadata, [ Subject, ...]
    nwb_file_: Nwb file being passed into metadata

    Returns
    -----------------
    nwb_file_: Nwb file being passed into metadata

    """
    subject = pynwb.file.Subject(subject_id=list_of_metadata[0])  # consider adding more metadata
    nwb_file_.subject = subject
    return nwb_file_


def save_nwb_file(nwb_file, save_dir):
    """ Function to save NWB file
    Attributes
    ----------------
    nwb_file:
    save_dir:

    Returns
    ----------------
    """
    filename = save_dir + '\\' + nwb_file.identifier + '.nwb'
    io = pynwb.NWBHDF5IO(filename, mode='w')
    io.write(nwb_file)
    io.close()


def add_trodes_analog(nwb_file, trodes_data):
    """ Function to add trodes analog data from an experimental session
    Attributes
    --------------
    nwb_file:
    trodes_data:

    Returns
    --------------
    nwb_file:

    """
    analog_keys = [*trodes_data['analog']]
    pot_keys = [key for key in analog_keys if '_pot' in key]
    analog = pynwb.behavior.Position(name='trodes analog')
    analog.create_spatial_series(
        name=analog_keys[0],
        data=trodes_data['analog'][pot_keys[0]],
        reference_frame='0',
        timestamps=trodes_data['time']['time']
    )
    analog.fields['spatial_series'][analog_keys[0]].fields['unit'] = 'trodes bits'
    for key in pot_keys[1:]:
        spatial_series = pynwb.behavior.SpatialSeries(
            name=key,
            data=trodes_data['analog'][key],
            reference_frame='0',
            timestamps=trodes_data['time']['time']
        )
        analog.add_spatial_series(spatial_series)
        analog.fields['spatial_series'][key].fields['unit'] = 'trodes bits'
    nwb_file.add_acquisition(analog)
    return nwb_file


def add_trodes_dio(nwb_file, trodes_data):
    """ function to add digital IO data from our trodes/spikegadgets experimental system
    Attributes
    ------------
    nwb_file:
    trodes_data:

    Returns
    ---------------
    nwb_file:

    """
    dio_keys = [*trodes_data['DIO']]
    dio = pynwb.behavior.BehavioralEpochs(name='trodes dio')
    for key in dio_keys:
        start_times = trodes_data['DIO'][key][1::2]
        stop_times = trodes_data['DIO'][key][2::2]
        if len(start_times) > 0:
            interval_series = pynwb.misc.IntervalSeries(
                name=key,
                data=[1, -1],
                timestamps=[start_times[0], stop_times[0]]
            )
            for i in range(len(start_times))[1:]:
                interval_series.add_interval(
                    start=float(start_times[i]),
                    stop=float(stop_times[i])
                )
            dio.add_interval_series(interval_series)
    nwb_file.add_acquisition(dio)
    return nwb_file


def trodes_to_nwb(nwb_file, data_dir, trodes_name):
    """ function to export all trodes data (analog/digital data)
    Attributes
    ----------------
    nwb_file:
    data_dir:
    trodes_name:

    Returns
    ------------
    nwb_file:
    """
    trodes_data = trodes_edp.import_trodes_data(data_dir, trodes_name)
    nwb_file = add_trodes_analog(nwb_file, trodes_data)
    nwb_file = add_trodes_dio(nwb_file, trodes_data)
    return nwb_file


def controller_to_nwb(nwb_file, controller_dir):
    """ Function to export controller data to NWB
    Attributes
    ------------
    nwb_file:
    controller_dir:

    Returns
    ------------
    nwb_file:

    """
    controller_data = import_controller_data(controller_dir)
    nwb_file = add_controller_data(nwb_file, controller_data)
    return nwb_file


def add_controller_data(nwb_file, controller_data):
    """ Function to add complete arduino micro-controller data stream from a given experiment.
    Attributes
    -------------
    nwb_file:
    controller_data:

    Returns
    ----------
    nwb_file:

    """
    controller_keys = ['time', 'trial', 'exp_response', 'rob_moving', 'image_triggered', 'in_Reward_Win', 'z_POI']
    c_file = pynwb.behavior.BehavioralEpochs(name='controller_data')
    for key in controller_keys:
        start_times = controller_data[key][1::2]
        stop_times = controller_data[key][2::2]
        if len(start_times) > 0:
            interval_series = pynwb.misc.IntervalSeries(
                name=key,
                data=[1, -1],
                timestamps=[start_times[0], stop_times[0]]
            )
            for i in range(len(start_times))[1:]:
                interval_series.add_interval(
                    start=float(start_times[i]),
                    stop=float(stop_times[i])
                )
            c_file.add_interval_series(interval_series)
    nwb_file.add_acquisition(c_file)
    return nwb_file


def config_to_nwb(nwb_file, config_dir):
    """ Function to extract config data into NWB file for a given experimental session

    Attributes
    -------------
    nwb_file:
    config_dir:

    Returns
    -------------
    nwb_file:

    """
    config_data = import_config_data(config_dir)
    for key in config_data.keys(): # each dict has a key,
        key_parse = config_data[key]
        for item in key_parse.keys():
            print(item)
            # What to add, what class
    # session_id
    # stimulus_notes
    # devices
    return nwb_file


def link_videos(nwb_file, video_dir):
    """ Function to link videos to NWB file
    Attributes
    -------------
    nwb_file:
    video_dir:

    Returns
    -----------
    nwb_file:

    """
    # get cam1,2,3 videos
    for vfiles in os.listdir(video_dir):
        if '_cam1.mp4' in vfiles:
            pass
        if '_cam2.mp4' in vfiles:
            pass
        if '_cam3.mp4' in vfiles:
            pass
    return nwb_file


def link_calibration_videos(nwb_file, cal_dir):
    """ Function to link individual day calibration videos to a given NWB file

    Attributes
    -----------
    nwb_file:
    cal_dir:

    Returns
    -------------
    nwb_file:

    """
    for vfiles in os.listdir(cal_dir):
        if ".mp4" in vfiles:
            # append it
            pass
    return nwb_file


def link_3d_coordinates(nwb_file,video_dir):
    """ Function to link 3-D positional coordinates across a given experimental session to a NWB file
    Attributes
    ------------
    nwb_file:
    video_dir:

    Returns
    -------------
    nwb_file:

    """
    # ReachMaster3D saves a pandas dataframe for each session inside the video directory
    for vfiles in os.listdir(video_dir):
        if 'fullReconstruction3D' in vfiles:
            # add positions to NWB
            pass
    return nwb_file


def add_reachsplitter_predictions(nwb_file,video_dir):# Reachsplitter vector added here
    """ Function to add our lab's custom-built ReachSplitter predictions to NWB file. """
    for vfiles in os.listdir(video_dir):
        if "splitpredictions" in vfiles:
            # add prediction indices to NWB
            pass
    return nwb_file


def link_DLC_predictions(nwb_file,video_dir):
    """ Function to link non-filtered DLC predictions from a given network to a NWB file

    Attributes
    ---------------
    nwb_file:
    video_dir:

    Returns
    ------------
    nwb_file:

    """
    for vfiles in os.listdir(video_dir):
        if "":
            print(vfiles)
    return nwb_file


def link_filtered_DLC_predictions(nwb_file,video_dir): # add the DLC files (need code/write code)
    """ Function to link filtered DLC predictions to a NWB file"""
    return nwb_file


def convertToDateTime(date):
    date = "2" + str(date)
    date_time_obj = datetime.strptime(date, '%Y%m%d')
    return date_time_obj

# this function retrieves the correct position block from a dataframe for a rat, date, and session.

def make_trial_masks(controller_data, experiment_data):
    """
    Parameters
    ----------
    controller_data : list
        list of data from the experimental microcontroller
    experiment_data : dict
        dict of trodes experimental data over sessions
    Returns
    -------
    mask_array : array
        mask containing trial numbers over trodes time
    """
    time = experiment_data['time']['time']
    trials = np.asarray(controller_data['trial'])
    matched_times = match_times(controller_data, experiment_data)
    trial_transitions = np.where(np.roll(trials, 1) != trials)[0]
    num_trials = np.amax(trials)
    mask_array = np.zeros(len(time))
    trial_transitions = matched_times[trial_transitions]
    trial_index = np.searchsorted(time, trial_transitions)
    for i in range(0, num_trials - 1):
        mask_array[trial_index[i]:trial_index[i + 1]] = i + 1
    return mask_array


def get_successful_trials(controller_data, matched_time, experiment_data):
    """
    Parameters
    ----------
    controller_data : list
        list of data from the microcontroller
    matched_time : array
        controller event times converted to trodes time
    experiment_data : dict
        trodes experimental data for each session

    Returns
    -------
    success_rate : list
        list of successful trials
    """
    success_rate = []
    lick_data = experiment_data['DIO']['IR_beam']
    reach_indices = get_reach_indices(controller_data)
    reach_times = get_reach_times(matched_time, reach_indices)
    reach_start = reach_times['start']
    reach_stop = reach_times['stop']
    trial_num = 0
    if lick_data.any():
        for xi in range(len(reach_start)-1):
            i = reach_start[xi]  # these are start and stop times on trodes time
            j = reach_stop[xi]

            if True in np.vectorize(lambda x: i <= x <= j)(lick_data):
                success_rate.append(xi)
    else:
        success_rate.append(-1)
    return success_rate


def make_split_trial_videos(video_path, reach_times):
    """
    Function to split an experimental trial video into discrete videos of each trial inside the session
    Parameters
    ----------
    video_path : str
       full path to video of experiment session
    reach_times : dict
        dict of arrays containing reach start and stop times


    Returns
    -------

    """

    start_times = reach_times['start']  # normalize these times
    stop_times = reach_times['stop']
    try:
        if not os.path.exists(video_path + 'trials'):
            os.makedirs(video_path + 'trials')
    except OSError:
        print('Error: Creating directory of data')
    current_clip = 0
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(str(current_clip) + 'reach.avi', -1, 1, (frameWidth, frameHeight))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    for xi in range(len(start_times)):
        start = start_times[xi] + 1000
        stop = stop_times[xi] + 1000
        for i in range(start, stop):
            video.write(buf[i])
    cv2.destroyAllWindows()
    video.release()
    return


def match_times(controller_data, experiment_data):
    """

    Parameters
    ----------
    controller_data : list
        list of experimental controller variables and values
    experiment_data : dict
        dict of trodes experimental data per session
    Returns
    -------
    controller_time_normalized : array
        array of controller times matched to trodes times, syncing controller and trodes signals
    """
    controller_time = np.asarray(controller_data['time'] / 1000)  # convert to s
    exposures = experiment_data['DIO']['top_cam']  # exposure data
    exposures = get_exposure_times(exposures)
    controller_time_normalized = controller_time - controller_time[-1] + exposures[-1]
    return controller_time_normalized


