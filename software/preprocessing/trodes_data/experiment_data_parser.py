"""A module for reading, parsing, and preprocessing trodes data
collected during experiments.

"""

import json
import os

import numpy as np
import pandas as pd
from . import readTrodesExtractedDataFile3 as read_trodes


def get_trodes_files(data_dir, trodes_name, win_dir=False):  # pass in data directory, name of rec file
    """Generate names of all the trodes files in an experimental session.
       Assumes data is saved in the default trodes filesystem and channels are
       named appropriately in the trodes configuration file.

       Parameters
       ----------
       data_dir : str
           Parent directory where the trodes data lives
       trodes_name : str
           Name of original .rec trodes file
        win_dir : boolean
            Flag, True if using a windows directory system
       Returns
       -------
       trodes_files : dict
           The file names for each channel recording during experiments. More
           specifically, 'time' would be the .dat file containing timestamps for each experimental frame.
           'solenoid' would contain data from the .dat file containing experimental information about the
           solenoid used to deliver water as a reward during experiments.

       """
    if win_dir:
        trodes_files = {
            'time_file': data_dir + '\\%s\\%s.analog\\%s.timestamps.dat' % (trodes_name, trodes_name, trodes_name),
            'x_push_file': data_dir + '\\%s\\%s.DIO\\%s.dio_xPush.dat' % (trodes_name, trodes_name, trodes_name),
            'x_pull_file': data_dir + '\\%s\\%s.DIO\\%s.dio_xPull.dat' % (trodes_name, trodes_name, trodes_name),
            'y_push_file': data_dir + '\\%s\\%s.DIO\\%s.dio_yPush.dat' % (trodes_name, trodes_name, trodes_name),
            'y_pull_file': data_dir + '\\%s\\%s.DIO\\%s.dio_yPull.dat' % (trodes_name, trodes_name, trodes_name),
            'z_push_file': data_dir + '\\%s\\%s.DIO\\%s.dio_zPush.dat' % (trodes_name, trodes_name, trodes_name),
            'z_pull_file': data_dir + '\\%s\\%s.DIO\\%s.dio_zPull.dat' % (trodes_name, trodes_name, trodes_name),
            'moving_file': data_dir + '\\%s\\%s.DIO\\%s.dio_moving.dat' % (trodes_name, trodes_name, trodes_name),
            'triggers_file': data_dir + '\\%s\\%s.DIO\\%s.dio_triggers.dat' % (trodes_name, trodes_name, trodes_name),
            'IR_beam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_IRbeam.dat' % (trodes_name, trodes_name, trodes_name),
            'led_file': data_dir + '\\%s\\%s.DIO\\%s.dio_led.dat' % (trodes_name, trodes_name, trodes_name),
            'left_Cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_leftCam.dat' % (trodes_name, trodes_name, trodes_name),
            'right_Cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_rightCam.dat' % (trodes_name, trodes_name, trodes_name),
            'top_Cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_topCam.dat' % (trodes_name, trodes_name, trodes_name),
            'lights_file': data_dir + '\\%s\\%s.DIO\\%s.dio_lights.dat' % (trodes_name, trodes_name, trodes_name),
            'solenoid_file': data_dir + '\\%s\\%s.DIO\\%s.dio_solenoid.dat' % (trodes_name, trodes_name, trodes_name),
            'x_pot_file': data_dir + '\\%s\\%s.analog\\%s.analog_potX.dat' % (trodes_name, trodes_name, trodes_name),
            'y_pot_file': data_dir + '\\%s\\%s.analog\\%s.analog_potY.dat' % (trodes_name, trodes_name, trodes_name),
            'z_pot_file': data_dir + '\\%s\\%s.analog\\%s.analog_potZ.dat' % (trodes_name, trodes_name, trodes_name)
        }
    else:
        trodes_files = {
            'time_file': data_dir + '/%s/%s.analog/%s.timestamps.dat' % (trodes_name, trodes_name, trodes_name),
            'x_push_file': data_dir + '/%s/%s.DIO/%s.dio_xPush.dat' % (trodes_name, trodes_name, trodes_name),
            'x_pull_file': data_dir + '/%s/%s.DIO/%s.dio_xPull.dat' % (trodes_name, trodes_name, trodes_name),
            'y_push_file': data_dir + '/%s/%s.DIO/%s.dio_yPush.dat' % (trodes_name, trodes_name, trodes_name),
            'y_pull_file': data_dir + '/%s/%s.DIO/%s.dio_yPull.dat' % (trodes_name, trodes_name, trodes_name),
            'z_push_file': data_dir + '/%s/%s.DIO/%s.dio_zPush.dat' % (trodes_name, trodes_name, trodes_name),
            'z_pull_file': data_dir + '/%s/%s.DIO/%s.dio_zPull.dat' % (trodes_name, trodes_name, trodes_name),
            'moving_file': data_dir + '/%s/%s.DIO/%s.dio_moving.dat' % (trodes_name, trodes_name, trodes_name),
            'triggers_file': data_dir + '/%s/%s.DIO/%s.dio_triggers.dat' % (trodes_name, trodes_name, trodes_name),
            'IR_beam_file': data_dir + '/%s/%s.DIO/%s.dio_IRbeam.dat' % (trodes_name, trodes_name, trodes_name),
            'led_file': data_dir + '/%s/%s.DIO/%s.dio_led.dat' % (trodes_name, trodes_name, trodes_name),
            'left_Cam_file': data_dir + '/%s/%s.DIO/%s.dio_leftCam.dat' % (trodes_name, trodes_name, trodes_name),
            'right_Cam_file': data_dir + '/%s/%s.DIO/%s.dio_rightCam.dat' % (trodes_name, trodes_name, trodes_name),
            'top_Cam_file': data_dir + '/%s/%s.DIO/%s.dio_topCam.dat' % (trodes_name, trodes_name, trodes_name),
            'lights_file': data_dir + '/%s/%s.DIO/%s.dio_lights.dat' % (trodes_name, trodes_name, trodes_name),
            'solenoid_file': data_dir + '/%s/%s.DIO/%s.dio_solenoid.dat' % (trodes_name, trodes_name, trodes_name),
            'x_pot_file': data_dir + '/%s/%s.analog/%s.analog_potX.dat' % (trodes_name, trodes_name, trodes_name),
            'y_pot_file': data_dir + '/%s/%s.analog/%s.analog_potY.dat' % (trodes_name, trodes_name, trodes_name),
            'z_pot_file': data_dir + '/%s/%s.analog/%s.analog_potZ.dat' % (trodes_name, trodes_name, trodes_name)
        }
    return trodes_files


def read_data(trodes_files, sampling_rate=3000):
    """Read all the trodes file data using the SpikeGadgets
    `readTrodesExtractedDataFile` script.

    Parameters
    ----------
    trodes_files : dict
        The file names for each channel recording during an experiment. For
        example, as returned by get_trodes_files().
    sampling_rate : int
        Specifying a rate (Hz) lower than the SpikeGadgets MCU clock rate of
        30 kHz will downsample the data to speed up parsing (ds is the downsampling variable).

    Returns
    -------
    experimental_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for an experiment.

    """
    clockrate = np.float_(read_trodes.readTrodesExtractedDataFile(trodes_files['time_file'])['clockrate'])
    ds = int(clockrate / sampling_rate)
    experimental_data = {
        'clockrate': clockrate,
        'sampling_rate': sampling_rate,
        'time': {
            'units': 'samples',
            'time': read_trodes.readTrodesExtractedDataFile(trodes_files['time_file'])['data'][0:-1:ds]
        },
        'DIO': {
            'x_push': read_trodes.readTrodesExtractedDataFile(trodes_files['x_push_file'])['data'],
            'x_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['x_pull_file'])['data'],
            'y_push': read_trodes.readTrodesExtractedDataFile(trodes_files['y_push_file'])['data'],
            'y_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['y_pull_file'])['data'],
            'z_push': read_trodes.readTrodesExtractedDataFile(trodes_files['z_push_file'])['data'],
            'z_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['z_pull_file'])['data'],
            'moving': read_trodes.readTrodesExtractedDataFile(trodes_files['moving_file'])['data'],
            'IR_beam': read_trodes.readTrodesExtractedDataFile(trodes_files['IR_beam_file'])['data'],
            'triggers': read_trodes.readTrodesExtractedDataFile(trodes_files['triggers_file'])['data'],
            'lights': read_trodes.readTrodesExtractedDataFile(trodes_files['lights_file'])['data'],
            'led': read_trodes.readTrodesExtractedDataFile(trodes_files['led_file'])['data'],
            'topCam': read_trodes.readTrodesExtractedDataFile(trodes_files['top_Cam_file'])['data'],
            'leftCam': read_trodes.readTrodesExtractedDataFile(trodes_files['left_Cam_file'])['data'],
            'rightCam': read_trodes.readTrodesExtractedDataFile(trodes_files['right_Cam_file'])['data'],
            'solenoid': read_trodes.readTrodesExtractedDataFile(trodes_files['solenoid_file'])['data']
        },
        'analog': {
            'x_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['x_pot_file'])['data']['voltage'][0:-1:ds],
            'y_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['y_pot_file'])['data']['voltage'][0:-1:ds],
            'z_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['z_pot_file'])['data']['voltage'][0:-1:ds]
        }
    }
    return experimental_data


def to_numpy(experimental_data):
    """Convert experimental data to numpy arrays

    Parameters
    ----------
    experimental_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a experimental recording. For example, as returned by
        read_data().

    Returns
    -------
    experimental_data : dict
        Numpy-converted experimental data.

    """
    experimental_data['time']['time'] = np.array(
        [t[0] for t in experimental_data['time']['time']],
        dtype='float_'
    )
    for key in experimental_data['DIO'].keys():
        experimental_data['DIO'][key] = np.array(
            [i[0] for i in experimental_data['DIO'][key]],
            dtype='float_'
        )
    return experimental_data


def to_seconds(experimental_data, start_at_zero=True):
    """Convert the experimental data time units to seconds.

    Parameters
    ----------
    experimental_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for a experiment. For example, as returned by
        read_data().
    start_at_zero : bool
        If True, the start time will be set to 0.

    Returns
    -------
    experimental_data : dict
        Seconds-converted experimental data

    """
    if experimental_data['time']['units'] is not 'seconds':
        if start_at_zero:
            for key in experimental_data['DIO'].keys():
                experimental_data['DIO'][key] = (
                                                        experimental_data['DIO'][key] -
                                                        experimental_data['time']['time'][
                                                            0]
                                                ) / experimental_data['clockrate']
            experimental_data['time']['time'] = (
                                                        experimental_data['time']['time'] -
                                                        experimental_data['time']['time'][0]
                                                ) / experimental_data['clockrate']
        else:
            for key in experimental_data['DIO'].keys():
                experimental_data['DIO'][key] = experimental_data['DIO'][key] / experimental_data['clockrate']
            experimental_data['time']['time'] = experimental_data['time']['time'] / experimental_data['clockrate']
    else:
        pass
    return experimental_data


def read_controller_file(controller_path):
    """ Read Microcontroller Metadata file into a pandas data frame
        Parameters
        ----------
        controller_path: str
            path to Microcontroller metadata file

        Returns
        -------
        params : pandas data frame
            Microcontroller Metadata  ['time', 'trial', 'exp_response', 'rob_moving', 'image_triggered', 'in_Reward_Win', 'z_POI']

    """
    controller_files = os.listdir(controller_path)[0]
    params = pd.read_csv(controller_files, delim_whitespace=True, skiprows=1)
    params.columns = ['time', 'trial', 'exp_response', 'rob_moving', 'image_triggered', 'in_Reward_Win', 'z_POI']
    return params


def create_DIO_mask(time_data, trodes_data):
    """

    Parameters
    ----------
    time_data : array
        trodes experimental time array eg ['time']['time']
    trodes_data
        data used to create mask eg 'led' or 'topCam'
    Returns
    -------
    mask : array
        binary mask (1 indicates ongoing signal from variable)
    """
    mask = np.empty(len(time_data))
    for idx, val in enumerate(time_data):
        if any(val == c for c in trodes_data):
            mask[idx] = 1
        else:
            continue
    return mask


def obtain_times(experiment_data, time_length):
    """

    Parameters
    ----------
    experiment_data : dict
        dict containing trodes experimental data
    time_length : int
        variable to truncate time array by eg 5 would truncate time to 5 seconds

    Returns
    -------
    time_vector : array
        truncated array of trodes times
    """
    time = experiment_data['time']['time']
    x = 0
    for i in time < time_length:
        x += i
        continue
    time_vector = time[0:x]
    return time_vector


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
    exposures = experiment_data['DIO']['topCam']  # exposure data
    exposures = get_exposure_times(exposures)
    controller_time_normalized = controller_time - controller_time[-1] + exposures[-1]
    return controller_time_normalized


def get_exposure_times(exposures):
    """

    Parameters
    ----------
    exposures : array
        trodes DIO file containing camera exposures

    Returns
    -------
    real_exposures : array
        estimated true exposure events

    """
    exposures_high = exposures[1::2]
    real_exposures = exposures_high
    return real_exposures


def get_exposure_masks(exposures, time):
    """

    Parameters
    ----------
    exposures : array
        trodes DIO file containing camera exposure times
    time : array
        array of trodes times

    Returns
    -------
    mask_array : array
        array containing exposure masks (1 indicates exposure process starting)

    """
    exposures_low = exposures[1::2]
    exposures_high = exposures[2::2]
    mask_array = np.zeros(len(time))
    high_index = np.searchsorted(time, exposures_high)
    for y in high_index:
        mask_array[y] = 1
    return mask_array


def import_controller_data(mc_path):
    """

    Parameters
    ----------
    mc_path : str
        full path of microcontroller data file

    Returns
    -------
    controller_data : list
        list of arrays containing controller data (reach events, robot movement etc)
    """
    controller_data = read_controller_file(mc_path)
    return controller_data


def import_trodes_data(trodes_path, trodes_name, win_dir=False):
    """

    Parameters
    ----------
    trodes_path : str
        location of trodes data files
    trodes_name : str
        name of trodes file eg. RM1520190917_xxxx
    win_dir : boolean
        indicate if your computer is using a windows path with True

    Returns
    -------
    experiment_data : dict
        returns dict of arrays containing trodes data from a session
    """
    if win_dir:
        experiment_files = get_trodes_files(trodes_path, trodes_name, win_dir=True)
    else:
        experiment_files = get_trodes_files(trodes_path, trodes_name)
    experiment_data = read_data(experiment_files)
    experiment_data = to_numpy(experiment_data)
    experiment_data = to_seconds(experiment_data)
    return experiment_data


def get_reach_indices(controller_data):
    """

    Parameters
    ----------
    controller_data : list
        list containing data from experimental microcontroller

    Returns
    -------
    reach_indices : list
        list containinng start and stop indices of the controller data
    """
    end_index = []
    start_index = []
    for i, j in enumerate(controller_data['exp_response']):
        if j == 'e':
            end_index.append(i)
        if j == 'r':
            if controller_data['exp_response'][i - 1] == 'r':
                continue
            else:
                start_index.append(i)
    reach_indices = {'start': start_index, 'stop': end_index}
    return reach_indices


def get_reach_times(controller_time, reach_indices):
    """

    Parameters
    ----------
    controller_time : list
        list containing CONVERTED controller times (use match_times first!)
    reach_indices : list
        list containing reach indices corresponding to entries in controller data

    Returns
    -------
    reach_times : list
        list containing start and stop reach times in trodes time
    """
    reach_times = {'start': [], 'stop': []}
    reach_start = reach_indices['start']
    reach_stop = reach_indices['stop']
    for i in reach_start:
        reach_times['start'].append(controller_time[i])
    for i in reach_stop:
        reach_times['stop'].append(controller_time[i])
    return reach_times


def make_reach_masks(reach_times, time):
    """

    Parameters
    ----------
    reach_times : list
        list of array of reach times in converted trodes time
    time : array
        reach times converted into trodes time
    Returns
    -------
    mask_array : array
        array containing  binary mask for reach events (1 indicates ongoing reach)
    """
    reach_start = reach_times['start']
    reach_stop = reach_times['stop']
    mask_array = np.zeros(len(time))
    start_index = np.searchsorted(time, reach_start)
    stop_index = np.searchsorted(time, reach_stop)
    for xi in range(len(start_index)):
        i = start_index[xi]
        j = stop_index[xi]
        mask_array[i:j] = 1
    return mask_array


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
    for xi in range(len(reach_start)):
        i = reach_start[xi]  # these are start and stop times on trodes time
        j = reach_stop[xi]
        if True in np.vectorize(lambda x: i <= x <= j)(lick_data):
            success_rate.append(xi)
    return success_rate


def get_config(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to experimental metadata file

    Returns
    -------
    config_file : dict
        dict of experimental metadata from each experiment session
    """
    files = [i for i in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, i)) and \
             'Workspace%' in i]
    os.chdir(config_path)
    config_file = json.load(open(files[0]))
    return config_file


def import_config_data(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to the experimental configuration file

    Returns
    -------
    config data : dict
        dict containing relevant experimental metadata
    """
    data = get_config(config_path)
    config_data = {'command__file': data['RobotSettings']['commandFile'], 'x_pos': data['RobotSettings']['xCommandPos'],
                   'y_pos': data['RobotSettings']['yCommandPos'], 'z_pos': data['RobotSettings']['zCommandPos']}
    return config_data


def get_trodes_robot_data (e_data):
    data = {

    }
    return data


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
    while (fc < frameCount and ret):
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
