"""A module for reading, parsing, and preprocessing trodes data
collected during experiments.

"""
import codecs
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import readTrodesExtractedDataFile3 as read_trodes  # what does the from . import X mean?


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


def load_params(mc_address):
    """Load Microcontroller Metadata File that contains information about the experiment from PNS system

        Parameters
        ----------
        mc_address: str
            path to ECU metadata file

        Returns
        -------
        params : pandas dataframe
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

        """
    os.chdir(mc_address)
    params = read_file(mc_address)
    return params


def read_controller_file(controller_path):
    """ Read Microcontroller Metadata file into a pandas data frame
        Parameters
        ----------
        controller_path: str
            path to Microcontroller metadata file

        Returns
        -------
        params : pandas data frame
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

    """
    controller_files = os.listdir(controller_path)[0]
    params = pd.read_csv(controller_files, delim_whitespace=True, skiprows=1)
    params.columns = ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']
    # time trial exp_response rob_moving image_triggered in_reward_window z_poi
    return params


def find_transitions(params):
    """ Find PNS timestamps that correspond to trial start/end times using ECU metadata
            Parameters
            ----------
            params : pandas data frame
                Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']
            Returns
            -------
            transition_list: numpy array
                A matrix containing N pairs of [start,stop] timestamps
        """
    # find column numbers where s turns to r
    # then find column numbers where r turns to s
    # compare locations, is s > r? is s < r?
    # if s > r, the first transition matrix will be [s, r] entries
    # otherwise, the transition matrix will be [r, s] entries
    # if PNS_flag starts at 0, then we are in the 's' position

    changes = {}

    for col in params.columns:
        changes[col] = [0] + [idx for idx, (i, j) in enumerate(zip(params[col], params[col][1:]), 1) if i != j]
    transition_list = changes['PNS_flag']
    if transition_list[0] > transition_list[1]:
        del transition_list[0]  # get rid of first experiment trial where it initializes in the 'ready' trial state
    len_var = len(transition_list)
    transition_list = np.asarray(transition_list)
    return transition_list


def match_timestamps_index(transition_list, experimental_data):
    """ Find and match PNS and ECU timestamps that correspond to trial start/end times using ECU metadata using
        data from the micro-controller shared between PNS and ECU systems
                Parameters
                ----------
                transition_list: numpy array
                    A matrix containing N pairs of [start,stop] timestamps
                experimental_data : dict
                    Seconds-converted experimental data
                Returns
                -------
                matched_index: np array
                    For N experimental trials, matched_index holds N transition timestamps


            """
    # find the first timestamp in trodes data corresponding to metadata
    matched_index = []
    beam = experimental_data['DIO']['IR_beam']
    for col in beam.columns:
        matched_index[col] = [0] + [idx for idx, (i, j) in enumerate(zip(beam[col], beam[col][1:]), 1) if i != j]
    matched_index = np.asarray(matched_index)
    # check for starting in the 'on' position, delete if true
    if experimental_data['DIO']['IR_beam'][matched_index[0]] == 1:
        del matched_index[0]
    return matched_index


def create_time_mask(transition_list, matched_index):
    """ Create a 'mask' for extracting experiments from raw video data in DLC
                   Parameters
                   ----------
                   transition_list: numpy array
                       A matrix containing N pairs of [start,stop] timestamps
                  matched_index: np array
                    For N experimental trials, matched_index holds N transition timestamps
                   Returns
                   -------
                   time_mask: np array
                       For N experimental trials, holds appropriate start/stop times


               """
    return time_mask


def create_experiment_dataframe(data_dir, trodes_name, mc_dir, sampling_rate=3000, df_address=False):
    """ Create a data frame holding all time-synced experimental metadata
        Parameters
       ----------
       data_dir : str
           Parent directory where the trodes data lives
       trodes_name : str
           Name of original .rec trodes file
        Returns
        --------
        experimental_data : dict
            All of the digital (DIO) and analog data corresponding to the trodes
            files for a experiment. For example, as returned by
            read_data().

        params : pandas data frame
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

        experimental_dataframe : pandas data frame
            data frame containing all time-synced experimental variables

    """
    # Preprocessing steps
    trodes_files = get_trodes_files(data_dir, trodes_name)
    experimental_data = read_data(trodes_files, sampling_rate)
    experimental_data = to_numpy(experimental_data)
    experimental_data = to_seconds(experimental_data)
    params = load_params(mc_dir)
    transitions = find_transitions(params)
    match_timestamps_index(transitions, experimental_data)
    time_mask = create_time_mask(transitions, experimental_data)
    if df_address:
        experimental_dataframe = load_existing_dataframe(df_address)
    else:
        experimental_dataframe = save_existing_dataframe(experimental_data, params, transitions, match_timestamps_index,
                                                         time_mask)
    return experimental_dataframe


def load_existing_dataframe(df_address):
    """ This function loads in previously saved  Hdf5 objects
         Parameters
         ---------
         df_address : str
            path to previously computed experimental dataframe

         Returns
         -------
         old_structure: pandas dataframe
            previous experimental dataframe

         """
    os.chdir(df_address)
    old_structure = pd.read_hdf('exp_df.txt')
    return old_structure


def save_existing_dataframe(e_d, param, trans, match, mask):
    """ This function loads in previously saved  Hdf5 objects
         Parameters
         ---------


         Returns
         -------
         new_df: pandas dataframe (hdf5)
            dataframe (index tbd) containing experimental metadata

         """
    # merge dataframes into object to store
    new_df = {'trodes_data': e_d, 'hyperparams': param, 'transitions': trans, 'matched_time': match, 'masks': mask}
    # save new large df as hdf5 format object.
    with open('exp_df.txt', 'wb') as f:
        json.dump(new_df, codecs.getwriter('utf-8')(f), sort_keys=True, indent=4, ensure_ascii=False)
    return new_df


def plot_data(experiment_data, var_name, time_set=False):
    time = experiment_data['time']['time']
    if time_set:
        time = obtain_times(experiment_data, 30)
    exp_var = experiment_data['DIO'][var_name]
    mask = np.empty(len(time))
    for idx, val in enumerate(time):
        if any(val == c for c in exp_var):
            mask[idx] = 1
        else:
            continue
    plt.plot(time, mask)
    plt.xlabel('time (s)')
    plt.ylabel(var_name)
    plt.title(var_name + ' over experiment')
    plt.show()

    return


def obtain_times(experiment_data, time_length):
    time = experiment_data['time']['time']
    x = 0
    for i in time < time_length:
        x += i
        continue
    time_vector = time[0:x]
    return time_vector


def match_times(controller_data, experiment_data):
    controller_time = controller_data['time']
    exposures = experiment_data['DIO']['triggers']
    controller_time_normalized = controller_time[len(controller_time) - 1]
    last_exposure_time = exposures[len(exposures) - 1]
    norm_array = [controller_time_normalized - c_time for c_time in controller_time]
    norm_array = [last_exposure_time + controller_time for last_exposure_time in controller_time]
    return norm_array


def import_data(trodes_path, trodes_name, ecu_path, win_dir=False):
    controller_data = read_controller_file(ecu_path)
    if win_dir:
        experiment_files = get_trodes_files(trodes_path, trodes_name, win_dir=True)
    else:
        experiment_files = get_trodes_files(trodes_path, trodes_name)
    experiment_data = read_data(experiment_files)
    experiment_data = to_numpy(experiment_data)
    experiment_data = to_seconds(experiment_data)
    norm_times = match_times(controller_data, experiment_data)
    return controller_data, experiment_data, norm_times
"""A module for reading, parsing, and preprocessing trodes data
collected during experiments.

"""
import codecs
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import readTrodesExtractedDataFile3 as read_trodes  # what does the from . import X mean?


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


def load_params(mc_address):
    """Load Microcontroller Metadata File that contains information about the experiment from PNS system

        Parameters
        ----------
        mc_address: str
            path to ECU metadata file

        Returns
        -------
        params : pandas dataframe
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

        """
    os.chdir(mc_address)
    params = read_file(mc_address)
    return params


def read_controller_file(controller_path):
    """ Read Microcontroller Metadata file into a pandas data frame
        Parameters
        ----------
        controller_path: str
            path to Microcontroller metadata file

        Returns
        -------
        params : pandas data frame
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

    """
    controller_files = os.listdir(controller_path)[0]
    params = pd.read_csv(controller_files, delim_whitespace=True, skiprows=1)
    params.columns = ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']
    # time trial exp_response rob_moving image_triggered in_reward_window z_poi
    return params


def find_transitions(params):
    """ Find PNS timestamps that correspond to trial start/end times using ECU metadata
            Parameters
            ----------
            params : pandas data frame
                Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']
            Returns
            -------
            transition_list: numpy array
                A matrix containing N pairs of [start,stop] timestamps
        """
    # find column numbers where s turns to r
    # then find column numbers where r turns to s
    # compare locations, is s > r? is s < r?
    # if s > r, the first transition matrix will be [s, r] entries
    # otherwise, the transition matrix will be [r, s] entries
    # if PNS_flag starts at 0, then we are in the 's' position

    changes = {}

    for col in params.columns:
        changes[col] = [0] + [idx for idx, (i, j) in enumerate(zip(params[col], params[col][1:]), 1) if i != j]
    transition_list = changes['PNS_flag']
    if transition_list[0] > transition_list[1]:
        del transition_list[0]  # get rid of first experiment trial where it initializes in the 'ready' trial state
    len_var = len(transition_list)
    transition_list = np.asarray(transition_list)
    return transition_list


def match_timestamps_index(transition_list, experimental_data):
    """ Find and match PNS and ECU timestamps that correspond to trial start/end times using ECU metadata using
        data from the micro-controller shared between PNS and ECU systems
                Parameters
                ----------
                transition_list: numpy array
                    A matrix containing N pairs of [start,stop] timestamps
                experimental_data : dict
                    Seconds-converted experimental data
                Returns
                -------
                matched_index: np array
                    For N experimental trials, matched_index holds N transition timestamps


            """
    # find the first timestamp in trodes data corresponding to metadata
    matched_index = []
    beam = experimental_data['DIO']['IR_beam']
    for col in beam.columns:
        matched_index[col] = [0] + [idx for idx, (i, j) in enumerate(zip(beam[col], beam[col][1:]), 1) if i != j]
    matched_index = np.asarray(matched_index)
    # check for starting in the 'on' position, delete if true
    if experimental_data['DIO']['IR_beam'][matched_index[0]] == 1:
        del matched_index[0]
    return matched_index


def create_time_mask(transition_list, matched_index):
    """ Create a 'mask' for extracting experiments from raw video data in DLC
                   Parameters
                   ----------
                   transition_list: numpy array
                       A matrix containing N pairs of [start,stop] timestamps
                  matched_index: np array
                    For N experimental trials, matched_index holds N transition timestamps
                   Returns
                   -------
                   time_mask: np array
                       For N experimental trials, holds appropriate start/stop times


               """
    return time_mask


def create_experiment_dataframe(data_dir, trodes_name, mc_dir, sampling_rate=3000, df_address=False):
    """ Create a data frame holding all time-synced experimental metadata
        Parameters
       ----------
       data_dir : str
           Parent directory where the trodes data lives
       trodes_name : str
           Name of original .rec trodes file
        Returns
        --------
        experimental_data : dict
            All of the digital (DIO) and analog data corresponding to the trodes
            files for a experiment. For example, as returned by
            read_data().

        params : pandas data frame
            Microcontroller Metadata  ['time', 'trial', 'PNS', 'PNS_flag', 'triggered', 'inRewardWin', 'zPOI']

        experimental_dataframe : pandas data frame
            data frame containing all time-synced experimental variables

    """
    # Preprocessing steps
    trodes_files = get_trodes_files(data_dir, trodes_name)
    experimental_data = read_data(trodes_files, sampling_rate)
    experimental_data = to_numpy(experimental_data)
    experimental_data = to_seconds(experimental_data)
    params = load_params(mc_dir)
    transitions = find_transitions(params)
    match_timestamps_index(transitions, experimental_data)
    time_mask = create_time_mask(transitions, experimental_data)
    if df_address:
        experimental_dataframe = load_existing_dataframe(df_address)
    else:
        experimental_dataframe = save_existing_dataframe(experimental_data, params, transitions, match_timestamps_index,
                                                         time_mask)
    return experimental_dataframe


def load_existing_dataframe(df_address):
    """ This function loads in previously saved  Hdf5 objects
         Parameters
         ---------
         df_address : str
            path to previously computed experimental dataframe

         Returns
         -------
         old_structure: pandas dataframe
            previous experimental dataframe

         """
    os.chdir(df_address)
    old_structure = pd.read_hdf('exp_df.txt')
    return old_structure


def save_existing_dataframe(e_d, param, trans, match, mask):
    """ This function loads in previously saved  Hdf5 objects
         Parameters
         ---------


         Returns
         -------
         new_df: pandas dataframe (hdf5)
            dataframe (index tbd) containing experimental metadata

         """
    # merge dataframes into object to store
    new_df = {'trodes_data': e_d, 'hyperparams': param, 'transitions': trans, 'matched_time': match, 'masks': mask}
    # save new large df as hdf5 format object.
    with open('exp_df.txt', 'wb') as f:
        json.dump(new_df, codecs.getwriter('utf-8')(f), sort_keys=True, indent=4, ensure_ascii=False)
    return new_df


def plot_data(experiment_data, var_name, time_set=False):
    time = experiment_data['time']['time']
    if time_set:
        time = obtain_times(experiment_data, 30)
    exp_var = experiment_data['DIO'][var_name]
    mask = np.empty(len(time))
    for idx, val in enumerate(time):
        if any(val == c for c in exp_var):
            mask[idx] = 1
        else:
            continue
    plt.plot(time, mask)
    plt.xlabel('time (s)')
    plt.ylabel(var_name)
    plt.title(var_name + ' over experiment')
    plt.show()

    return


def obtain_times(experiment_data, time_length):
    time = experiment_data['time']['time']
    x = 0
    for i in time < time_length:
        x += i
        continue
    time_vector = time[0:x]
    return time_vector


def match_times(controller_data, experiment_data):
    controller_time = controller_data['time']
    exposures = experiment_data['DIO']['triggers']
    controller_time_normalized = controller_time[len(controller_time) - 1]
    last_exposure_time = exposures[len(exposures) - 1]
    norm_array = [c_time - controller_time_normalized + last_exposure_time for c_time in controller_time]
    return norm_array


def import_data(trodes_path, trodes_name, ecu_path, win_dir=False):
    controller_data = read_controller_file(ecu_path)
    if win_dir:
        experiment_files = get_trodes_files(trodes_path, trodes_name, win_dir=True)
    else:
        experiment_files = get_trodes_files(trodes_path, trodes_name)
    experiment_data = read_data(experiment_files)
    experiment_data = to_numpy(experiment_data)
    experiment_data = to_seconds(experiment_data)
    norm_times = match_times(controller_data, experiment_data)
    return controller_data, experiment_data, norm_times
