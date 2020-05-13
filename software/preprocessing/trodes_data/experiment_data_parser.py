"""A module for reading, parsing, and preprocessing trodes data
collected during experiments.

"""

import numpy as np

from . import readTrodesExtractedDataFile3 as read_trodes


def get_trodes_files(data_dir, trodes_name, win_dir=False):  # pass in data directory, name of rec file
    """Generate names of all the trodes files in an experimental session.
       Assumes data is saved in the default trodes filesystem and channels are
       named appropriately in the trodes configuration file.

       Parameters
       ----------
       win_dir : boolean
            Flag, True if using a windows directory system
       data_dir : str
           Parent directory where the trodes data lives
       trodes_name : str
           Name of original .rec trodes file

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
            'left_cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_leftCam.dat' % (trodes_name, trodes_name, trodes_name),
            'right_cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_rightCam.dat' % (trodes_name, trodes_name, trodes_name),
            'top_cam_file': data_dir + '\\%s\\%s.DIO\\%s.dio_topCam.dat' % (trodes_name, trodes_name, trodes_name),
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
            'left_cam_file': data_dir + '/%s/%s.DIO/%s.dio_leftCam.dat' % (trodes_name, trodes_name, trodes_name),
            'right_cam_file': data_dir + '/%s/%s.DIO/%s.dio_rightCam.dat' % (trodes_name, trodes_name, trodes_name),
            'top_cam_file': data_dir + '/%s/%s.DIO/%s.dio_topCam.dat' % (trodes_name, trodes_name, trodes_name),
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
            'top_cam': read_trodes.readTrodesExtractedDataFile(trodes_files['top_cam_file'])['data'],
            'left_cam': read_trodes.readTrodesExtractedDataFile(trodes_files['left_cam_file'])['data'],
            'right_cam': read_trodes.readTrodesExtractedDataFile(trodes_files['right_cam_file'])['data'],
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
    exposures_low = exposures[0::2]
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
