"""A module for reading, parsing, and preprocessing controller data
collected during experiments.

"""
import os

import numpy as np
import pandas as pd


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
