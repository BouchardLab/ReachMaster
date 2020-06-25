"""Script to import trodes, micro-controller, and config data

"""
import glob
import os

import numpy as np
import pandas as pd

from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data, get_exposure_times


def load_files(trodes_dir, exp_name, controller_path, config_dir, analysis=False, cns=False, pns=False):
    # importing data
    if cns:
        os.chdir(cns)
    trodes_data = import_trodes_data(trodes_dir, exp_name, win_dir=True)

    if pns:
        os.chdir(pns)
    config_data = import_config_data(config_dir)
    # import config differently?
    # can analyze per each slice
    controller_data = import_controller_data(controller_path)
    # analysis
    if analysis:
        time = trodes_data['time']
        true_time = match_times(controller_data, trodes_data)
        reach_indices = get_reach_indices(controller_data)
        successful_trials = get_successful_trials(controller_data, true_time, trodes_data)
        exposures = get_exposure_times(trodes_data['DIO']['top_cam'])
        reach_masks = get_reach_times(true_time, reach_indices)
        reach_masks_start = np.asarray(reach_masks['start'])
        reach_masks_stop = np.asarray(reach_masks['stop'])
        reach_indices_start = reach_indices['start']
        reach_indices_stop = reach_indices['stop']
        trial_masks = trial_mask(true_time, reach_indices_start, reach_indices_stop)
        np.savetxt('reach_masks_start.csv', reach_masks_start, delimiter=',')
        np.savetxt('reach_masks_stop.csv', reach_masks_stop, delimiter=',')
        np.savetxt('succ_trials.csv', np.asarray(successful_trials), delimiter=',')
        np.savetxt('true_time.csv', true_time, delimiter=',')
        np.savetxt('trial_masks.csv', trial_masks, delimiter=',')
        np.savetxt('reach_indices_start.csv', reach_indices_start, delimiter=',')
        np.savetxt('reach_indices_stop.csv', reach_indices_stop, delimiter=',')

        dataframe = to_df(exp_name, config_data, true_time, reach_masks_start, reach_masks_stop, reach_indices_start,
                          reach_indices_stop, successful_trials, trial_masks)
    return dataframe


def name_scrape(file, pns, cns):
    """

    Parameters
    ----------
    file - string of a file name
    pns - string, address of pns folder

    Returns
    -------
    controller_file - string containing address of controller file
    trodes_files - string containing address of trodes files
    config_file - string containing address of config file
    exp_name - string containing experiment name eg 'RMxxYYYYMMDD_time', found through parsing the trodes file
    """
    path_d = file.split('/', [-1])  # get rid of last dir
    path_deleveled = path_d.split('/', [-1])
    config_path = path_deleveled + '/workspaces/'
    controller_path = path_deleveled + '/sensor_data/'
    name = ''
    if '/S' in file:
        sess = file.split('/')
        sess = str(sess[2])  # get 'session' part of the thing
    exp_name = sess + name
    return controller_path, config_path, exp_name


def host_off(save_path):
    Cns = '~/bnelson/CNS/'
    pns = '~/bnelson/PNS_data/'
    # cns is laid out rat/day/session/file_name/localdir (we want to be in localdir)
    # search for all directory paths containing .rec files
    i = 0
    for file in glob.glob('*.rec*'):
        controller_path, config_path, exp_name = name_scrape(file, pns, cns)
        print(exp_name + ' is being added..')
        list_of_df = load_files(file, exp_name, controller_path, config_path, save_path, analysis=True)
        if i == 0:
            main_df = list_of_df
        else:
            main_df = pd.concat([main_df, list_of_df], axis=1)
        i += 1
    print('Finished!!')
    return main_df


def to_df(file_name, config_data, true_time, reach_masks_start, reach_masks_stop, reach_indices_start,
          reach_indices_stop, successful_trials, trial_masks):
    # functions to get specific items from config file
    # functions to fetch rat, date, and session from file name
    rat, date = get_name(file_name)
    c = pd.MultiIndex.from_product([rat], [date], [session], [dim], [true_time], [reach_masks_start],
                                   [reach_masks_stop],
                                   [reach_indices_start], [reach_indices_stop], [successful_trials], [trial_masks],
                                   names=['Rat', 'Date', 'Session', 'trial_dim', 'true_time', 'masks_start',
                                          'masks_stop',
                                          'indices_start', 'indices_stop', 'SF', 'masks'])
    new_df = pd.DataFrame(new_trial, columns=c)
    new_df.keys
    if i == 0:
        trial_dataframe = new_df
    else:
        trial_dataframe = pd.concat([trial_dataframe, new_df], axis=1)
    return trial_dataframe


def get_name(file_name):
    # split file name
    rat = file_name[2:4]
    date = file_name[5:12]

    return rat, date


def trial_mask(matched_times, r_i_start, r_i_stop, s_t):
    new_times = np.zeros(len(matched_times[0]))
    for i, j in zip(range(0, len(r_i_start) - 1), range(0, len(r_i_stop) - 1)):
        ix = int(r_i_start[i])
        jx = int(r_i_stop[i])
        if any(i == s_t):
            new_times[ix:jx] = 2
        else:
            new_times[ix:jx] = 1
    return new_times
