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
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data


def load_files(trodes_dir, exp_name, controller_path, config_dir, analysis=False, cns=False, pns=False):
    # importing data
    exp_name = exp_name[2:-1]
    exp_name = exp_name.rsplit('.', 1)[0]
    trodes_dir = trodes_dir.rsplit('/', 1)[0]
    if cns:
        os.chdir(cns)
    trodes_data = import_trodes_data(trodes_dir, exp_name, win_dir=False)

    if pns:
        os.chdir(pns)

    config_data = import_config_data(config_dir)
    # import config differently?
    # can analyze per each slice
    controller_data = import_controller_data(controller_path)
    # analysis
    if analysis:
        true_time = match_times(controller_data, trodes_data)
        reach_indices = get_reach_indices(controller_data)
        successful_trials = get_successful_trials(controller_data, true_time, trodes_data)
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


def name_scrape(file):
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
    # controller_data
    name = file.split('/')
    path_d = file.rsplit('/', 2)[0]
    path_d = file.replace('/CNS', '/PNS')
    path_d = file.rsplit('/RM', 2)[0]
    config_path = path_d + '/workspaces/'
    controller_path = path_d + '/sensor_data/'
    # trodes_data
    n = file.rsplit('/', 1)[1]
    if '/S' in file:
        sess = file.rsplit('/S')
        sess = str(sess[1])  # get 'session' part of the namestring
        ix = 'S' + sess[0]
    exp_name = str(ix) + n
    return controller_path, config_path, exp_name, n


def host_off(save_path=False):
    cns_pattern = '/home/kallanved/Desktop/P/CNS/**/*.rec'
    pns = '/home/kallanved/Desktop/P/PNS_data/'
    cns = '/home/kallanved/Desktop/P/CNS'
    #cns = '~/bnelson/CNS/'
    #pns = '~/bnelson/PNS_data/'
    # cns is laid out rat/day/session/file_name/localdir (we want to be in localdir)
    # search for all directory paths containing .rec files
    main_df = pd.DataFrame()
    for file in glob.glob(cns_pattern, recursive=True):
        controller_path, config_path, exp_name, trodes_name = name_scrape(file)
        print(exp_name + ' is being added..')
        list_of_df = load_files(file, exp_name, controller_path, config_path, analysis=True, cns=cns, pns=pns)
        main_df = pd.concat([main_df, list_of_df], axis=1)
    print('Finished!!')
    if save_path:
        main_df.to_csv(save_path)

    return main_df


def get_config_data(config_data):
    config_dataframe = ''
    return config_dataframe


def to_df(file_name, config_data, true_time, reach_masks_start, reach_masks_stop, reach_indices_start,
          reach_indices_stop, successful_trials, trial_masks, trial_dataframe):
    session = file_name[0:2]
    # functions to get specific items from config file
    config_dataframe = get_config_data(config_data)
    rat, date = get_name(file_name)
    dim = config_dataframe[0]
    robot_coordinates = config_dataframe[1]
    c = pd.MultiIndex.from_product([rat], [date], [session], [dim], [true_time], [reach_masks_start],
                                   [reach_masks_stop],
                                   [reach_indices_start], [reach_indices_stop], [successful_trials], [trial_masks],
                                   [robot_coordinates],
                                   names=['Rat', 'Date', 'Session', 'trial_dim', 'true_time', 'masks_start',
                                          'masks_stop',
                                          'indices_start', 'indices_stop', 'SF', 'masks', 'robot coordinates'])
    new_df = pd.DataFrame(columns=c)
    return new_df


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
