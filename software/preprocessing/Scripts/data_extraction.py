"""Script to import trodes, micro-controller, and config data

"""
import glob
import os

import numpy as np
import pandas as pd

from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials, make_trial_masks
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data


def load_files(trodes_dir, exp_name, controller_path, config_dir, save_path, analysis=False, scrape=False):
    # importing data
    trodes_data = import_trodes_data(trodes_dir, exp_name, win_dir=True)
    config_data = import_config_data(config_dir)
    controller_data = import_controller_data(controller_path)
    # analysis
    if analysis:
        true_time = match_times(controller_data, trodes_data)
        reach_indices = get_reach_indices(controller_data)
        successful_trials = get_successful_trials(controller_data, true_time, trodes_data)
        reach_masks = get_reach_times(true_time, reach_indices)
        reach_masks_start = np.asarray(reach_masks['start'])
        reach_masks_stop = np.asarray(reach_masks['stop'])
        trial_masks = make_trial_masks(controller_data, trodes_data)
        reach_indices_start = reach_indices['start']
        reach_indices_stop = reach_indices['stop']

        # changes: masks must be in exp time as binary variables to export
        # 0 for fail, 1 for success
        # should also extract the handle positions for coordinate xforms

        # save as np objects
        np.savetxt('reach_masks_start.csv', reach_masks_start, delimiter=',')
        np.savetxt('reach_masks_stop.csv', reach_masks_stop, delimiter=',')
        np.savetxt('succ_trials.csv', np.asarray(successful_trials), delimiter=',')
        np.savetxt('true_time.csv', true_time, delimiter=',')
        np.savetxt('trial_masks.csv', trial_masks, delimiter=',')
        np.savetxt('reach_indices_start.csv', reach_indices_start, delimiter=',')
        np.savetxt('reach_indices_stop.csv', reach_indices_stop, delimiter=',')
        dataframe = to_df()
    return dataframe


def name_scrape(file, controller_dir, config_dir):
    """

    Parameters
    ----------
    file - string of a file name


    Returns
    -------
    controller_file - string containing address of controller file
    trodes_files - string containing address of trodes files
    config_file - string containing address of config file
    exp_name - string containing experiment name eg 'RMxxYYYYMMDD_time', found through parsing the trodes file
    """
    controller_file = ""
    trodes_files = ""
    config_file = ""
    exp_name = ""
    return controller_file, trodes_files, config_file, exp_name


def make_big_df(times, bodyparts, start, stop, mask, scorer, date, session, trial, dim, path=False, dataframe=False):
    for i, j in enumerate(bodyparts):
        # list_of_df = to_df(times, t, start, stop, mask, part_name, scorer, rat, date, session, dim)
        list_of_df = 0
        if i == 0:
            main_df = list_of_df
        else:
            main_df = pd.concat([main_df, list_of_df], axis=1)
    print('Finished!')
    return main_df


def to_df(times, part_file, trial_mask_start, trial_mask_stop, mask, part, scorer, rat, date, session, dim):
    for i in range(0, len(trial_mask_start) - 1):
        i_x = int(trial_mask_start[i])
        j_x = int(trial_mask_stop[i])
        val = mask[i_x]
        if val == 1:
            trial = 'Fail'
        if val == 2:
            trial = 'Success'
        trial_array = part_file[i_x:j_x]
        times = np.asarray(times)
        time_start = (times[i_x])
        time_stop = (times[j_x])
        total = float(time_stop - time_start)
        time_list = [time_start, time_stop, total]
        new_trial = np.asarray(trial_array)
        c = pd.MultiIndex.from_product([[scorer], [rat], [date], [session], [dim], [i], [part], [trial],
                                        [total], ['x', 'y', 'z']],
                                       names=['Scorer', 'Rat', 'Date', 'Session', 'trial_dim', 'trials', 'Bodypart',
                                              'S/F', 'exp_times', 'coords'])
        new_df = pd.DataFrame(new_trial, columns=c)
        new_df.keys
        if i == 0:
            trial_dataframe = new_df
        else:
            trial_dataframe = pd.concat([trial_dataframe, new_df], axis=1)
    return trial_dataframe


def scrape_trodes_df(trodes_dir, controller_dir, config_dir, save_path):
    os.chdir(trodes_dir)
    for file in glob.glob("*.DIO"):
        c_path, trodes_path, config_path, exp_name = name_scrape(file, controller_dir, config_dir)
        list_of_df = load_files(trodes_path, exp_name, c_path, config_path, save_path)

    return
