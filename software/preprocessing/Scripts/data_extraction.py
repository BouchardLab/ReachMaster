"""Script to import trodes, micro-controller, and config data

"""
import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import csv
from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data


def load_files(trodes_dir, exp_name, controller_path, config_dir, rat, session, analysis=False, cns=False, pns=False, save_path=False):
    # importing data
    exp_names = exp_name[2:-1]
    exp_names = exp_names.rsplit('.', 1)[0]
    trodes_dir = trodes_dir.rsplit('/', 1)[0]
    if cns:
        os.chdir(cns)
    trodes_data = import_trodes_data(trodes_dir, exp_names, win_dir=False)

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
        trial_masks = trial_mask(true_time, reach_indices_start, reach_indices_stop,successful_trials)
    if save_path:
        os.chdir(save_path)
        np.savetxt('reach_masks_start.csv', reach_masks_start, delimiter=',')
        np.savetxt('reach_masks_stop.csv', reach_masks_stop, delimiter=',')
        np.savetxt('succ_trials.csv', np.asarray(successful_trials), delimiter=',')
        np.savetxt('true_time.csv', true_time, delimiter=',')
        np.savetxt('trial_masks.csv', trial_masks, delimiter=',')
        np.savetxt('reach_indices_start.csv', reach_indices_start, delimiter=',')
        np.savetxt('reach_indices_stop.csv', reach_indices_stop, delimiter=',')

    dataframe = to_df(exp_names, config_data, true_time, reach_masks_start, reach_masks_stop, reach_indices_start,
                          reach_indices_stop, successful_trials, trial_masks, rat, session)
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
    name = file.split('/')[6]
    path_d = file.rsplit('/', 2)[0]
    path_d = file.replace('/CNS', '/PNS_data')
    path_d = path_d.rsplit('/R', 2)[0]

    config_path = path_d + '/workspaces'
    controller_path = path_d + '/sensor_data'
    # trodes_data
    n = file.rsplit('/', 1)[1]
    if '/S' in file:
        sess = file.rsplit('/S')
        sess = str(sess[1])  # get 'session' part of the namestring
        ix = 'S' + sess[0]
    exp_name = str(ix) + n
    return controller_path, config_path, exp_name,name,ix, n


def host_off(save_path=False):
    cns_pattern = '/home/kallanved/Desktop/P/CNS/**/*.rec'
    pns = '/home/kallanved/Desktop/P/PNS_data/'
    cns = '/home/kallanved/Desktop/P/CNS'
    #cns = '~/bnelson/CNS/'
    #pns = '~/bnelson/PNS_data/'
    # cns is laid out rat/day/session/file_name/localdir (we want to be in localdir)
    # search for all directory paths containing .rec files
    main_df = pd.DataFrame()
    i = 0
    for file in glob.glob(cns_pattern, recursive=True):
        controller_path, config_path, exp_name, name, ix, trodes_name = name_scrape(file)
        print(exp_name + ' is being added..')
        list_of_df = load_files(file, exp_name, controller_path, config_path, name, ix, analysis=True, cns=cns, pns=pns)
        if i == 0:
            d = list_of_df
            i += 1
        else:
            for k, v in list_of_df.items():
                if (k in d):
                    d[k].update(list_of_df[k])
                else:
                    d[k] = list_of_df[k]
    print('Finished!!')
    if save_path:
        with open(save_path, 'wb') as f:
            w = csv.DictWriter(f, d.keys())
            w.writeheader()
            w.writerow(d)

    return list_of_df


def get_config_data(config_data):
    exp_type = config_data['RobotSettings']['commandFile']
    reward_dur = config_data['ExperimentSettings']['rewardWinDur']
    robot_config = config_data['RobotSettings']

    return exp_type, reward_dur, robot_config


def to_df(file_name, config_data, true_time, reach_masks_start, reach_masks_stop, reach_indices_start,
          reach_indices_stop, successful_trials, trial_masks, rat, session):
    # functions to get specific items from config file
    dim, reward_dur, robot_config = get_config_data(config_data)
    date = get_name(file_name)
    successful_trials = np.asarray(successful_trials)
    dict = make_dict()
    dict[rat][date][session][dim]['time'] = true_time
    dict[rat][date][session][dim]['masks_start'] = reach_masks_start
    dict[rat][date][session][dim]['mask_stop'] = reach_masks_stop
    dict[rat][date][session][dim]['SF'] = successful_trials
    dict[rat][date][session][dim]['masks'] = trial_masks
    dict[rat][date][session][dim]['dur'] = reward_dur
    return dict


def make_dict():
    return defaultdict(make_dict)


def get_name(file_name):
    # split file name
    date = file_name[5:12]
    return date


def trial_mask(matched_times, r_i_start, r_i_stop, s_t):
    lenx = int(matched_times.shape[0])
    new_times = np.zeros((lenx))
    for i, j in zip(range(0, len(r_i_start) - 1), range(0, len(r_i_stop) - 1)):
        ix = int(r_i_start[i])
        jx = int(r_i_stop[i])
        if any(i == s for s in s_t):
            new_times[ix:jx] = 2
            print(i)
        else:
            new_times[ix:jx] = 1
    return new_times
