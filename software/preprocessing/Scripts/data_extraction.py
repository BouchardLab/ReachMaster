"""Script to import trodes, micro-controller, and config data

"""
import numpy as np
import glob
import os
from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials, make_trial_masks
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data, get_exposure_times



def load_files(trodes_dir,exp_name,controller_path,config_dir,save_path, analysis=False,scrape=False):
    # importing data
    trodes_data = import_trodes_data(trodes_dir, exp_name, win_dir=True)
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
        trial_masks = make_trial_masks(controller_data, trodes_data)
        reach_indices_start = reach_indices['start']
        reach_indices_stop = reach_indices['stop']

        np.savetxt('reach_masks_start.csv', reach_masks_start, delimiter=',')
        np.savetxt('reach_masks_stop.csv', reach_masks_stop, delimiter=',')
        np.savetxt('succ_trials.csv', np.asarray(successful_trials), delimiter=',')
        np.savetxt('true_time.csv', true_time, delimiter=',')
        np.savetxt('trial_masks.csv', trial_masks, delimiter=',')
        np.savetxt('reach_indices_start.csv', reach_indices_start, delimiter=',')
        np.savetxt('reach_indices_stop.csv', reach_indices_stop, delimiter=',')
        dataframe = to_df(true_time,reach_masks_start,reach_masks_stop,reach_indices_start,reach_indices_stop,successful_trials)
    return dataframe


def name_scrape(file,pns):
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
    #config, controller data use the 11 characters before extensions name
    # import cns data and pns path as string
    path = str(file)  # or something like that, check
    # scrape name from current dir!
    exp_name = ''
    os.chdir(path)
    # pass to df function
    # scrape file path to get pns file path
    os.chdir(pns)
    path_deleveled = str(file)  # get rid of last dir (RM...)

    os.chdir(path_deleveled)
    config_path = path_deleveled + 'workspaces/'
    controller_path = path_deleveled + 'sensor_data/'
    return controller_path, config_path, exp_name

def host_off(save_path):
    Cns = '~/bnelson/CNS/'
    pns = '~/bnelson/PNS_data/'
    # cns is laid out rat/day/session/file_name/localdir (we want to be in localdir)
    # search for all directory paths containing .rec files
    os.chdir(Cns)
    for file in glob.glob('*.rec*'):
        controller_path,config_path,exp_name = name_scrape(file,pns)
        df = load_files(file, exp_name, controller_path, config_path, save_path, analysis=True)

    return df


def to_df():
    df=''
    return df