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


def name_scrape(file):
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

    return controller_file,trodes_files,config_file,exp_name


def to_df():

    return dataframe


def scrape_trodes_df(trodes_dir,controller_dir,config_dir,save_path,):
    os.chdir(trodes_dir)
    for file in glob.glob("*.DIO"):
        c_path, trodes_path, config_path, exp_name = name_scrape(file)
        list_of_df = load_files(trodes_path, exp_name, c_path, config_path, save_path)

    return