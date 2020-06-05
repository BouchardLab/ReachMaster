"""Script to import trodes, micro-controller, and config data

"""
import numpy as np

from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials, make_trial_masks
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data, get_exposure_times


def load_files(analysis=False):
    trodes_dir = 'C:\\Users\\bassp\\PycharmProjects\\ReachMaster\\software\\preprocessing\\trodes_data'
    exp_name = 'RM1520190927_144153'
    controller_path = 'C:\\Users\\bassp\\OneDrive\\Desktop\\Project\\RM15\\927\\S3\\Controller_Data'
    config_dir = 'C:\\Users\\bassp\\PycharmProjects\\ReachMaster\\software\\preprocessing\\Scripts'
    save_path = 'C:\\Users\\bassp\\PycharmProjects\\ReachMaster\\software\\preprocessing\\Scripts'
    # importing data
    trodes_data = import_trodes_data(trodes_dir, exp_name, win_dir=True)
    config_data = import_config_data(config_dir)
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
        # changes: masks must be in exp time as binary variables to export
        # 0 for fail, 1 for success
        # should also extract the handle positions for coordinate xforms

        # save as np objects
        np.savetxt('reach_masks_start.csv', reach_masks_start, delimiter=',')
        np.savetxt('reach_masks_stop.csv', reach_masks_stop, delimiter=',')
        np.savetxt('succ_trials.csv', np.asarray(successful_trials), delimiter=',')
        np.savetxt('true_time.csv', true_time, delimiter=',')
        np.savetxt('trial_masks.csv', trial_masks, delimiter=',')
        np.savetxt('reach_indices.csv', reach_indices, delimiter=',')
    return
