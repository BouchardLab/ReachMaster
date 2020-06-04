"""Script to import trodes, micro-controller, and config data

"""
import pandas as pd

from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data, get_exposure_times


def load_files(analysis=False):
    trodes_dir = 'C:\\Users\\bassp\\PycharmProjects\\ReachMaster\\software\\preprocessing\\trodes_data'
    exp_name = 'RM1520190927_144153'
    controller_path = 'C:\\Users\\bassp\\PycharmProjects\\ReachMaster\\software\\preprocessing\\Scripts'
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
        r_m = pd.Series({'reach_masks': reach_masks})
        e = pd.Series({'exposures': exposures})
        r_i = pd.Series({'ri': reach_indices})
        s_t = pd.Series({'s_t': successful_trials})
        r_m.to_csv('r_m.csv')
        e.to_csv('e.csv')
        r_i.to_csv('r_i.csv')
        s_t.to_csv('s_t.csv')

    return successful_trials
