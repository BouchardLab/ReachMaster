"""Script to import trodes, micro-controller, and config data

"""
import pandas as pd

from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, \
    get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data, get_exposure_times,

trodes_dir = ''
exp_name = ''
controller_dir = ''
config_dir = ''
controller_path = ''
save_path = ''
# importing data
trodes_data = import_trodes_data(trodes_dir, exp_name)
config_data = import_config_data(config_dir)
controller_data = import_controller_data(controller_path)

# analysis

time = trodes_data['time']
true_time = match_times(controller_data, trodes_data)
reach_indices = get_reach_indices(controller_data)
successful_trials = get_successful_trials(controller_data, true_time)
exposures = get_exposure_times(trodes_data['DIO']['top_cam'])
reach_masks = get_reach_times(true_time, reach_indices)

edf = pd.DataFrame({'successful_trials': successful_trials, 'exposures': exposures, 'reach_masks': reach_masks,
                    'controller_data': controller_data, 'trodes_time': time, 'true_time': true_time})
edf.to_csv(save_path)
