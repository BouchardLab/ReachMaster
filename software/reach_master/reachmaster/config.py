"""Module for working with configuration files.

Configurations files are json files that store all the settings 
used in a ReachMaster session. Functions are provided to
generate default configuration files, as well as to save and 
load previous configurations. This encourages persistence of 
settings across experiments. 

Todo:
    * Automate unit tests 
    * Integrate ffmpeg settings
    * Flesh out field descriptions

"""

import os
import numpy as np
import json
import datetime


def default_config():
    """Generate a configuration populated with sensible defaults.

  Returns
  -------
  dict
    A default configuration.      

  """
    config = {
        'ReachMaster': {
            'data_dir': os.getcwd(),
            'config_file': 'Default',
            'exp_control_port': '/dev/ttyACM0',
            'rob_control_port': '/dev/ttyACM1',
            'serial_baud': 2000000,
            'control_timeout': 5
        },
        'CameraSettings': {
            'num_cams': 3,
            'imgdataformat': 'XI_RAW8',
            'fps': 200,
            'exposure': 2000,
            'gain': 15.0,
            'sensor_feature_value': 1,
            'gpi_selector': 'XI_GPI_PORT1',
            'gpi_mode': 'XI_GPI_TRIGGER',
            'trigger_source': 'XI_TRG_EDGE_RISING',
            'gpo_selector': 'XI_GPO_PORT1',
            'gpo_mode': 'XI_GPO_EXPOSURE_ACTIVE',
            'img_width': 688,
            'img_height': 688,
            'offset_x': 304,
            'offset_y': 168,
            'downsampling': 'XI_DWN_1x1',
            'saved_pois': [],
            'poi_threshold': 15,
        },
        'ExperimentSettings': {
            'baseline_dur': 5.0,
            'reach_timeout': 4000,
            'flush_dur': 10000,
            'solenoid_open_dur': 75,
            'solenoid_bounce_dur': 500,
            'reward_win_dur': 3000,
            'max_rewards': 3,
            'lights_off_dur': 3000,
            'lights_on_dur': 5000,
            'reach_delay': 100
        },
        'RobotSettings': {
            'calibration_file': None,
            'dis': None,
            'pos': None,
            'x_push_dur': None,
            'x_pull_dur': None,
            'y_push_dur': None,
            'y_pull_dur': None,
            'z_push_dur': None,
            'z_pull_dur': None,
            'command_type': None,
            'command_set': None,
            'command_file': None,
            'x_command_pos': None,
            'y_command_pos': None,
            'z_command_pos': None,
            'r_command_pos': None,
            'theta_y_command_pos': None,
            'theta_z_command_pos': None,
            'reach_dist_min': 10,
            'reach_dist_max': 40,
            'reach_angle_max': np.pi / 3,
            'ygimbal_to_joint': 64,
            'zgimbal_to_joint': 47,
            'xgimbal_xoffset': 168,
            'ygimbal_yoffset': 100,
            'zgimbal_zoffset': 117,
            'x_origin': 1024,
            'y_origin': 608,
            'z_origin': 531,
            'pos_smoothing': 0.1,
            'tol': np.round(1023.0 / 50.0 / 3.0, decimals=1),
            'period': 125.0 * 1000.0,
            'off_dur': 1000,
            'num_tol': 5,
            'x_push_wt': 1.0,
            'x_pull_wt': 1.0,
            'y_push_wt': 1.0,
            'y_pull_wt': 1.0,
            'z_push_wt': 1.0,
            'z_pull_wt': 1.0,
            'rew_zone_x': 1000,
            'rew_zone_y_min': 558,
            'rew_zone_y_max': 658,
            'rew_zone_z_min': 481,
            'rew_zone_z_max': 581
        },
        'Protocol': {
            'type': 'TRIALS'
        }
    }
    return config


def load_config(file_handle):
    """Load a configuration file.

    Typically loads a file selected by user from the 
    configuration browser in the root ReachMaster 
    application window.

    Parameters
    ----------
    file_handle : file
        An open, previously saved json configuration file.

    Returns
    -------
    dict
      A default configuration.       

    """
    return json.load(open(file_handle))


def save_config(config):
    """Save a configuration file.

    This is typically called upon closing application if the
    user responds yes to the save prompt. The configuration is 
    given the current datetime as a name and saved to a json 
    in the config folder of the user-selected data output 
    directory.

    Parameters
    ----------
    config : dict
        The currently loaded configuration.      

    """
    config_path = config['ReachMaster']['data_dir'] + "/config/"
    if not os.path.isdir(config_path):
        os.makedirs(config_path)
    fn = config_path + 'config: ' + str(datetime.datetime.now()) + '.json'
    with open(fn, 'w') as outfile:
        json.dump(config, outfile, indent=4)
    return fn


def save_tmp(config):
    """Save the configuration to a temp file.

    The global state of ReachMaster is tracked using a temp
    file. Whenever a child window of the main application is
    created or destroyed, the temp file is updated to reflect
    whatever changes in settings may have occurred. If the 
    user forgets to save the configuration file at the end of
    a session, this temp file can be used for recovery. It is 
    saved to the temp folder in the ReachMaster root directory.
    **Warning: the temp file is reset to the default 
    configuration whenever a new instance of ReachMaster is 
    started!**

    Parameters
    ----------
    config : dict
        The currently loaded configuration.      

    """
    configPath = "./temp/"
    if not os.path.isdir(configPath):
        os.makedirs(configPath)
    fn = configPath + 'tmp_config.json'
    with open(fn, 'w') as outfile:
        json.dump(config, outfile, indent=4)
    return fn
