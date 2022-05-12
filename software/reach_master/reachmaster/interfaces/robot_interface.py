"""This module provides a set of functions for interfacing
with the robot microcontroller. It provides functions to 
recognize and establish serial connections to the 
controller, read and write settings, and to load 
calibration and command files. Soon it will provide 
methods to load user-selected scripts to/from the
controller in order to execute various robot calibration 
routines. 

Todo:
    * Object orient
    * Automate unit tests
    * Ability to load user-selected controller scipts

"""

import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
import pdb
import ast
import json
import time


# private functions----------------------------------------------

def _load_calibration_variable(rob_controller, varname, value):
    rob_controller.write(b"c")
    if rob_controller.read() == b"c":
        rob_controller.write((varname + "\n").encode('utf-8'))
        if rob_controller.read() == b"c":
            rob_controller.write(str(value).encode('utf-8'))
    else:
        rob_controller.write(b"c")
        pdb.set_trace()
    if rob_controller.read() == b"c":
        print((varname + ' loaded'))
    else:
        print(varname + ' calibration failed')
        raise Exception(varname)


def _load_command_variable(rob_controller, varname, value):
    rob_controller.write(b"p")
    if rob_controller.read() == b"p":
        rob_controller.write((varname + "\n").encode('utf-8'))
        if rob_controller.read() == b"p":
            rob_controller.write(str(value).encode('utf-8'))
    if rob_controller.read() == b"p":
        print((varname + ' loaded'))
    else:
        print(varname + 'load failed')
        raise Exception(varname)


def _variable_read(rob_controller, varname):
    rob_controller.write(b"g")
    if rob_controller.read() == b"g":
        rob_controller.write((varname + "\n").encode('utf-8'))
        return rob_controller.readline().decode('utf-8')


def _variable_write(rob_controller, varname, value):
    rob_controller.write(b"v")
    if rob_controller.read() == b"v":
        rob_controller.write(bytes(varname + "\n", 'utf-8'))
        if rob_controller.read() == b"v":
            rob_controller.write(bytes(value + "\n", 'utf-8'))


def read_calibration_file(calibration_file):
    calibration_dict = pd.read_csv(calibration_file)  # Read in Json
    return calibration_dict


def set_calibration_config(config):
    cal_file = read_calibration_file(config['RobotSettings']['calibration_file'])
    config['RobotSettings']['dis'] = np.asarray(cal_file['displacement'])
    config['RobotSettings']['pos'] = np.asarray(cal_file['position'])
    config['RobotSettings']['x_push_dur'] = np.asarray(cal_file['xPushDuration'])
    config['RobotSettings']['x_pull_dur'] = np.asarray(cal_file['xPullDuration'])
    config['RobotSettings']['y_push_dur'] = np.asarray(cal_file['yPushDuration'])
    config['RobotSettings']['y_pull_dur'] = np.asarray(cal_file['yPullDuration'])
    config['RobotSettings']['z_push_dur'] = np.asarray(cal_file['zPushDuration'])
    config['RobotSettings']['z_pull_dur'] = np.asarray(cal_file['zPullDuration'])
    return


# public functions---------------------------------------------------

def start_interface(config):
    """Establish a serial connection with the robot
    controller.

    Parameters
    ----------
    config : dict
        The currently loaded configuration file.

    Returns
    -------
    rob_controller : serial.serialposix.Serial  
        The serial interface to the robot controller.      

    """
    rob_controller = serial.Serial(config['ReachMaster']['rob_control_port'],
                                   config['ReachMaster']['serial_baud'],
                                   timeout=config['ReachMaster']['control_timeout'])
    rob_controller.flushInput()
    rob_controller.write(b"h")
    response = rob_controller.read()
    if response == b"h":
        rob_connected = True
        return rob_controller
    else:
        raise Exception("Robot controller failed to connect.")


def stop_interface(rob_controller):
    """Perform a soft reboot of the robot controller
    and close the serial connection.

    Parameters
    ----------
    rob_controller : serial.serialposix.Serial
        The serial interface to the robot controller.

    """
    rob_controller.write(b"e")
    rob_controller.close()


def get_ports():
    """List all serial port with connected devices.

    Returns
    -------
    port_list : list(str)  
        Names of all the serial ports with connected devices. 

    """
    port_list = list(list_ports.comports())
    for i in range(len(port_list)):
        port_list[i] = port_list[i].device
    return port_list


def load_config_calibration(rob_controller, config):
    """Load the calibration parameters to the robot controller.

    Uses the calibration file selected in the configuration
    file.

    Parameters
    ----------
    rob_controller : serial.serialposix.Serial
        The serial interface to the robot controller.
    config : dict
        The currently loaded configuration file.     

    """
    # UD 1/2022. Force-fetch calibration items from calibration file before writing.
    try:
        _load_calibration_variable(rob_controller, 'dis', config['RobotSettings']['dis'])
        _load_calibration_variable(rob_controller, 'pos', config['RobotSettings']['pos'])
        _load_calibration_variable(rob_controller, 'x_push_dur', config['RobotSettings']['x_push_dur'])
        _load_calibration_variable(rob_controller, 'x_pull_dur', config['RobotSettings']['x_pull_dur'])
        _load_calibration_variable(rob_controller, 'y_push_dur', config['RobotSettings']['y_push_dur'])
        _load_calibration_variable(rob_controller, 'y_pull_dur', config['RobotSettings']['y_pull_dur'])
        _load_calibration_variable(rob_controller, 'z_push_dur', config['RobotSettings']['z_push_dur'])
        _load_calibration_variable(rob_controller, 'z_pull_dur', config['RobotSettings']['z_pull_dur'])
    except:
        pdb.set_trace()


def load_config_commands(rob_controller, config):
    """Load the position commands to the robot controller.
    
    Uses the method determined by the command type 
    selected in the configuration file. Currently, the
    options are read_from_file, sample_from_file, or 
    parametric_sample. The read_from_file option takes 
    the commands directly from the command file. The 
    sample_from_file option generates a sequence of 
    commands by sampling rows from the command file with 
    replacement. For both of these options, the command 
    file is assumed to have three columns ordered as 
    reach distance, azimuth, elevation. The 
    parametric_sample option does not use the command 
    file. Rather, it samples commands uniformly from the 
    reach volume determined by the user-selected inverse 
    kinematics parameters. 

    Todo:
        * Further functionalize for better clarity.

    Parameters
    ----------
    rob_controller : serial.serialposix.Serial
        The serial interface to the robot controller.
    config : dict
        The currently loaded configuration file. 

    Returns
    -------
    config : dict
        The configuration file possibly updated to include
        the command values that were loaded.    

    """
    # extract robot kinematic settings
    ygimbal_to_joint = config['RobotSettings']['ygimbal_to_joint']
    zgimbal_to_joint = config['RobotSettings']['zgimbal_to_joint']
    xgimbal_xoffset = config['RobotSettings']['xgimbal_xoffset']
    ygimbal_yoffset = config['RobotSettings']['ygimbal_yoffset']
    zgimbal_zoffset = config['RobotSettings']['zgimbal_zoffset']
    x_origin = config['RobotSettings']['x_origin']
    y_origin = config['RobotSettings']['y_origin']
    z_origin = config['RobotSettings']['z_origin']
    reach_dist_min = config['RobotSettings']['reach_dist_min']
    reach_dist_max = config['RobotSettings']['reach_dist_max']
    reach_angle_max = config['RobotSettings']['reach_angle_max']
    # generate commands according to selected command type
    n = 100
    if config['RobotSettings']['command_type'] == "parametric_sample":
        r = (
                reach_dist_min +
                (reach_dist_max - reach_dist_min) *
                np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=500 * n
                ) ** (1.0 / 3.0)
        )
        theta_y = reach_angle_max * np.random.uniform(low=-1, high=1, size=500 * n)
        theta_z = reach_angle_max * np.random.uniform(low=-1, high=1, size=500 * n)
        theta = np.sqrt(theta_y ** 2 + theta_z ** 2)
        r = r[theta <= reach_angle_max][0:n]
        theta_y = theta_y[theta <= reach_angle_max][0:n]
        theta_z = theta_z[theta <= reach_angle_max][0:n]
    elif config['RobotSettings']['command_type'] == "sample_from_file":
        r_set, theta_y_set, theta_z_set = np.loadtxt(
            config['RobotSettings']['command_file'],
            skiprows=1,
            delimiter=',',
            unpack=True,
            usecols=(1, 2, 3)
        )
        rand_sample = np.random.choice(list(range(len(r_set))), replace=True, size=n)
        r = r_set[rand_sample]
        theta_y = theta_y_set[rand_sample]
        theta_z = theta_z_set[rand_sample]
    elif config['RobotSettings']['command_type'] == "read_from_file":
        r, theta_y, theta_z = np.loadtxt(
            config['RobotSettings']['command_file'],
            skiprows=1,
            delimiter=',',
            unpack=True,
            usecols=(1, 2, 3)
        )
    else:
        raise Exception("Invalid command type.")
        # pass generated commands though inverse kinematic transformation
    Ax = np.sqrt(
        xgimbal_xoffset ** 2 + r ** 2 - 2 * xgimbal_xoffset * r * np.cos(theta_y) * np.cos(theta_z)
    )
    gammay = -np.arcsin(
        np.sin(theta_y) *
        np.sqrt(
            (r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        ) /
        np.sqrt(
            (xgimbal_xoffset - r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        )
    )
    gammaz = -np.arcsin(r * np.sin(theta_z) / Ax)
    Ay = np.sqrt(
        (ygimbal_to_joint - ygimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_yoffset - ygimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Az = np.sqrt(
        (zgimbal_to_joint - zgimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_zoffset - zgimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Ax = np.round((Ax - xgimbal_xoffset) / 50 * 1024 + x_origin, decimals=1)
    Ay = np.round((Ay - ygimbal_yoffset) / 50 * 1024 + y_origin, decimals=1)
    Az = np.round((Az - zgimbal_zoffset) / 50 * 1024 + z_origin, decimals=1)
    # convert tranformed commands to appropriate data types/format
    x = np.array2string(Ax, formatter={'float_kind': lambda Ax: "%.1f" % Ax})
    y = np.array2string(Ay, formatter={'float_kind': lambda Ay: "%.1f" % Ay})
    z = np.array2string(Az, formatter={'float_kind': lambda Az: "%.1f" % Az})
    r = np.array2string(r, formatter={'float_kind': lambda r: "%.1f" % r})
    theta_y = np.array2string(theta_y, formatter={'float_kind': lambda theta_y: "%.1f" % theta_y})
    theta_z = np.array2string(theta_z, formatter={'float_kind': lambda theta_z: "%.1f" % theta_z})
    x = x[1:-1] + ' '
    y = y[1:-1] + ' '
    z = z[1:-1] + ' '
    r = r[1:-1] + ' '
    theta_y = theta_y[1:-1] + ' '
    theta_z = theta_z[1:-1] + ' '
    # load commands to robot
    try:
        _load_command_variable(rob_controller, 'x_command_pos', x)
        _load_command_variable(rob_controller, 'y_command_pos', y)
        _load_command_variable(rob_controller, 'z_command_pos', z)
    except Exception as varname:
        raise Exception("Failed to load: " + varname)
    # record the loaded commands to the config
    config['RobotSettings']['x'] = x
    config['RobotSettings']['y'] = y
    config['RobotSettings']['z'] = z
    config['RobotSettings']['r'] = r
    config['RobotSettings']['theta_y'] = theta_y
    config['RobotSettings']['theta_z'] = theta_z
    return config


def set_rob_controller(rob_controller, config):
    """Load all the robot settings to the robot controller.

    Parameters
    ----------
    rob_controller : serial.serialposix.Serial
        The serial interface to the robot controller.
    config : dict
        The currently loaded configuration file.

    Returns
    -------
    config : dict
        The configuration file possibly updated to include
        the command values that were loaded.

    """
    set_calibration_config(config)  # Make sure current calibration settings are uploaded.
    config = load_config_commands(rob_controller, config)
    load_config_calibration(rob_controller, config)
    _variable_write(rob_controller, 'pos_smoothing', str(config['RobotSettings']['pos_smoothing']))
    _variable_write(rob_controller, 'tol', str(config['RobotSettings']['tol']))
    _variable_write(rob_controller, 'period', str(config['RobotSettings']['period']))
    _variable_write(rob_controller, 'off_dur', str(config['RobotSettings']['off_dur']))
    _variable_write(rob_controller, 'num_tol', str(config['RobotSettings']['num_tol']))
    _variable_write(rob_controller, 'x_push_wt', str(config['RobotSettings']['x_push_wt']))
    _variable_write(rob_controller, 'x_pull_wt', str(config['RobotSettings']['x_pull_wt']))
    _variable_write(rob_controller, 'y_push_wt', str(config['RobotSettings']['y_push_wt']))
    _variable_write(rob_controller, 'y_pull_wt', str(config['RobotSettings']['y_pull_wt']))
    _variable_write(rob_controller, 'z_push_wt', str(config['RobotSettings']['z_push_wt']))
    _variable_write(rob_controller, 'z_pull_wt', str(config['RobotSettings']['z_pull_wt']))
    _variable_write(rob_controller, 'rew_zone_x', str(config['RobotSettings']['rew_zone_x']))
    _variable_write(rob_controller, 'rew_zone_y_min', str(config['RobotSettings']['rew_zone_y_min']))
    _variable_write(rob_controller, 'rew_zone_y_max', str(config['RobotSettings']['rew_zone_y_max']))
    _variable_write(rob_controller, 'rew_zone_z_min', str(config['RobotSettings']['rew_zone_z_min']))
    _variable_write(rob_controller, 'rew_zone_z_max', str(config['RobotSettings']['rew_zone_z_max']))
    return config
