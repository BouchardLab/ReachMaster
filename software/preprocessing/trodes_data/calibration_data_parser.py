"""A library for reading, parsing, and preprocessing trodes data 
collected during robot calibration routines.

"""

import os
from . import readTrodesExtractedDataFile3 as read_trodes
import numpy as np
import pandas as pd
import subprocess as sp 

def get_trodes_files(data_dir, trodes_name):
    """Generate names of all the trodes files from a calibration recording.
    Assumes data is saved in the default trodes filesystem and channels are
    named appropriately in the trodes configuration file. 

    Parameters
    ----------
    data_dir : str
        Parent directory where the trodes data lives
    trodes_name : str
        Name of original .rec trodes file

    Returns
    -------
    trodes_files : dict  
        The file names for each channel of a calibration recording. More 
        specifically, `x_push_file` is the *.dat file for the `x` actuator
        `push` valve command recording. Similarly, `y_pot_file` is the 
        *.dat for the `y` actuator poteniometer recording.  
        
    """
    trodes_files = {
        'time_file':  data_dir + '/%s/%s.analog/%s.timestamps.dat' % (trodes_name, trodes_name, trodes_name),
        'x_push_file': data_dir + '/%s/%s.DIO/%s.dio_xPush.dat' % (trodes_name, trodes_name, trodes_name),
        'x_pull_file': data_dir + '/%s/%s.DIO/%s.dio_xPull.dat' % (trodes_name, trodes_name, trodes_name),
        'y_push_file': data_dir + '/%s/%s.DIO/%s.dio_yPush.dat' % (trodes_name, trodes_name, trodes_name),
        'y_pull_file': data_dir + '/%s/%s.DIO/%s.dio_yPull.dat' % (trodes_name, trodes_name, trodes_name),
        'z_push_file': data_dir + '/%s/%s.DIO/%s.dio_zPush.dat' % (trodes_name, trodes_name, trodes_name),
        'z_pull_file': data_dir + '/%s/%s.DIO/%s.dio_zPull.dat' % (trodes_name, trodes_name, trodes_name),
        'x_pot_file': data_dir + '/%s/%s.analog/%s.analog_potX.dat' % (trodes_name, trodes_name, trodes_name),
        'y_pot_file': data_dir + '/%s/%s.analog/%s.analog_potY.dat' % (trodes_name, trodes_name, trodes_name),
        'z_pot_file': data_dir + '/%s/%s.analog/%s.analog_potZ.dat' % (trodes_name, trodes_name, trodes_name)
    }
    return trodes_files

def read_data(trodes_files, sampling_rate = 3000):
    """Read all the trodes file data using the SpikeGadgets 
    `readTrodesExtractedDataFile` script. 

    Parameters
    ----------
    trodes_files : dict
        The file names for each channel of a calibration recording. For
        example, as returned by get_trodes_files(). 
    sampling_rate : int
        Specifying a rate (Hz) lower than the SpikeGadgets MCU clock rate of
        30 kHz will downsample the data to speed up parsing.

    Returns
    -------
    calibration_data : dict  
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording.
        
    """
    clockrate = np.float_(read_trodes.readTrodesExtractedDataFile(trodes_files['time_file'])['clock rate'])
    ds = int(clockrate / sampling_rate)
    calibration_data = {
        'clockrate': clockrate,
        'time': read_trodes.readTrodesExtractedDataFile(trodes_files['time_file'])['data'][0:-1:ds],
        'DIO': {
            'x_push': read_trodes.readTrodesExtractedDataFile(trodes_files['x_push_file'])['data'], 
            'x_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['x_pull_file'])['data'],
            'y_push': read_trodes.readTrodesExtractedDataFile(trodes_files['y_push_file'])['data'], 
            'y_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['y_pull_file'])['data'], 
            'z_push': read_trodes.readTrodesExtractedDataFile(trodes_files['z_push_file'])['data'], 
            'z_pull': read_trodes.readTrodesExtractedDataFile(trodes_files['z_pull_file'])['data']
        },
        'analog':{
            'x_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['x_pot_file'])['data']['voltage'][0:-1:ds],
            'y_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['y_pot_file'])['data']['voltage'][0:-1:ds],
            'z_pot': read_trodes.readTrodesExtractedDataFile(trodes_files['z_pot_file'])['data']['voltage'][0:-1:ds]
        }
    }
    return calibration_data

def to_numpy(calibration_data):
    """Convert the calibration data to numpy arrays 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 

    Returns
    -------
    calibration_data : dict  
        Numpy-converted calibration data.
        
    """
    calibration_data['time'] = np.array(
        [t[0] for t in calibration_data['time']],
        dtype='float_'
        )
    for key in calibration_data['DIO'].keys():
        calibration_data['DIO'][key] = np.array(
            [i[0] for i in calibration_data['DIO'][key]],
            dtype='float_'
            )   
    return calibration_data

def to_seconds(calibration_data, start_at_zero = True):
    """Convert the calibration data time units to seconds. 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 
    start_at_zero : bool
        If True, the start time will be set to 0.

    Returns
    -------
    calibration_data : dict  
        Seconds-converted calibration data
        
    """
    if start_at_zero:
        for key in calibration_data['DIO'].keys():
            calibration_data['DIO'][key] = (
                calibration_data['DIO'][key] - calibration_data['time'][0]
                ) /  calibration_data['clockrate']
        calibration_data['time'] = (
            calibration_data['time'] - calibration_data['time'][0]
            ) / calibration_data['clockrate']
    else:
        for key in calibration_data['DIO'].keys():
            calibration_data['DIO'][key] = calibration_data['DIO'][key] / calibration_data['clockrate']
        calibration_data['time'] = calibration_data['time'] / calibration_data['clockrate']
    return calibration_data

def pots_to_cm(calibration_data, supply_voltage = 3.3, pot_range = 5.0):
    """Convert the potentiometer data units to cm. 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 
    supply_voltage : float
        Maximum voltage for the potentiometers
    pot_range : float
        Potentiometer maximum travel range in cm 

    Returns
    -------
    calibration_data : dict  
        Calibration data with potentiometer data convert to cm
        
    """
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = (
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts / supply_voltage * pot_range
            )
    return calibration_data

def pots_to_volts(calibration_data):
    """Convert the potentiometer data units to volts. 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 

    Returns
    -------
    calibration_data : dict  
        Calibration data with potentiometer data convert to volts
        
    """
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = (
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts 
            )
    return calibration_data

def pots_to_bits(calibration_data, supply_voltage = 3.3, controller_max_bits = 1023):
    """Convert the potentiometer data units to microcontroller bits. 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 
    supply_voltage : float
        Maximum voltage for the potentiometers
    controller_max_bits : int
        Maximum bits for the microcontroller 

    Returns
    -------
    calibration_data : dict  
        Calibration data with potentiometer data convert to microcontroller 
        bits
        
    """
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = np.round(
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts / supply_voltage * controller_max_bits
            )
    return calibration_data

def get_valve_transitions(calibration_data):
    """Get the valve start and stop times. 

    Parameters
    ----------
    calibration_data : dict
        All of the digital (DIO) and analog data corresponding to the trodes
        files for the a calibration recording. For example, as returned by
        read_data(). 

    Returns
    -------
    start_times : dict  
        Times at which each of the valves transitioned from closed to open
    stop_times : dict
        Times at which each of the valves transitioned from open to closed
        
    """
    start_times = {
        'x_push': calibration_data['DIO']['x_push'][1::2],
        'x_pull': calibration_data['DIO']['x_pull'][1::2],
        'y_push': calibration_data['DIO']['y_push'][1::2],
        'y_pull': calibration_data['DIO']['y_pull'][1::2],
        'z_push': calibration_data['DIO']['z_push'][1::2],
        'z_pull': calibration_data['DIO']['z_pull'][1::2]
    }
    stop_times = {
        'x_push': calibration_data['DIO']['x_push'][2::2],
        'x_pull': calibration_data['DIO']['x_pull'][2::2],
        'y_push': calibration_data['DIO']['y_push'][2::2],
        'y_pull': calibration_data['DIO']['y_pull'][2::2],
        'z_push': calibration_data['DIO']['z_push'][2::2],
        'z_pull': calibration_data['DIO']['z_pull'][2::2]
    }
    return start_times, stop_times

def get_calibration_frame(data_dir, trodes_name, sampling_rate = 3000, valve_period = 0.175, pot_units = 'cm'):
    """Generate a data frame for estimating robot calibration parameters.
    State variables include the starting positions and valve open
    durations for each actuator, and the response variable is 
    displacement. By convention, durations and displacements are assumed 
    to be negative for the pull valves.

    Parameters
    ----------
    data_dir : str
        Parent directory where the trodes data lives
    trodes_name : str
        Name of original .rec trodes file
    sampling_rate : int
        Specifying a rate (Hz) lower than the SpikeGadgets MCU clock rate of
        30 kHz will downsample the data and speed up the parsing.
    valve_period : float
        Time (sec) between valve command onsets
    pot_units : str
        Units to return potentiometer recordings. Can be `cm`, `volts`, or 
        `bits`.

    Returns
    -------
    data_frame : pandas.core.frame.DataFrame  
        A pandas data frame with columns `start_time`, `x_position`, 
        `x_duration`, `x_displacement`, `y_position`, `y_duration`, 
        `y_displacement`, `z_position`, `z_duration`, and `z_displacement`.

    """
    trodes_files = get_trodes_files(data_dir, trodes_name)
    calibration_data = read_data(trodes_files, sampling_rate)
    calibration_data = to_numpy(calibration_data)
    calibration_data = to_seconds(calibration_data)
    if pot_units == 'cm':
        calibration_data = pots_to_cm(calibration_data)
    elif pot_units == 'volts':
        calibration_data = pots_to_volts(calibration_data)
    elif pot_units == 'bits':
        calibration_data = pots_to_bits(calibration_data)
    start_times, stop_times = get_valve_transitions(calibration_data)
    num_events = start_times['x_push'].size + start_times['x_pull'].size
    #sort start times
    x_start_times = np.concatenate([start_times['x_push'], start_times['x_pull']])
    y_start_times = np.concatenate([start_times['y_push'], start_times['y_pull']])
    z_start_times = np.concatenate([start_times['z_push'], start_times['z_pull']])
    x_order = np.argsort(x_start_times)
    y_order = np.argsort(y_start_times)
    z_order = np.argsort(z_start_times)
    #initialize data frame
    data_frame = pd.DataFrame(
        data = {
            'start_time': x_start_times[x_order],
            'x_position': np.zeros(num_events), 
            'x_duration': np.concatenate([
                stop_times['x_push'] - start_times['x_push'], 
                start_times['x_pull'] - stop_times['x_pull'] #sign convention
                ])[x_order], 
            'x_displacement': np.zeros(num_events),
            'y_position': np.zeros(num_events),  
            'y_duration': np.concatenate([
                stop_times['y_push'] - start_times['y_push'], 
                start_times['y_pull'] - stop_times['y_pull'] #sign convention
                ])[y_order], 
            'y_displacement': np.zeros(num_events),
            'z_position': np.zeros(num_events),  
            'z_duration': np.concatenate([
                stop_times['z_push'] - start_times['z_push'], 
                start_times['z_pull'] - stop_times['z_pull'] #sign convention
                ])[z_order], 
            'z_displacement': np.zeros(num_events)
        }
    )
    #estimate positions and displacements
    start_indices = np.searchsorted(calibration_data['time'], data_frame['start_time'])
    stop_indices = start_indices + int(valve_period * sampling_rate)
    data_frame['x_position'] = calibration_data['analog']['x_pot'][start_indices]
    data_frame['y_position'] = calibration_data['analog']['y_pot'][start_indices]
    data_frame['z_position'] = calibration_data['analog']['z_pot'][start_indices]
    data_frame['x_displacement'] = (
        calibration_data['analog']['x_pot'][stop_indices] - 
        calibration_data['analog']['x_pot'][start_indices]
        )
    data_frame['y_displacement'] = (
        calibration_data['analog']['y_pot'][stop_indices] - 
        calibration_data['analog']['y_pot'][start_indices]
        )
    data_frame['z_displacement'] = (
        calibration_data['analog']['z_pot'][stop_indices] - 
        calibration_data['analog']['z_pot'][start_indices]
        )
    return data_frame

# def get_traces_frame(
#   data_dir, 
#   trodes_name, 
#   sampling_rate = 3000, 
#   valve_period = 0.175,
#   pot_units = 'cm'
#   ):

#   return data_frame

    

    
