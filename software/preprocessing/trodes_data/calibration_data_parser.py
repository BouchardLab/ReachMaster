"""A library for reading, parsing, and preprocessing trodes data 
collected during robot calibration routines.

"""

import os
from . import readTrodesExtractedDataFile3 as read_trodes
import numpy as np
import pandas as pd
import subprocess as sp 

def _get_trodes_files(data_dir, trodes_name):
    trodes_files = {
        'time_file':  data_dir + '/%s/%s.analog/%s.timestamps.dat' % (trodes_name, trodes_name, trodes_name),
        'x_push_file': data_dir + '/%s/%s.DIO/%s.dio_x_push.dat' % (trodes_name, trodes_name, trodes_name),
        'x_pull_file': data_dir + '/%s/%s.DIO/%s.dio_x_pull.dat' % (trodes_name, trodes_name, trodes_name),
        'y_push_file': data_dir + '/%s/%s.DIO/%s.dio_y_push.dat' % (trodes_name, trodes_name, trodes_name),
        'y_pull_file': data_dir + '/%s/%s.DIO/%s.dio_y_pull.dat' % (trodes_name, trodes_name, trodes_name),
        'z_push_file': data_dir + '/%s/%s.DIO/%s.dio_z_push.dat' % (trodes_name, trodes_name, trodes_name),
        'z_pull_file': data_dir + '/%s/%s.DIO/%s.dio_z_pull.dat' % (trodes_name, trodes_name, trodes_name),
        'x_pot_file': data_dir + '/%s/%s.analog/%s.analog_potX.dat' % (trodes_name, trodes_name, trodes_name),
        'y_pot_file': data_dir + '/%s/%s.analog/%s.analog_potY.dat' % (trodes_name, trodes_name, trodes_name),
        'z_pot_file': data_dir + '/%s/%s.analog/%s.analog_potZ.dat' % (trodes_name, trodes_name, trodes_name)
    }
    return trodes_files

def _read_data(trodes_files, sampling_rate = 3000):
    clockrate = np.float_(read_trodes.readTrodesExtractedDataFile(time_file)['clock rate'])
    ds = clockrate / sampling_rate
    calibration_data = {
        'clockrate': clockrate,
        'time': read_trodes.readTrodesExtractedDataFile(time_file)['data'][0:-1:ds],
        'DIO': {
            'x_push': read_trodes.readTrodesExtractedDataFile(x_push_file)['data'], 
            'x_pull': read_trodes.readTrodesExtractedDataFile(x_pull_file)['data'],
            'y_push': read_trodes.readTrodesExtractedDataFile(y_push_file)['data'], 
            'y_pull': read_trodes.readTrodesExtractedDataFile(y_pull_file)['data'], 
            'z_push': read_trodes.readTrodesExtractedDataFile(z_push_file)['data'], 
            'z_pull': read_trodes.readTrodesExtractedDataFile(z_pull_file)['data']
        },
        'analog':{
            'x_pot': read_trodes.readTrodesExtractedDataFile(x_pot_file)['data']['voltage'][0:-1:ds],
            'y_pot': read_trodes.readTrodesExtractedDataFile(y_pot_file)['data']['voltage'][0:-1:ds],
            'z_pot': read_trodes.readTrodesExtractedDataFile(z_pot_file)['data']['voltage'][0:-1:ds]
        }
    }
    return calibration_data

def _to_numpy(calibration_data):
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

def _to_seconds(calibration_data, start_at_zero = True):
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
    return calibration_data

def _pots_to_cm(calibration_data, supply_voltage = 3.3, pot_range = 5.0):
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = (
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts / supply_voltage * pot_range
            )
    return calibration_data

def _pots_to_volts(calibration_data):
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = (
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts 
            )
    return calibration_data

def _pots_to_bits(calibration_data, supply_voltage = 3.3, controller_max_bits = 1023):
    trodes_max_bits = 32767.0
    trodes_max_volts = 10.0
    for key in calibration_data['analog'].keys():
        calibration_data['analog'][key] = np.round(
            calibration_data['analog'][key] / trodes_max_bits * trodes_max_volts / supply_voltage * controller_max_bits
            )
    return calibration_data

def _get_valve_transitions(calibration_data):
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

def get_summary_frame(data_dir, trodes_name, sampling_rate = 3000, valve_period = 0.175, pot_units = 'cm'):
    """Generate a data frame that can be modeled for robot calibration.

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
    data_frame : type(pandas.DataFrame)  
        A pandas data frame 
    """

    trodes_files = _get_trodes_files(data_dir, trodes_name)
    calibration_data = _read_data(trodes_files, sampling_rate)
    calibration_data = _to_numpy(calibration_data)
    calibration_data = _to_seconds(calibration_data)
    if pot_units == 'cm':
        calibration_data = _pots_to_cm(calibration_data)
    elif pot_units == 'volts':
        calibration_data = _pots_to_volts(calibration_data)
    elif pot_units == 'bits':
        calibration_data = _pots_to_bits(calibration_data)
    start_times, stop_times = _get_valve_transitions(calibration_data)
    num_events = start_times['x_push'].size + start_times['x_pull'].size
    #preallocate data frame
    data_frame = pd.DataFrame(
        data = {
            'onset_time': np.zeros(num_events),
            'x_position': np.zeros(num_events), 
            'x_duration': np.zeros(num_events), 
            'x_displacement': np.zeros(num_events),
            'y_position': np.zeros(num_events), 
            'y_duration': np.zeros(num_events), 
            'y_displacement': np.zeros(num_events),
            'z_position': np.zeros(num_events), 
            'z_duration': np.zeros(num_events), 
            'z_displacement': np.zeros(num_events)
        }
    )
    #estimate valve open durations
    durations = start_times
    for key in durations.keys():
        durations[key] = stop_times[key] - start_times[key]
    #match events based on start times
    old_x_start = 0
    old_y_start = 0
    old_z_start = 0
    for key in start_times.keys():
        #lazy hack to make sure min() below always exists
        np.append(start_times[key], [99999999999999999999]) 
    for i in range(num_events):
        #determine if x actuator pushed or pulled
        x_push_start = np.min(
            start_times['x_push'][start_times['x_push'] > old_x_start]
            )
        x_pull_start = np.min(
            start_times['x_pull'][start_times['x_pull'] > old_x_start]
            )
        if(x_push_start < x_pull_start): #event was an x_push
            idx1 = np.min(np.where(
                    calibration_data['time'] >= x_push_start
                    ))
            idx2 = np.min(np.where(
                    calibration_data['time'] >= (
                        calibration_data['time'] + valve_period
                        )
                    ))
            data_frame['x_duration'][i] = durations['x_push'][np.where(start_times['x_push'] == x_push_start)]
            old_x_start = x_push_start
        else: #event was an x_pull
            idx1 = np.min(np.where(
                    calibration_data['time'] >= x_pull_start
                    ))
            idx2 = np.min(np.where(
                    calibration_data['time'] >= (
                        calibration_data['time'] + valve_period
                        )
                    ))
            data_frame['x_duration'][i] = -durations['x_pull'][np.where(start_times['x_pull'] == x_pull_start)]
            old_x_start = x_pull_start
        #determine if y actuator pushed or pulled
        y_push_start = np.min(
            start_times['y_push'][start_times['y_push'] > old_y_start]
            )
        y_pull_start = np.min(
            start_times['y_pull'][start_times['y_pull'] > old_y_start]
            )
        if(y_push_start < y_pull_start): #event was a y_push
            data_frame['y_duration'][i] = durations['y_push'][np.where(start_times['y_push'] == y_push_start)]
            old_y_start = y_push_start
        else: #event was a y_pull
            data_frame['y_duration'][i] = -durations['y_pull'][np.where(start_times['y_pull'] == y_pull_start)]
            old_y_start = y_pull_start
        #determine if z actuator pushed or pulled
        z_push_start = np.min(
            start_times['z_push'][start_times['z_push'] > old_z_start]
            )
        z_pull_start = np.min(
            start_times['z_pull'][start_times['z_pull'] > old_z_start]
            )
        if(z_push_start < z_pull_start): #event was a z_push
            data_frame['z_duration'][i] = durations['z_push'][np.where(start_times['z_push'] == z_push_start)]
            old_z_start = z_push_start
        else: #event was a z_pull
            data_frame['z_duration'][i] = -durations['z_pull'][np.where(start_times['z_pull'] == z_pull_start)]
            old_z_start = z_pull_start 
        data_frame['onset_time'][i] = calibration_data['time'][idx1]   
        data_frame['x_position'][i] = calibration_data['x_pot'][idx1]
        data_frame['y_position'][i] = calibration_data['y_pot'][idx1]
        data_frame['z_position'][i] = calibration_data['z_pot'][idx1]
        data_frame['x_displacement'][i] = calibration_data['x_pot'][idx1] - calibration_data['x_pot'][idx1]
        data_frame['y_displacement'][i] = calibration_data['y_pot'][idx2] - calibration_data['y_pot'][idx1]
        data_frame['z_displacement'][i] = calibration_data['z_pot'][idx2] - calibration_data['z_pot'][idx1]
    return data_frame

# def get_traces_frame(
#   data_dir, 
#   trodes_name, 
#   sampling_rate = 3000, 
#   valve_period = 0.175,
#   pot_units = 'cm'
#   ):

#   return data_frame

    

    
