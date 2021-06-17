"""Script to import trodes, micro-controller, and config data

"""

import glob
import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from software.preprocessing.video_data.DLC.Reconstruction import get_kinematic_data
from software.preprocessing.config_data.config_parser import import_config_data
from software.preprocessing.controller_data.controller_data_parser import import_controller_data, get_reach_indices, get_reach_times
from software.preprocessing.reaching_without_borders.rwb import match_times, get_successful_trials
from software.preprocessing.trodes_data.experiment_data_parser import import_trodes_data
from software.preprocessing.trodes_data.calibration_data_parser import get_traces_frame


def load_files(trodes_dir, exp_name, controller_path, config_dir, rat, session,video_path, analysis=False,
               cns_flag=False, pns=False,positional=False, force_rerun_of_data = True, sample_rate = 150):
    """

    Parameters
    ----------
    save_path : str, path location to save data extraction at ex '/larry/lobsters/home/book/'
    trodes_dir : directory containing trodes .rec file
    exp_name : name of folder containing .rec file/ video file
    controller_path : full path to micro-controller data
    config_dir : directory containing .json file with configuration parameters
    rat : name of rat eg RM16
    session : name of experimental session eg S1
    analysis : boolean, set as True to extract experimental analysis
    video_path : path to video data
    cns_flag : boolean, manual set of cns path
    pns : boolean, manual set of pns path


    Returns
    -------
    dataframe : pandas dataframe containing experimental values for a single experimental session
    """
    # importing data
    exp_names = exp_name[2:-1]
    exp_names = exp_names.rsplit('.', 1)[0]
    trodes_dir = trodes_dir.rsplit('/', 1)[0]
    # search trodes_dir for files named 
    experimental_data_found = 0
    for ff in glob.glob(trodes_dir +'/**experimental_df.h5'):     
        experimental_data_found = ff
        if force_rerun_of_data:
            experimental_data_found = 0
    if experimental_data_found:
        dataframe = pd.read_hdf(experimental_data_found)
    else:
        print('Generating sensor data manually!')
        if positional:
            positional_data = get_traces_frame(trodes_dir, exp_names)
            r_x = positional_data['x_start_position']
            r_y = positional_data['y_start_position']
            r_z = positional_data['z_start_position']
            t_x = positional_data['x_duration']
            d_x = positional_data['x_displacement']
            t_y = positional_data['y_duration']
            d_y = positional_data['y_displacement']
            t_z = positional_data['z_duration']
            d_z = positional_data['z_displacement']
        if cns_flag:
            os.chdir(cns_flag)
        trodes_data = import_trodes_data(trodes_dir, exp_names,sampling_rate = sample_rate ) # take first entry of list (starmap returns list)
        if pns:
            os.chdir(pns)
        try:
            config_data = import_config_data(config_dir)
            controller_data = import_controller_data(controller_path)
        except:
            print('Cant get config or controller data')
        if analysis:
            x_pot=trodes_data['analog']['x_pot']
            y_pot=trodes_data['analog']['y_pot']
            z_pot=trodes_data['analog']['z_pot']
            lick_data = trodes_data['DIO']['IR_beam']
            true_time = match_times(controller_data, trodes_data)
            reach_indices = get_reach_indices(controller_data)
            successful_trials = get_successful_trials(controller_data, true_time, trodes_data)
            reach_masks = get_reach_times(true_time, reach_indices)
            reach_masks_start = np.asarray(reach_masks['start'])
            reach_masks_stop = np.asarray(reach_masks['stop'])
            reach_indices_start = reach_indices['start']
            reach_indices_stop = reach_indices['stop']
            trial_masks = trial_mask(true_time, reach_indices_start, reach_indices_stop, successful_trials)       
            dataframe = to_df(exp_names, config_data, true_time, successful_trials, trial_masks, rat, session, lick_data, controller_data, reach_indices,
                                    x_pot,y_pot,z_pot,reach_masks_start,reach_masks_stop)
            exp_save_dir = trodes_dir + '/experimental_df.h5'
            dataframe.to_hdf(exp_save_dir,key='df')
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
    path_d = file.replace('/cns', '/PNS_data')
    path_d = path_d.rsplit('/R', 2)[0]
    config_path = path_d + '/workspaces'
    controller_path = path_d + '/sensor_data'
    video_path = path_d + '/videos/**.csv'
    # trodes_data
    n = file.rsplit('/', 1)[1]
    if '/S' in file:
        sess = file.rsplit('/S')
        sess = str(sess[1])  # get 'session' part of the namestring
        ix = 'S' + sess[0]
    exp_name = str(ix) + n
    return controller_path, config_path, exp_name, name, ix, n, video_path


def host_off(cns,save_path = False):
    """

    Parameters
    ----------
    save_path : path to save experimental dataframe
    cns : path to cns

    Returns
    -------
    save_df : complete experimental data frame

    """
    pns = '/clusterfs/NSDS_data/brnelson/PNS_data/'
    cns_pattern = cns + '/**/*.rec'
    print(cns_pattern)
    # cns is laid out rat/day/session/file_name/localdir (we want to be in localdir)
    # search for all directory paths containing .rec files
    save_df = pd.DataFrame()
    for file in tqdm(glob.glob(cns_pattern,recursive=True)):
        controller_path, config_path, exp_name, name, ix, trodes_name,video_path = name_scrape(file)
        print(exp_name + ' is being added..')
        list_of_df = load_files(file, exp_name, controller_path, config_path, name, ix,video_path,
                                analysis=True, cns_flag=cns, pns=pns,force_rerun_of_data = True, sample_rate = 600)
        save_df= pd.concat([save_df,list_of_df])
    print('Finished!!')
    if save_path:
        save_df.to_pickle(save_path)
    return save_df


def return_block_kinematic_df(f):
    file_=f[0]
    dlt_path=f[1]
    controller_path, config_path, exp_name, name, ix, trodes_name, video_path = name_scrape(file_)
    date = get_name(file_)
    print(video_path + ' is being added..')
    m = get_kinematic_data(video_path,dlt_path,'101',name,date,ix,0)
    return m


def get_kinematics(cns, dlt_path, rat_name, pik=True, save_path='tkd_filt.pkl'):
    """Function to iterate over a data directory and extract 3-D positional data from CatScan or other data directory.

    Parameters
    -----------
    cns : str, path to cns directory
    dlt_path : str, path to DLT string
    rat_name : str, name of rat ex: 'RM16', 'RM9', 'RF1'
    pik : bool, flag to save resulting dataframe as a pickle file
    save_path : str, path to save our list of dataframes

    Returns
    ---------
    df_list : list, contains dataframe(s) of 3-D positions over an entire experimental block session

    """
    cns_path=cns + rat_name
    cns_pattern = cns_path+ '/**/*.rec'
    d = []
    cl=glob.glob(cns_pattern,recursive=True)
    print(str(len(cl)) + ' of experimental blocks for this rat.')
    df_list=[]
    for file in tqdm(glob.glob(cns_pattern, recursive=True)):
        mergies = return_block_kinematic_df([file,dlt_path])
        df_list.append(mergies)
    print('Saving DataFrame')
    os.chdir(cns)
    if pik:
        with open(save_path, 'wb') as output:
            pickle.dump(df_list, output, pickle.HIGHEST_PROTOCOL)
    return df_list


def get_config_data(config_data):
    """

    Parameters
    ----------
    config_data : list containing config parameters

    Returns
    -------
    [config parameters] : various parameters and values from the config file
    """
    exp_type = config_data['RobotSettings']['commandFile']
    reward_dur = config_data['ExperimentSettings']['rewardWinDur']
    x_p = config_data['RobotSettings']['xCommandPos']
    y_p = config_data['RobotSettings']['yCommandPos']
    z_p = config_data['RobotSettings']['zCommandPos']
    x0 = config_data['RobotSettings']['x0']
    y0 = config_data['RobotSettings']['y0']
    z0 = config_data['RobotSettings']['z0']
    r = config_data['RobotSettings']['x']
    t1 = config_data['RobotSettings']['y']
    t2= config_data['RobotSettings']['z']
    return exp_type, reward_dur, x_p, y_p, z_p, x0, y0, z0,r,t1,t2


def to_df(file_name, config_data, true_time,successful_trials, trial_masks, rat, session,
            lick_data, controller_data, reach_indices,x_pot,y_pot,z_pot,mstart,mstop, save_as_dict=False):
    """

    Parameters
    ----------
    file_name : name of experiment file
    config_data : experimental parameters saved as a config json file for each experiment
    true_time : normalized time array
    successful_trials : array containing indices of successful trials eg [1,3,6..]
    trial_masks : mask array of normalized times containing binary success [1] and fail [0] values
    rat : rat name eg RM16
    session : experimental session eg S1
    lick_data : array of lick start and stop times
    controller_data : list containing controller data
    reach_indices : list of 'start' and 'stop' indices for reaching trials
    save_as_dict : boolean, saves the results as a dict (depreciated)

    Returns
    -------
    dict : pandas dataframe containing an experiments data
    """
    # functions to get specific items from config file
    dim, reward_dur, x_pos, y_pos, z_pos, x0, y0, z0, r, t1, t2 = get_config_data(config_data)
    date = get_name(file_name,Trodes=False)
    moving = controller_data['rob_moving']
    r_w = controller_data['in_Reward_Win']
    exp_response = controller_data['exp_response']
    successful_trials = np.asarray(successful_trials)  
    block_dict = pd.DataFrame(
            {'rat': rat, 'S': session, 'Date': date, 'dim': dim, 'time': [np.asarray(true_time).tolist()],
             'SF': [successful_trials], 't_m': [trial_masks],'m_start':[mstart],'m_stop':[mstop],
             'lick': [np.asarray(lick_data).tolist()], 
             'x_p': [np.asarray(x_pos).tolist()], 'y_p': [np.asarray(y_pos).tolist()],
             'z_p': [np.asarray(z_pos).tolist()], 'x0': [x0], 'y0': [y0], 'z0': [z0],
             'moving': [np.asarray(moving, dtype=int)], 'RW': [r_w], 'r_start': [reach_indices['start']],
             'r_stop': [reach_indices['stop']], 'r':[r],'t2': [t2], 't1':[t1],
             'exp_response' : [exp_response], 'x_pot':[x_pot],'y_pot':[y_pot],'z_pot':[z_pot]})
    return block_dict


def make_dict():
    return defaultdict(make_dict)


def get_name(file_name,Trodes=False):
    """Function to fetch name data from string of names

    Parameters
    ----------
    file_name : un-cleaned file name

    Returns
    -------
    date: string, cleaned experiment data
    """
    # split file name
    if Trodes:
        date=file_name[5:12]
    else:
        try:
            file_name=file_name.split('/')[-2]
            date = file_name[5:12]
        except:
            date=file_name[5:12]
    print(date)
    return date


def trial_mask(matched_times, r_i_start, r_i_stop, s_t):
    """Function to

    Parameters
    ----------
    matched_times : array of normalized timestamps
    r_i_start : indices of reaching experiments, start
    r_i_stop : indices of reaching experiments, stop
    s_t : success or fail indices eg [1, 4, 7..]

    Returns
    -------
    new_times : array of experiment times
    """
    lenx = int(matched_times.shape[0])
    new_times = np.zeros((lenx))
    for i, j in zip(range(0, len(r_i_start) - 1), range(0, len(r_i_stop) - 1)):
        ix = int(r_i_start[i])
        jx = int(r_i_stop[i])
        if any(i == s for s in s_t):
            new_times[ix:jx] = 2
        else:
            new_times[ix:jx] = 1
    return new_times

# run
