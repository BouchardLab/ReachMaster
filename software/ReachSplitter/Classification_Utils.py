"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 12/8/2020
               Emily Nguyen, UC Berkeley

    This code is intended to create and implement structure supervised classification of coarsely
    segmented trial behavior from the ReachMaster experimental system.
    Functions are designed to work with a classifier of your choice.

    Edited: 12/8/2020
"""

import numpy as np
import pandas as pd

from networkx.drawing.tests.test_pylab import plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import decomposition
from sklearn import preprocessing
import pdb
from sklearn.pipeline import make_pipeline
from yellowbrick.model_selection import CVScores
from yellowbrick.features import Rank2D
from Classification_Visualization import visualize_model, print_preds, plot_decision_tree
from yellowbrick.classifier import ClassificationReport
import DataStream_Vis_Utils as utils
from scipy import ndimage

# for saving and loading files
import h5py
import os.path


####################################
# 1. Load data into pandas DataFrames
####################################

def unpack_pkl_df(rat_df1):
    """ Formats a pandas DataFrame.
        rat,date,session,dim levels are converted to columns.
        Each column's values (except rat,date,session,dim) are stored in a single array.

    Args:
        rat_df1 (df):  a multi-level DataFrame corresponding to an element in an un-pickled list

    Returns:
        new_df (df): a new formatted DataFrame with only 1 row for a specific rat,date,session,dim
    Notes:
        pkl_to_df helper
    """
    # create a new DataFrame (copy of original), then removes levels corresponding to (rat, date, session, dim)
    df1 = rat_df1.droplevel([0, 1, 2, 3], axis=1)

    # stores final row values for each rat,date,session
    newlist = []

    # create empty df by removing all rows
    new_df = df1[0:0]

    # inserts rat, date, session, dim columns
    for i in range(4):
        col_name = rat_df1.columns.names[i]
        val = rat_df1.columns.levels[i][0]
        newlist.append(val)
        new_df.insert(i, col_name, val)

    # turn each column's values into an array
    for i in range(len(df1.columns)):
        newlist.append(df1[df1.columns[i]].values)

    # append list of values into empty df
    to_append = newlist
    a_series = pd.Series(to_append, index=new_df.columns)
    new_df = new_df.append(a_series, ignore_index=True)
    return new_df


def pkl_to_df(pickle_file):
    """ Converts a pickle files into a pandas DataFrame indexed by rat,date,session,dim

    Args:
        pickle_file (str): file path to pickle file that contains a list of DataFrames

    Returns:
        df_to_return (df): DataFrame indexed by rat,date,session,dim
            with one row corresponding to one element in 'unpickled_list'
    """
    # unpickle file
    unpickled_list = pd.read_pickle(pickle_file)

    # true if have encountered a DataFrame in list
    encountered_df = False

    # iterate through list
    for i in range(len(unpickled_list)):
        try:
            rat_df1 = unpickled_list[i]

            # create new df with 1 row
            if (not encountered_df):
                encountered_df = True
                df_to_return = unpack_pkl_df(rat_df1)

            # concat new df to existing df
            else:
                df_to_append = unpack_pkl_df(rat_df1)
                df_to_return = pd.concat([df_to_return, df_to_append], axis=0, sort=False)

        except:
            print("do nothing, not a valid df")

    # sets index of new df
    df_to_return = df_to_return.set_index(['rat', 'date', 'session', 'dim'])
    return df_to_return


####################################
# 2. Label processing
####################################

def make_vectorized_labels(blist):
    """ Vectorizes list of DLC video trial labels for use in ML-standard format
        Converts labels which hand and tug vs no tug string labels into numbers.

    Args:
        blist (list of str and int): list of trial labels for a specific rat,date,session
            For more robust description please see github

    Returns:
        new_list (arr of lists): array of lists of numeric labels.
            One list corresponds to one labeled trial.
        ind_total (arr of list): array of lists of reaching indices .
            Currently all empty.

    Notes:
         Each index in a labeled trial are as follows: [int trial_num, int start, int stop, int trial_type,
         int num_reaches, str which_hand_reach, str tug_noTug, int hand_switch, int num_frames]

    Example:
    >>> l18=[[1,0,0,1,1,'l','noTug',0,0]]
    >>> l18l, ev18 = make_vectorized_labels(l18)
    >>> display(l18l)
    array([[ 1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.]])

    """
    ll = len(blist)
    new_list = np.empty((ll, 9))
    ind_total = []
    for ix, l in enumerate(blist):
        # transform non-numerics into numeric template
        # arm type: 'lr': 2 , 'l' : 1, 'bi' : 3 , 'lbi' : 4, 'r': 0, 'tug': 1
        if 'l' in str(l[5]):
            if 'lr' in str(l[5]):
                blist[ix][5] = 2
            else:
                blist[ix][5] = 1
        elif 'bi' in str(l[5]):
            if 'lbi' in str(l[5]):
                blist[ix][5] = 4
            else:
                blist[ix][5] = 3
        if 'r' in str(l[5]):
            blist[ix][5] = 0
        if l[5] == 0:
            blist[ix][5] = 5  # null trial
        try:
            if 'no' in str(l[6]):
                blist[ix][6] = 0
            else:
                blist[ix][6] = 1
        except:
            continue
        # code to split off any trial indices
        try:
            if len(l) > 9:  # are there indices?
                ind_total.append([l[9], l[10]])
            if len(l) > 11:  # second indices?
                ind_total.append([l[11], l[12]])
            if len(l) > 13:
                ind_total.append([l[13], l[14]])
            if len(l) > 15:
                ind_total.append([l[15], l[16]])
            if len(l) > 17:
                ind_total.append([l[17], l[18]])
            if len(l) > 19:
                ind_total.append([l[19], l[20]])
        except:
            print("index error", ix)
        new_list[ix, :] = blist[ix][0:9]
    return new_list, np.array(ind_total)


####################################
# 3. Format Blocks into Trial-ized Data
####################################

def xform_array(k_m, eit, cl):
    """ Does nothing. Returns 'k_m' unchanged.

    Args:
        k_m (list of list of ints): list of three x, y, z lists
        eit (int): et
        cl (int): el

    Returns:
        k_m
    """
    k = np.asarray(k_m)
    # rv=np.dot(k.reshape(k.shape[2],27,3),eit)+cl # good Xform value, then use this value
    return k_m


def transpose_kin_array(kc, et, el):
    # kc is dtype pandas dataframe
    # we want numpy data format
    # kc['colname'].to_numpy()
    for items in kc.items():
        # pdb.set_trace()
        if 'X' in items[0][1]:
            kc[items[0]] = xform_array(kc[items[0]], et, el)
        if 'Y' in items[0][1]:
            kc[items[0]] = xform_array(kc[items[0]], et, el)
        if 'Z' in items[0][1]:
            kc[items[0]] = xform_array(kc[items[0]], et, el)
    return kc


def block_pos_extract(kin_df, et, el, wv):
    """ Extracts a single block's posture data (X,Y,Z and prob1,2,3) and feature names.
        
    Args:
        kin_df (df): kinematic DataFrame for a single block
        et (int): coordinate change variable
            Will take the positional coordinates and put them into the robot reference frame.
        el (int): coordinate change variable
            Will take the positional coordinates and put them into the robot reference frame.
        wv (int): the wavelet # for the median filter applied to the positional data

    Returns:
        block (list of lists of arrays of ints): List of two lists:
            (1) three lists for X,Y,Z column data.
            (2) three lists for prob1,2,3 column data
            Each list corresponding to X,Y, or Z or prob1,2 or 3 contains filtered arrays for each feature.
            final Posture array is 2 x 3 x feat x coords np array.
        feat_name (list of str): collection of feature names for X,Y,Z followed by probabilities 1,2,3.
            1 for Left bodypart, 2 for Right bodypart
    Notes:
        make_s_f_trial_arrays_from_block helper
        Excludes probability column data
        
    Example:
        >>> print(len(block)) # 2 for XYZ and probability lists
            2
        >>> print(len(block[0])) # 3 for X,Y,Z data
            3
        >>> print(len(block[0][0]))
            27        # 27 for number of bodyparts for XYZ repetitions across columns
        >>> print(len(kin_df[('Handle', 'X')].values[0]), len(block[0][0][0]))
            91580, 91580    # 91580 coordinates in array same as num rows in df.
                     
    """
    # initialize data handling lists
    x_arr_ = []
    y_arr_ = []
    z_arr_ = []
    xp_ = []
    yp_ = []
    zp_ = []

    feat_name_X = []
    feat_name_Y = []
    feat_name_Z = []
    feat_name_prob1 = []
    feat_name_prob2 = []
    feat_name_prob3 = []

    # iterate across columns
    for (columnName, columnData) in kin_df.iteritems():

        # Apply median filter to X,Y,or Z array values
        try:
            if columnName[1] == 'X':
                x_arr_.append(ndimage.median_filter(columnData.values[0], wv))
                feat_name_X.append(columnName)
            elif '1' in columnName[1]:
                xp_.append(columnData.values[0])
                feat_name_prob1.append(columnName)
            elif columnName[1] == 'Y':
                y_arr_.append(ndimage.median_filter(columnData.values[0], wv))
                feat_name_Y.append(columnName)
            elif '2' in columnName[1]:
                yp_.append(columnData.values[0])
                feat_name_prob2.append(columnName)
            elif columnName[1] == 'Z':
                z_arr_.append(ndimage.median_filter(columnData.values[0], wv))
                feat_name_Z.append(columnName)
            elif '3' in columnName[1]:
                zp_.append(columnData.values[0])
                feat_name_prob3.append(columnName)
        except:
            print('No filtering..')

    # Concat all XYZ lists and probability lists of column values
    block = np.asarray(
        [[x_arr_, y_arr_, z_arr_], [xp_, yp_, zp_]])  # previously: xform_array([x_arr_, y_arr_, z_arr_], et, el)

    # print(block.shape) = (2, 3, 27, 91580)
    feat_name = feat_name_X+feat_name_Y+feat_name_Z+feat_name_prob1+feat_name_prob2+feat_name_prob3
    return block, feat_name


def reshape_posture_trials(a):
    # pdb.set_trace()
    for ix, array in enumerate(a):
        if ix == 0:
            arx = a

        else:
            ar = np.vstack((a))
            # pdb.set_trace()
            arx = np.concatenate((arx, ar))
    return arx


def split_trial(posture_array, _start, window_length, pre):
    """ Splits posture array of kinematic data into trials
            based on '_start' values.

    Args:
        posture_array (list of lists of array of ints): 2x3xfeatxcoords np array.
            Return value from block_pos_extract.
        _start (list of ints): video frame numbers for the start of each trial.
        window_length (int): number of frames to take from a trial
        pre (int): pre cut off before a trial starts, the number of frames to load data from before start time
            For trial splitting

    Returns:
        trials_list (nested arrays of ints): trial-ized kinematic data
            Shape (2, NumTrials, 3 for XYZ positions or prob123, NumFeatures or posture_array.shape[2], NumFrames)
            where NumFrames='window_length' + 'pre'
    Notes:
        make_s_f_trial_arrays_from_block helper
        ML format is time x feat x coords
    
    Examples:
        >>>print(len(trials_list[0][0])) # 3 for XYZ
            3
        >>>print(len(trials_list[0][0][0])) # 27 for pos names
            27
        >>>len(trials_list[0][0][0][0]) # num frames taken depending on 'window_length' and 'pre'
        >>>trials_list[0][0][0][0][0] # int value
    """
    # convert list to np array
    posture_array = np.asarray(posture_array)

    # create e.g (2 for xyz and prob123, e.g 26 trials, 3 for XYZ, 27 pos names, num frames + num pre frames to take) empty list
    trials_list = np.empty((2, len(_start), posture_array.shape[1], posture_array.shape[2], window_length + pre))

    # iterate over starting frames
    # jx=starting frame number
    for i, jx in enumerate(_start):
        try:
            trials_list[:, i, :, :, :] = posture_array[:, :, :, jx - pre:jx + window_length]
        except:
            print('no Kinematic Data imported')

    return trials_list


def onehot(r_df):
    """ Returns one hot array for robot data.

    Args:
        r_df (df): robot DataFrame for a single block

    Returns:
        one hot array of length number of trials in block
        
    Examples:
        >>> display(r_df['r_start'].values[0]) # sf variable
        array([ 1,  2,  4,  5,  6,  7,  9, 10, 12, 14, 15, 17], dtype=int64)
        >>> onehot(r_df) # 1 for index at sf value, 0 otherwise
        array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0])
        
    """
    try:
        # get number of trials
        m = len(r_df['r_start'].values[0])
    except:
        print('onehot vector error')
        # pdb.set_trace()

    # get SF values
    sf = r_df['SF'].values[0]

    # create empty array of size num trials
    hot_vec = np.zeros(m, dtype=int)
    hot_vec[sf] = 1
    return np.asarray(hot_vec)


def unwind_list(strial_list):
    for ix, trial in enumerate(strial_list):
        if ix == 0:
            tsx = trial
        if ix != 0:
            try:
                tsx = np.concatenate((tsx, trial), axis=0)
            except:
                print('concat error in unwind_list')

    return tsx


def reshape_list(strial_list):
    # pdb.set_trace()
    try:
        length, width, height = np.array(strial_list[0]).shape
        num_trials = len(strial_list)
        # make empty array
        emp = np.empty((num_trials, length, width, height))
        # loop over len num_trials
        for iterz, trials in enumerate(strial_list):
            ts = np.array(trials)
            try:
                emp[iterz, :, :, :] = ts
            except:
                emp[iterz, :, :, :] = np.zeros((length, width, height))
    except:
        # pdb.set_trace()
        print('Problem with reshaping the list..')
    return emp


def forward_xform_coords(x, y, z):
    """ Calculates x,y,z pot arrays to help with convertings units in meters to seconds. 
    
    Args:
        x (arr): split x pot array in block from (starting frame*sample_rate to stop frame * sampple_rate)
        y (arr): split y pot array in block
        z (arr): split z pot array in block
    
    Returns:
           r, theta, phi, x, y, z
    
    Notes: 
        calculate_robot_features helper. 
    
    """
    Axx = 168
    Ly = 64
    Ayy = 100
    Lz = 47
    Azz = 117
    X0 = 1024
    Y0 = 608
    Z0 = 531
    Ax_est = (x - X0) / (1024 * 50) + Axx
    Ay_est = (y - Y0) / (1024 * 50) + Ayy
    Az_est = (z - Z0) / (1024 * 50) + Azz
    c1 = np.asarray((0, 0, 0))
    c2 = np.asarray((Ly, Ayy, 0))
    c3 = np.asarray((Lz, 0, Azz))
    u = np.asarray((Ly, Ayy, 0)) / np.sqrt(Ly ** 2 + Ayy ** 2)
    v = c3 - np.dot(c3, u) * u
    v = v / np.sqrt(np.dot(v, v))
    w = np.cross(u, v)
    y1 = np.asarray((0, 1, 0))
    z1 = np.asarray((0, 0, 1))
    U2 = np.sqrt(np.sum((c2 - c1) ** 2))
    U3 = np.dot(c3, u)
    V3 = np.dot(c3, v)
    sd = np.dot(c3, c3)
    cos_top = (Az_est ** 2 + Lz ** 2 - sd)
    cos_bot = (2 * Az_est * Lz)
    r3 = np.sqrt(
        Az_est ** 2 + (Ly - Lz) ** 2 - (2 * Az_est * (Ly - Lz) * np.cos(np.pi - np.arccos((Az_est ** 2 + Lz ** 2 - sd)
                                                                                          / (2 * Az_est * Lz)))))
    Pu = (Ly ** 2 - Ay_est ** 2 + U2 ** 2) / (2 * U2)
    Pv = (U3 ** 2 + V3 ** 2 - 2 * U3 * Pu + Ly ** 2 - r3 ** 2) / (2 * V3)
    Pw = np.sqrt(-Pu ** 2 - Pv ** 2 + Ly ** 2)
    Py = Pu * np.dot(u, y1) + Pv * np.dot(v, y1) + Pw * np.dot(w, y1)
    Pz = Pu * np.dot(u, z1) + Pv * np.dot(v, z1) + Pw * np.dot(w, z1)
    gammay_est = np.arcsin(Py / (Ly * np.cos(np.arcsin(Pz / Ly))))
    gammaz_est = np.arcsin(Pz / Ly)
    r = np.sqrt(Axx ** 2 + Ax_est ** 2 - (2 * Axx * Ax_est * np.cos(gammay_est) * np.cos(gammaz_est)))
    dz = np.sin(-gammaz_est)
    dy = np.sin(-gammay_est)
    theta = np.arcsin(dz * Ax_est / r)
    phi = np.arcsin(Ax_est * dy * np.cos(-gammaz_est) / r / np.cos(theta))
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    return r, theta, phi, x, y, z


def calculate_robot_features(xpot, ypot, zpot, mstart_, mstop_, etimes, wle, pre_):
    """ Calculates robot features.
    
    Args:
        xpot(arr): x pot array in robot block
        ypot(arr): y pot array in robot block
        zpot(arr): z pot array in robot block
        mstart_ (arr): r_start array in robot block
        mstop_ (arr): r_stop array in robot block
        etimes (arr) : experimental times
    Returns:
        vx (arr): units of seconds
        vy (arr): units of seconds
        vz (arr): units of seconds
        x1 (arr): units of meters
        y1 (arr):  units of meters
        z1 (arr): units of meters
        
    Notes: 
        import_experiment_features helper.
        making 'sample_rate' too big can cause index out of bounds and empty array errors.
        'sampe_rate': the adc rate is to take the analog trodes time (x.xxx seconds) and 
            transform it into the digital bit time (frames at some exposure).
    """
    # print(len(etimes))
    # pdb.set_trace()
    etimes = etimes[mstart_ - pre_: mstart_ + wle]
    # print len
    # print(mstart_,mstop_)
    sample_rate = 3000  # sample rate per second
    # take event time, multiply by sample rate
    # split into sample
    try:
        start__ = int(etimes[0] * sample_rate)
        stop__ = int(etimes[-1] * sample_rate)
    except:
        # pdb.set_trace()
        print('bad sampling conversion')

    #print(start__, stop__)
    try:
        # print("START:", start__, "STOP:", stop__, "LEN POT ARR:", len(xpot))
        xp = xpot[start__:stop__]
        yp = ypot[start__:stop__]
        zp = zpot[start__:stop__]
        # if (len(xp)==0):
        # print("START:", start__, "STOP:", stop__, "LEN POT ARR:", len(xpot))
        # print("sample_rate too large or start>stop index")
    except:
        print('bad pot data')
        # pdb.set_trace()
    try:
        r1, theta1, phi1, x1, y1, z1 = forward_xform_coords(xp, yp, zp)  # units of meters
    except:
        print('bad xform')
        # pdb.set_trace()
    # vta = np.diff(vt)
    try:
        vx = np.diff(x1) / sample_rate  # units of seconds
        vy = np.diff(y1) / sample_rate  # units of seconds
        vz = np.diff(z1) / sample_rate  # units of seconds
    except:
        print('bad potentiometer derivatives')
        # pdb.set_trace()
    # ve=np.absolute(np.gradient(np.asarray([x1,y1,z1]),axis=0))/30000
    # ae = np.gradient(ve)/3000
    return vx, vy, vz, x1, y1, z1


def import_experiment_features(exp_array, starts, window_length, pre):
    """ Extracts features from experiment array.
        Features to extract are pot_x, pot_y, pot_z, lick_array, rew_zone robot features.

    Args:
        exp_array (df): single block from robot DataFrame
        starts (list of ints): list of ints corresponding to video frame numbers for each trial
        window_length (int): trial splitting window length, the number of frames to load data from
        pre (int): pre cut off before a trial starts, the number of frames to load data from before start time
            For trial splitting

    Returns:
        exp_feat_array (list of arrays): dimensions Num trials X num robot Features X wlength

    Notes:
        make_s_f_trial_arrays_from_block helper
        making 'window_length' or 'pre' too big can cause index out of bounds errors
        
    """
    # features to extract from experiment array : pot_x, pot_y, pot_z, lick_array, rew_zone robot features
    wlength = (starts[0] + window_length) - (
                starts[0] - pre)  # get length of exposures we are using for trial classification

    # create empty np array Ntrials X Features X wlength
    exp_feat_array = np.empty((len(starts), 12, wlength))

    # loop over each trial
    for ixs, vals in enumerate(starts):
        # get experimental feature values
        try:
            time_ = exp_array['time'].to_numpy()[0]
            moving = exp_array['moving'].to_numpy()[0]
            rz = exp_array['RW'].to_numpy()[0]
        except:
            print('f1')

        rz = rz[vals - pre:vals + window_length]
        lick = exp_array['lick'].to_numpy()[0]
        exp_time = np.asarray(time_, dtype=float) - time_[0]
        exp_save_time = exp_time[vals - pre:vals + window_length]
        lmi = np.searchsorted(exp_time, lick)

        # get lick times, subtract time[0] from lick times (this normalizes them to cam data), then make your vector
        lick_mask_array = np.zeros(exp_time.shape[0] + 1, dtype=int)
        lick_mask_array[lmi] = 1

        # FutureWarning: in the future negative indices will not be ignored by `numpy.delete`.
        #   np.delete(lick_mask_array, [-1])
        np.delete(lick_mask_array, [-1])

        lick_mask_array = lick_mask_array[vals - pre:vals + window_length]

        # moving = moving[vals-pre:vals+window_length] # TODO repl with is_reach_rewarded
        moving = is_reach_rewarded(lick_mask_array)  ### is int 0
        try:
            # get robot data
            vcx, vcy, vcz, xp1, yp1, zp1 = calculate_robot_features(exp_array['x_pot'].to_numpy()[0],
                                                                    exp_array['y_pot'].to_numpy()[0],
                                                                    exp_array['z_pot'].to_numpy()[0],
                                                                    exp_array['r_start'].to_numpy()[0][ixs],
                                                                    exp_array['r_stop'].to_numpy()[0][ixs],
                                                                    exp_array['time'].to_numpy()[0],
                                                                    window_length, pre
                                                                    )
        except:
            print("bad pot data, r/m_start and stop column name error")

        # fill in array
        # decimate potentiometer from s to exposure !~ in frames (we dont have exact a-d-c)
        # make same length as other features, to avoid problems with ML (this means we will lose ~5 data points at end of experiment)
        # try:
        #   pdb.set_trace()
        # xp1 = xp1[0: window_length+pre]
        # yp1 = yp1[0: window_length+pre]
        # zp1 = zp1[0: window_length+pre]
        # vcx = vcz[0: window_length+pre]
        # vcy = vcy[0: window_length+pre]
        # vcz = vcz[0: window_length+pre]
        # except:
        # print('bad decimate')

        try:
            #print(xp1.shape, vcz.shape, yp1.shape)
            ds_rate = int(xp1.shape[0] / (window_length + pre))  # n frames of analog / n frames of digital
            exp_feat_array[ixs, 0, :] = exp_save_time
            exp_feat_array[ixs, 6, :] = rz

            try:
                exp_feat_array[ixs, 7, :] = xp1[::ds_rate][0:window_length + pre]
                exp_feat_array[ixs, 8, :] = yp1[::ds_rate][0:window_length + pre]
                exp_feat_array[ixs, 9, :] = zp1[::ds_rate][0:window_length + pre]
            # exp_feat_array[ixs, 1, :] = vcx
            # exp_feat_array[ixs, 2, :] = vcy
            # exp_feat_array[ixs, 3, :] = vcz
            except:
                exp_feat_array[ixs, 7, :] = xp1[::ds_rate - 1]
                exp_feat_array[ixs, 8, :] = yp1[::ds_rate - 1]
                exp_feat_array[ixs, 9, :] = zp1[::ds_rate - 1]

            exp_feat_array[ixs, 10, :] = lick_mask_array
            exp_feat_array[ixs, 11, :] = moving
        except:
            print('bad robot data fit')

    #print('Finished experimental feature generation')
    return exp_feat_array


def get_kinematic_block(kin_df_, rat, kdate, session):
    """ Retrieves a single block (row) in a kinematic dataframe for given rat,date,session
        
    Args:
        kin_df_ (df): kinematic dataframe indexed by 'rat','date','session',dim
        rat (str): rat ID
        kdate (str): block date in 'kin_df_'
        session (str): block session
        
    Returns:
        kin_block_df (df): desired block row in 'kin_df_' indexed by rat,date,session,dim
    
    Raises:
        LookupError: If desired block does not exist
    """
    try:
        kin_block_df = kin_df_[kin_df_.index.get_level_values('rat') == rat]
        kin_block_df = kin_block_df[kin_block_df.index.get_level_values('date') == kdate]
        kin_block_df = kin_block_df[kin_block_df.index.get_level_values('session') == session]
        return kin_block_df
    except:
        raise LookupError('Not in kinematic dataframe : Trial ' + rat + " " + kdate + " " + session) from None


def make_s_f_trial_arrays_from_block(kin_block_df, exp_block_df, et, el, wv=5, window_length=250,
                                     pre=10):
    """ Returns trialized and formatted features from kinematic and experimental data
    
    Args:
        kin_block_df (df): kinematic DateFrame block indexed by rat,date,session,dim
        exp_block_df (df): robot experimental DataFrame block
        et (int): coordinate change variable
            Will take the positional coordinates and put them into the robot reference frame.
        el (int): coordinate change variable
            Will take the positional coordinates and put them into the robot reference frame.
        wv (int): the wavelet # for the median filter applied to the positional data (default 5)
        window_length (int): trial splitting window length, the number of frames to load data from(default 250)
            Set to 4-500. 900 is too long.
        pre (int): pre cut off before a trial starts, the number of frames to load data from before start time
            For trial splitting, set to 10. 50 is too long. (default 10)
    
    Returns:
        _hot_vector (array): one hot array of robot block data of length num trials
        _tt1 (nested array of ints): trialized kinematic data
            shape (2, NumTrials, 3 for XYZ positions or prob123, NumFeatures, NumFrames).
        feature_names_list (list of str): list of feature names from 'kin_block_df'
        exp_features (list of arrays): experimental features
            shape (Num trials X Features X 'pre'+'window_length).

    """
    # extract posture arrays, and feature names
    block_pos_arr, feature_names_list = block_pos_extract(kin_block_df, et, el, wv)

    # get starting_frames_list column value, which is a list of ints corresponding to video frame numbers
    start = exp_block_df['r_start'].values[0]

    # trial-ize kinematic data
    _tt1 = split_trial(block_pos_arr, start, window_length, pre)

    # swap pos and prob values for trialized kin data (corrects data mismatch)
    pos_data = _tt1[0]
    prob_data = _tt1[1]
    _tt1 = merge_in_swap(pos_data, prob_data, plot=False)

    # format exp features
    exp_features = import_experiment_features(exp_block_df, start, window_length, pre)

    # One hot exp data
    _hot_vector = onehot(exp_block_df)

    # Finished trial splitting
    return _hot_vector, _tt1, feature_names_list, exp_features

################################
# 4. Create ML and Feature arrays
################################


def match_stamps(kinematic_array_, label_array, exp_array_):
    """ Matches kinematic and experimental data with labeled trial vectors.

    Args:
        kinematic_array_ (nested array of ints): trialized kinematic data returned from make_s_f_trial_arrays_from_block
            shape (2, NumTrials, 3 for XYZ positions or prob123, NumFeatures, NumFrames).
        label_array (list of list of ints and str): vectorized DLC video labels returned from make_vectorized_labels
            shape (num labeled trials x 9 labeling features).
        exp_array_(list of arrays): experimental features returned from make_s_f_trial_arrays_from_block
            shape (Num trials X Features X Length)

    Returns:
        mk_array (arr of lists of ints): formatted kin data
            shape (2, Num labeled Trials, 3 for XYZ positions or prob123, NumFeatures, NumFrames).
        ez_array (arr of lists of ints): formatted exp data
            shape (Num labeled trials X Features X Length)

    Examples:
        >>> kinematic_array_.shape, exp_array_.shape
         (2, 26, 3, 27, 6) (26, 12, 6)
        >>> len_array
        19
        >>> mk_array.shape, ez_array.shape
        (2, 19, 3, 27, 6) (19, 12, 6)
    """
    # Create trial arrays
    len_array = len(label_array)
    try:
        # (2 for XYZ and prob123, num labeled trials, 3 for XYZ or prob123, kin features, window_length+pre)
        mk_array = np.empty((2, len_array, kinematic_array_[0].shape[1], kinematic_array_[0].shape[2],
                             kinematic_array_[0].shape[3]))

    except:
        print('No Kinematic Data to Match')

    try:
        # (num labeled trials, num exp features, window_length+pre)
        ez_array = np.empty((len_array, exp_array_.shape[1], exp_array_.shape[2]))
    except:
        print('Bad Robot Data Pass')

    # From label array, first column states the trial # of the row
    for dr, tval in enumerate(label_array):

        # iterate over the index and value of each trial element
        ixd = int(label_array[dr][0])  # get trial

        # reshape kin and exp data
        try:
            mk_array[:, ixd, :, :, :] = kinematic_array_[:, ixd, :, :, :]
            ez_array[ixd, :, :] = exp_array_[ixd, :, :]
        except:
            print('Cant set kinematic and robot arrays in place')

    return mk_array, ez_array


def create_ML_array(matched_kinematics_array, matched_exp_array):
    """ Reshapes and combines kin and experimental features.

    Args:
        matched_kinematics_array (arr of lists of ints): matched kin data returned by match_stamps
            shape (2 for XYZ and prob123, num labeled trials, 3, num feat, window_length+pre)
        matched_exp_array (arr of lists of ints): matched exp data returned by match_stamps
            shape shape (Num labeled trials X Num exp Features X window_length+pre)

    Returns:
        kin_exp_XYZ (array of lists of ints): reshaped ML arrays for kinematics and experimental data
            shape (num labeled trials, 3*num kin feat, window_length+pre)
        kin_exp_prob123: same as 'kin_exp_XYZ' except with probability values
             shape (num labeled trials, 3*num kin feat + num exp features, window_length+pre)

    """
    num_feat_times_3 = matched_kinematics_array.shape[2] * matched_kinematics_array.shape[3]
    length = matched_kinematics_array.shape[4]
    num_labeled_trials = matched_kinematics_array.shape[1]

    # reshape from shape (2 for XYZ and prob123, num labeled trials, 3, num feat, window_length+pre)
    #    into shape (2 for XYZ and prob123, num labeled trials, 3*num feat, window_length+pre)
    reshaped_kin = matched_kinematics_array.reshape(2, num_labeled_trials, num_feat_times_3, length)
    kin_XYZ = reshaped_kin[0]
    kin_prob123 = reshaped_kin[1]

    # append kin and exp array features
    #   rm exp from kin because redundant exp feat when concat with prob later
    kin_exp_XYZ = kin_XYZ # np.concatenate((kin_XYZ, matched_exp_array), axis=1)
    kin_exp_prob123 = np.concatenate((kin_prob123, matched_exp_array), axis=1)

    return np.array(kin_exp_XYZ), np.array(kin_exp_prob123)


def stack_ML_arrays(list_of_k, list_of_f):
    """ Vertically stacks ML arrays and DLC video labeling vectors for each block.

    Args:
        list_of_k (list of nested arrays of ints): list of ML arrays for each labeled block
            return value of create_ML_array
            shape (num labeled trials, 3*num kin feat + num exp features, window_length+pre)
        list_of_f (list of lists of lists of int): list of vectorized DLC video trial labels for each block
            return value of make_vectorized_labels
            shape (num labeled trials, 3*num kin feat + num exp features window_length+pre)

    Returns:
        ogk (array of nested arrays of ints): final_ML_array
            shape (total num labeled trials, 3*num kin feat+num exp features=list_of_k.shape[1], window_length+pre)
        ogf (array of lists of ints): final_feature_array
            shape (total num labeled trials, 9 for DLC label features)

    Examples:
        >>>final_ML_array, final_feature_array
            = CU.stack_ML_arrays([c, c1, c2, c3, c4],[labellist, elists, l18l, nl1lists, nl2lists])
        >>>print(final_ML_array.shape, final_feature_array.shape)
            (175, 93, 6) (175, 9) # 93 because 3 for XYZ *27 kin feat + 12 exp feat
    """
    for idd, valz in enumerate(list_of_k):
        if idd == 0:
            ogk = valz
            ogf = list_of_f[idd]
        else:
           ogk = np.vstack((ogk, valz))
           ogf = np.vstack((ogf, list_of_f[idd]))

    return ogk, ogf


###########################
# Generate Features and Aid Classification Functions
##########################

def is_tug_no_tug(moving_times):
    """
    Function to classify trials with post-reaching behavior from well-behaved handle release.
    Gives a simple estimate of if theres tug of war or not
    """
    # ask if there is robot velocity after a trial ends (for around a second)
    reward_end_times = np.argwhere(moving_times == 1)[0]  # take the first index when a robot movement command is issued
    movement_end_times = np.argwhere(moving_times == 1)[-1]
    # Handle Probability thresholding
    # post_trial_robot_movement = robot_vector[:,:,reward_end_times:reward_end_times+100] # get all values from end of trial to +100
    move_time = 20  # parameter threshold needed to evoke TOW
    # Histogram total results of this parameter
    if movement_end_times - reward_end_times > move_time:
        tug_preds = 1  # tug of war
    else:
        tug_preds = 0  # no tug of war
    return tug_preds, movement_end_times - reward_end_times


def is_reach_rewarded(lick_data_):
    """
    Function to simply classify trials as rewarded with water or not using sensor data from ReachMaster (lick detector).
    Tells if the reach was rewarded or not
    """
    rew_lick_ = 0
    if lick_data_.any():
        try:
            if np.where(lick_data_ == 1) > 3:  # filter bad lick noise
                rew_lick_ = 1
        except:
            rew_lick_ = 0
    return rew_lick_


#####
# Functions below generate some features with the positional data
#####

def right_arm_vector_from_kinematics(tka):
    """
    """
    # shape of [N, 27,3] Win
    tz = tka[:, :, 15:27, :]
    return tz


def left_arm_vector_from_kinematics(tka):
    # shape of [N, 27,3, Win
    tz = tka[:, :, 3:15, :]
    return tz


def left_int_position(left_arm):
    left_int_pos = np.mean(left_arm[:, :, :, :], axis=2)  # shape Trials x dimensions x frames
    return left_int_pos


def left_int_velocity(left_int_pos, time_vector=False):
    int_vel = np.diff(left_int_pos, axis=2)
    if time_vector:
        int_vel = int_vel / np.diff(time_vector)  # dx /dt
    return int_vel


def left_hand(tka):
    lh = tka[:, :, 6:15, :]
    return lh


def left_skeleton(tka):
    left_skeleton = tka[:, :, 3:6, :]
    return left_skeleton


def right_skeleton(tka):
    right_skeleton = tka[:, :, 15:18, :]
    return right_skeleton


def right_skeleton_velocity(right_skeleton_):
    right_skeleton_velocity = np.diff(right_skeleton_, axis=2)
    return right_skeleton_velocity


def create_arm_feature_arrays_trial(a, e_d, p_d, ii, left=False, hand=False):
    """
    Function to create arm and hand objects from DLC-generated positional data on a per-trial basis.
    Inputs:
    a : Array size [Trials, Dims, BParts, WLen + Pre) 
    e_d : (Trials, 12,  WLen+Pre)
    p_d : (Trials, Dims, BParts, WLen+Pre)
    Outputs:
    a_ : unique type of feature vector (Trials, Dims, features, WLen+Pre)
    """
    # pdb.set_trace()
    if left:
        if hand:
            a_ = a[ii, :, 7:15, :]
        else:
            a_ = a[ii, :, 4:7, :]  # sum over shoulder, forearm, palm, wrist
    else:
        if hand:
            a_ = a[ii, :, 19:27, :]
        else:
            a_ = a[ii, :, 16:19, :]  # sum over shoulder, forearm, palm, wrist
    # pdb.set_trace()
    #for tc in range(0, 3):
        #a_[:, tc, :] = prob_mask_segmentation(p_d[ii, :, :, :], a_[:, tc, :])  # threshold possible reach values
    # pdb.set_trace()
    return a_


#########################
# Classification_Structure helpers
#########################

def norm_and_zscore_ML_array(ML_array, robust=False, decomp=False, gauss=False):
    """ Unused. Replaced by sklearn pipeline.
    default preprocessing is simple MinMax L1 norm
    Args:
        ML_array : array shape : (Cut Trials, Features, Frames)   where Cut Trials refers to either the number of Trials
        inside the testing data or training data (Don't call this function for just the total ML data, split beforehand..)
        robust: boolean flag, use sci-kit learn robust scaling to normalize our data
        decomp : boolean flag, post-processing step used to return first whitened 20 PCA components to remove linear dependence
        gauss : boolean flag, use sci-kit learn gaussian distribution scaling to normalize our data

    Returns:
         r_ML_array (2d array): shape (cut trials, num features * frames)
    """
    # ML_array
    if robust:
        pt = preprocessing.robust_scale()
    elif gauss:
        pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    else:
        pt = preprocessing.MinMaxScaler()

    # reshape to be number cut trials by (num features * frames)
    r_ML_array = pt.fit_transform(ML_array.reshape(ML_array.shape[0], ML_array.shape[1] * ML_array.shape[2]))
    # apply normalization to feature axis
    if decomp:  # used to decomp linear correlations, if they exist.
        pca = decomposition.PCA(n=20, whiten=True)
        r_ML_array = pca.fit(r_ML_array)

    return r_ML_array


def split_ML_array(Ml_array, labels_array, t=0.2):
    X_train, X_test, y_train, y_test = train_test_split(Ml_array, labels_array, test_size=t, random_state=0)
    return X_train, X_test, y_train, y_test


def onehot_nulls(type_labels_):
    # kwargs: n_f_fr_s_st: Trial type (null, failed, failed_rew,s ,succ_tug), label key [0, 1, 2, 3, 4]
    null_labels = np.zeros((type_labels_.shape[0]))
    null_labels[np.where(type_labels_ == 0)] = 1  # 1 if null, 0 if real trial
    return null_labels


def onehot_num_reaches(num_labels_):
    num_r_labels = np.zeros((num_labels_.shape[0]))  # 0 vector
    num_r_labels[np.where(num_labels_ > 1)] = 1  # 0 if <1, 1 if > 1 reaches
    return num_r_labels


def hand_type_onehot(hand_labels_, simple=True):
    # 'lr': 2 , 'l' : 1, 'bi' : 3 , 'lbi' : 4, 'r': 0,
    hand_type_label = np.zeros((hand_labels_.shape[0]))
    if simple:
        hand_type_label[np.where(hand_labels_ > 1)] = 1  # classify all non r,l reaches as 1
    else:
        hand_type_label[np.where(hand_labels_ > 2)] = 1  # classify 0 as r/l
    return hand_type_label


def get_ML_labels(fv):
    # #[int trial_num, int start, int stop,
    # int trial_type, int num_reaches, str which_hand_reach, str tug_noTug, int hand_switch, int num_frames]
    # shape (Trials, 9 ^)
    # TODO convert to sklearn one hot encoder?
    fv = fv[:, 3:-1]  # take label catagories necessary for trial classification
    type_labels = onehot_nulls(fv[:, 0])  # labels for trial type
    num_labels = onehot_num_reaches(fv[:, 1])  # labels for num reaches in trial
    hand_labels = hand_type_onehot(fv[:, 2])  # labels for which hand
    tug_labels = fv[:, 3]  # labels for tug/no tug, tug = 1
    switch_labels = fv[:, 4]  # labels for hand switching (y/n), 1 is switch
    return [type_labels, num_labels, hand_labels, tug_labels, switch_labels]


def run_classifier(_model, _X_train, _X_test, input_labels):
    """
    Function for manually running a given model, intended for hard-code/troubleshooting.
    """
    _model.fit(_X_train, input_labels)
    type_pred = _model.predict(_X_test)
    type_feature_imp = pd.Series(_model.feature_importances_).sort_values(ascending=True)
    return [type_pred, type_feature_imp]


def do_constant_split(model_, ml, feature):
    """
    classification_structure helper
    Args:
        ml : ML-ready feature vector containing experimental and kinematic data
        feature : labels for each class (vectorized using blist and get_ML_labels)
        model_ : classifier (sk-Learn compatible)
    Returns: 
        cs: list of arrays of classifier predictions
        model_
    """
    classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model_)
    cs = []

    # generate correct labels for test/train labels
    X_train, X_test, y_train, y_test = split_ML_array(ml, feature, t=0.2)
    train_labels = get_ML_labels(y_train)

    # norm and z-score test/train features
    X_train = norm_and_zscore_ML_array(X_train, robust=False, decomp=False, gauss=False)
    X_test = norm_and_zscore_ML_array(X_test, robust=False, decomp=False, gauss=False)

    for i, vals in enumerate(train_labels):
        cs.append(run_classifier(model_, X_train, X_test, vals))
        # TODO ?
        # need to add cross_val_score for X_train,X_test splits
    return cs, model_


def simple_classification_verification(train_labels, classifier_pipeline, ml, feature, kFold, model_, LOO, X_train,
                                       X_test):
    """
    classification_structure helper
    else case for verification and clarity
    
    Returns: 
        preds: list of (5 arrays of 5 elems) arrays of classifier predictions 
        model_
    """
    preds = []
    for i, vals in enumerate(train_labels):  # loop over each layer of classifier, this just does classification
        try:
            if kFold:
                preds.append(cross_val_score(classifier_pipeline,
                                             ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]),
                                             get_ML_labels(feature)[i], cv=kFold))
            elif LOO:
                preds.append(
                    cross_val_score(classifier_pipeline, ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]),
                                    get_ML_labels(feature)[i], cv=ml.shape[0] - 10))
            else:  # simple classification
                preds.append(run_classifier(model_, X_train, X_test, vals))
                continue
        except:
            print('Bad Classifier Entry (Line 500)')
            # pdb.set_trace()
    try:
        print_preds(preds, train_labels)
    except:
        print('...')
    return preds, model_


def save_CV_score_to_preds(preds, classifier_pipeline, ml, feature, idx, kFold):
    """
    helper for structured_classification
    Appends return value of sklearn.model_selection.cross_val_score to preds
    
    Args:
        preds: list of predictions
        
    Returns:
        preds with array of scores of the estimator for each run of the cross validation appended
    """
    preds.append(cross_val_score(classifier_pipeline,
                                 ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]),
                                 get_ML_labels(feature)[idx], cv=kFold))
    return preds, classifier_pipeline


def structured_classification(ml, feature, model_,
                              X_train, X_test, y_train, y_test, train_labels, classifier_pipeline,
                              kFold, pred, disc, bal, conf):
    """
    classification_structure helper
    Performs classification for each level in hierarchy
    
    Variables:
        vals: input_labels into run_classifer fn
    Returns: 
        preds
        model_
    """
    preds = []
    for idx, vals in enumerate(train_labels):
        # check for important class, then train inputs
        if idx == 0:  # Reach vs Null

            #
            # predict null or reach trial type
            # update preds == ...
            # split_n_reaches(ml)
            # take out null trials, then pass data fwd to predict n reaches
            # for given 'ml_array_RM16.h5' data 12/24/2020
            # idx is 0, 1,2,3, 4
            # vals = input labels for run_classfifer()

            # Save ML predictions, models
            preds, classifier_pipeline = save_CV_score_to_preds(preds, classifier_pipeline, ml, feature, idx, kFold)

            # Plot in Yellowbrick
            visualizer = CVScores(classifier_pipeline, cv=kFold, scoring='f1_weighted')
            visualizer.fit(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx])
            visualizer.show()
            visualize_model(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx]
                            , classifier_pipeline, pred=pred, disc=disc, conf=conf, bal=bal)

        if idx == 1:  # num reaches, 1 vs >1

            #
            # predict num reaches=1 or >1
            # n_zero_ml_array = get_nreach_classification()
            # if trial contains > int x reaches,
            # then zero out all features and data in trial array like [ix, :, :, :]=0
            # send x< reach array to simple segmentation
            # get_simple_segments = ...
            # update and save ML predictions and models below
            # save model == add to pipeline?

            # Save ML predictions, models
            preds, classifier_pipeline = save_CV_score_to_preds(preds, classifier_pipeline, ml, feature, idx, kFold)

            # Plot in Yellowbrick
            visualizer = CVScores(classifier_pipeline, cv=kFold, scoring='f1_weighted')
            visualizer.fit(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx])
            visualizer.show()
            visualize_model(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx]
                            , classifier_pipeline, pred=pred, disc=disc, conf=conf, bal=bal)

        if idx == 2:  # which hand reaches: l/r vs lra,bi,rla

            #
            # for isx in range(0, ml_cut.shape[0]):
            # classify LR or [LRA, RLA, BI]
            # preds_arm1 = pred_arm(ml_cut[isx,:,:,:])
            # split ml_cut into classes
            # func input , uses pred_arm1 as indicies to split ml_cut data
            # ml_LR, ml+BRL = arm_split(ml_cut, preds_arm1)
            # continue
            # pred_arm2 = ...

            # Save ML predictions, models
            preds, classifier_pipeline = save_CV_score_to_preds(preds, classifier_pipeline, ml, feature, idx, kFold)

            # Plot in YellowBrick
            visualizer = CVScores(classifier_pipeline, cv=kFold, scoring='f1_weighted')
            visualizer.fit(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx])
            visualizer.show()
            visualize_model(ml.reshape(ml.shape[0], ml.shape[1] * ml.shape[2]), get_ML_labels(feature)[idx]
                            , classifier_pipeline, pred=pred, disc=disc, conf=conf, bal=bal)
    return preds, model_


def classification_structure(ml, feature, model_, kFold=False, LOO=False, PCA_data=False, constant_split=False,
                             structured=True,
                             plot_correlation_matrix=False, pred=False, disc=True, bal=True, conf=True):
    """
    Args:
        ml : ML-ready feature vector containing experimental and kinematic data
            Shape (Cut Trials, Features, Frames)
        feature : labels for each class (vectorized using blist and get_ML_labels)
        model_ : classifier (sk-Learn compatible)
        kFold : int, number of folds if using kFold cross-validation from sk-Learn
        LOO : boolean flag, set True if using LOO cross-validation from sk-Learn
        PCA : boolean flag, set True if using PCA to reduce dimensions of feature vectors
        constant_split : boolean flag, set True if comparing results between classifiers
        structured: boolean flag, set True to do multiple binary classifications

    Args for yellowbrick visualizations
        plot_correlation_matrix:
        pred:
        disc:
        bal:
        conf:
    Variables:
        X_train : ML_array : array shape : shape num cut trials by (num features * frames) by norm_z function
        X_test : ML_array : array shape : shape num cut trials by (num features * frames) by norm_z function
        y_train : array shape : (Num Trails, 9). dim 9 for
             1 int trial_num, 2 int start, 3 int stop,
             4 int trial_type, 5 int num_reaches,6 str which_hand_reach,
             7 str tug_noTug, 8 int hand_switch, 9 int num_frames
        y_test : array shape : (num Trails, 9).
        train_labels : ML labels from y_train data.
            Format: list of arrays of 0s and 1s, where each array corresponds to
               trial type, num reaches, reach with which hand, is tug, hand switch.
               Each arrays is of len num trials.
     Returns:
         preds: list of (3 arrays of 5 elems for each classifier in hierarchy) arrays of classifier predictions
         model_

      Notes:
        kfold boolean arg vs KFold for sklearn.model_selection._split.KFold
    """
    # split before norming to prevent bias in test data
    if constant_split:
        return do_constant_split(model_, ml, feature)

    # Create Classifier Pipeline Object in SciKit Learn
    if PCA_data:
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(),
                                            decomposition.PCA(n_components=int(PCA_data)), model_)
    else:
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model_)

    # For simple Classifier:
    X_train, X_test, y_train, y_test = split_ML_array(ml, feature, t=0.2)
    # generate correct labels for test/train labels
    train_labels = get_ML_labels(y_train)
    # norm and z-score test/train features
    X_train = norm_and_zscore_ML_array(X_train, robust=False, decomp=False, gauss=False)
    X_test = norm_and_zscore_ML_array(X_test, robust=False, decomp=False, gauss=False)

    # Feature Work
    if PCA_data:
        pcs = decomposition.PCA()
        X_train = pcs.fit(X_train)
        X_test = pcs.fit(X_test)
        for ii, mi in enumerate(pcs.explained_variance_ratio_[:].sum()):
            if mi > .99:
                n_comps = ii
        X_train = X_train[0:ii, :]
        X_test = X_test[0:ii, :]

    if plot_correlation_matrix:
        pearson_features(X_train)

    # Run classification hierarchy
    if structured:
        return structured_classification(ml, feature, model_,
                                         X_train, X_test, y_train, y_test, train_labels, classifier_pipeline,
                                         kFold, pred, disc, bal, conf)
    else:
        return simple_classification_verification(train_labels, classifier_pipeline, ml, feature, kFold, model_, LOO,
                                                  X_train, X_test)


###############################
# Classification Hierarchy
###############################
def remove_trials(X, Y, preds, toRemove):
    """
    Removes trials from labels after classification.
    Used to prepare data for next classification in hierarchy.
    Args:
        X (array): features, shape (num trials, num feat*num frames)
        Y (array): labels
        shape # type_labels_y_train, num_labels_y_train, hand_labels_y_train, tug_labels_y_train, switch_labels_y_train
        preds (array): classifier trial predictions
        toRemove: 0 to remove trials classified as 0, 1 otherwise

    Returns:
        X (array): filtered
        Y (array): filtered

    Notes:
        Preserves order of values
        Careful to remove values in X and corresponding Y labels for each class!
    """
    new_X = []
    new_Y = []
    trial_indices_X = len(X)
    # delete trials backwards
    # for each class of labels
    for y_arr in Y:
        i = trial_indices_X - 1
        new = []
        for _ in np.arange(trial_indices_X):
            if preds[i] != toRemove:
                new.append(y_arr[i])
            i = i - 1
        new_Y.append(new)

    # remove x trials
    j = trial_indices_X - 1
    for _ in np.arange(trial_indices_X):
        if preds[j] != toRemove:
            new_X.append(X[j])
        j = j - 1
    return np.array(new_X), np.array(new_Y)







###################################
# Convert Nested lists/arrays into pandas DataFrames
##################################
def load_kin_exp_data(kin_name, robot_name):
    """
    Loads data.
    Args:
        kin_name (str): file path to kinematic data
        robot_name (str): file path to robot experimental data

    Returns:
        kin (pandas DataFrame): kinematic data
        exp (pandas DataFrame): experimental data
    """
    # read and format kinematic data
    d = pkl_to_df(kin_name)

    # read robot experimental data
    hdf = pd.read_pickle(robot_name)
    hdf = hdf.reset_index(drop=True)

    return d, hdf


def save_to_hdf(file_name, key, data):
    """
    Saves data as HDF file in current working directory
    Args:
        file_name (str): name of saved file
        key (str): group identifier to access data in file
        data: data to save

    Returns: None

    Notes:
        Check permissions so do not overwrite previously written data
        Ensure proper open/closing so do not corrupt file.

    """
    # non DataFrame types
    if os.path.exists(file_name):
        hf = h5py.File(file_name, 'r+')
        if key in hf.keys():
            # update data
            hf[key][:] = data
        else:
            # add data to pre-existing file
            hf[key] = data
    else:
        # create file and add data
        hf = h5py.File(file_name, 'w')
        hf[key] = data


def load_hdf(file_name, key):
    """
    Loads a HDF file.
    Args:
        file_name: (str) name of file to load.
        key (str): group identifier to access data in file

    Returns: data in file.

    """
    read_file = h5py.File(file_name, 'r')
    return read_file[key][:]

def make_vectorized_labels_to_df(labels):
    """ Convert return value from make_vectorized_labels into a pandas df
    
    Args:
        labels (arr of lists): return value from make_vectorized_labels
    
    Returns:
        newdf(df)
    
    Examples:
        >>> l18l, ev18 = CU.make_vectorized_labels(l18)
        >>> make_vectorized_labels_to_df(l18l)
    """
    newdf = pd.DataFrame(data=labels,
                         columns=['Trial Num', 'Start Frame', 'Stop Frame', 'Trial Type',
                                  'Num Reaches', 'Which Hand', 'Tug', 'Hand Switch', 'Num Frames'])
    return newdf


def block_pos_extract_to_df(block, xzy_or_prob):
    """ Converts return value of block_pos_extract to pandas dataframe
        for kinematic data. 
        
    Args: 
        block (list of arr): return value of block_pos_extract 
        xzy_or_prob (int): 0 for x,y,z data, else 1 for probability 1,2,3 data
    
    Returns:
        block_df (df)
        
    Examples:
        >>> block_pos_arr, feature_names_list = CU.block_pos_extract(kin_block_df, et, el, wv)
        >>> block_pos_extract_to_df(block_pos_arr, 0) # to get XYZ data
    
    """
    # define column names
    pos_names = [ 'Handle', 'Back Handle', 'Nose', 
             'Left Shoulder', 'Left Forearm', 'Left Wrist', 'Left Palm', 'Left Index Base', 'Left Index Tip',
             'Left Middle Base', 'Left Middle Tip', 'Left Third Base',
             'Left Third Tip', 'Left Fourth Finger Base', 'Left Fourth Finger Tip', 
             'Right Shoulder', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Right Index Base',
             'Right Index Tip', 'Right Middle Base', 'Right Middle Tip', 'Right Third Base',
             'Right Third Tip', 'Right Fourth Finger Base','Right Fourth Finger Tip']
    
    # define index values
    if (xzy_or_prob):
        index = ['prob1', 'prob2', 'prob3']
    else:
        index = ['X', 'Y', 'Z']
    
    # create DataFrame
    block_df = pd.DataFrame(data=block[xzy_or_prob].tolist(),
                       index = index,
               columns=pos_names)
    return block_df


def import_experiment_features_to_df(exp_features):
    """ Converts return value of import_experiment_features to pandas dataframe.
    Args: 
        exp_features (nested arrays): return value of import_experiment_features
    
    Returns:
        exp_df (df)
        
    Examples:
        >>> exp_features = CU.import_experiment_features(r_block_df, start, window_length, pre)
        >>> import_experiment_features_to_df(exp_features)
        
    Notes:
        Each row is a trial. 
        unused columns 4,5,6.
    """
    exp_features = exp_features.tolist()
    exp_names = ['Robot Velocity X', 'Robot Velocity Y',
                 'Robot Velocity Z', "unused idx 4", "unused idx 5",
                 "unused idx 6", 'Reward Zone', 'Robot Position X',
                 'Robot Position Y', 'Robot Position Z', 'Licking', 'Moving']
    exp_df = pd.DataFrame(data=exp_features,

                          columns=exp_names)
    return exp_df


def split_kin_trial_to_df(trials_list, num_trials, xzy_or_prob):
    """Converts return value of split_trial to pandas dataframe.
    
    Args: 
        trials_list (nested arrays): return value of split_trial
        num_trials(int): number of trials in block 
            (same as len of start array value in exp data block)
        xzy_or_prob (int): 0 for x,y,z data, else 1 for probability 1,2,3 data
    
    Returns:
        trialized_df (df)
        
    Examples:
        >>>_tt18 = CU.split_trial(block_pos_arr, start, window_length, pre)
        >>> split_kin_trial_to_df(ttl18, num_labeled_trials, 1) # where num_labeled_trials = ttl18.shape[1]
            # gets probability data
        
    """
    # define column names
    Trials = np.arange(num_trials)
    pos_names = ['Handle', 'Back Handle', 'Nose',
                 'Left Shoulder', 'Left Forearm', 'Left Wrist', 'Left Palm', 'Left Index Base', 'Left Index Tip',
                 'Left Middle Base', 'Left Middle Tip', 'Left Third Base',
                 'Left Third Tip', 'Left Fourth Finger Base', 'Left Fourth Finger Tip',
                 'Right Shoulder', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Right Index Base',
                 'Right Index Tip', 'Right Middle Base', 'Right Middle Tip', 'Right Third Base',
                 'Right Third Tip', 'Right Fourth Finger Base', 'Right Fourth Finger Tip']
    if (xzy_or_prob):
        trials_list = trials_list[xzy_or_prob]
        pos =  ['prob1', 'prob2', 'prob3']
    else:
        trials_list = trials_list[xzy_or_prob]
        pos = ['X', 'Y', 'Z']

    # initialize temp dictionary
    d = {}
    for idx1 in range(len(trials_list)):  # num trials
        t_key1 = Trials[idx1]
        d[t_key1] = {}
        for idx2 in range(len(trials_list[0])):  # XYZ
            p_key2 = pos[idx2]
            d[t_key1][p_key2] = {}
            for idx3 in range(len(trials_list[0][0])):  # pos names
                p_key3 = pos_names[idx3]
                d[t_key1][p_key2][p_key3] = trials_list[idx1][idx2][idx3]  # array value

    # create df
    # reference: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    user_dict = d
    trialized_df = pd.DataFrame.from_dict({(i, j): user_dict[i][j]
                                           for i in user_dict.keys()
                                           for j in user_dict[i].keys()},
                                          orient='index')
    return trialized_df


def final_ML_array_to_df(final_ML_feature_array, feat_names):
    """ Converts return value of block_pos_extract to pandas dataframe.
    Args:
        final_ML_feature_array (list of arr):
            shape (total num labeled trials
            x (3*num kin feat)*2 +num exp feat = 174 for XYZ and prob, window_length+pre)
        feat_names (array of str): X,Y,Z then prob1,2,3 feature names for kinematic data
            return value of make_s_f_trial_arrays_from_block
    Returns:
        feat_df (df): rows are total number of trials

    Examples:
        >>> final_ML_feature_array_df = final_ML_array_to_df(final_ML_feature_array, featsl18)

    """
    # define column names
    exp_names = ['Robot Velocity X', 'Robot Velocity Y',
                 'Robot Velocity Z', "unused idx 4", "unused idx 5",
                 "unused idx 6", 'Reward Zone', 'Robot Position X',
                 'Robot Position Y', 'Robot Position Z', 'Licking', 'Moving']

    column_names = feat_names + exp_names  # order is XYZ, prob123, exp features

    # define index values
    index = np.arange(final_ML_feature_array.shape[0])  # len total num labeled trials

    # create DataFrame
    feat_df = pd.DataFrame(data=final_ML_feature_array.tolist(),
                           index=index,
                           columns=column_names)
    return feat_df

###############################
# 
###############################
def merge_in_swap(init_arm_array, ip_array, plot):
    """ Swaps XYZ position and prob123 kinematic data to correct data mismatch.
    prob values range from (0,1) and those position values range from ~ -0.1 to 0.3

    Args:
        init_arm_array (nested arr): CU.split_trial output of trailized XYZ position kinematic data
        ip_array (nested arr): CU.split_trial output of trailized prob123 kinematic data
        plot (bool): True to plot data ranges, False for no plotting

    Returns:
        trialized kinematic data with appropriate XYZ and prob values
         shape (2, NumTrials, 3 for XYZ positions or prob123, NumFeatures, NumFrames).

    Examples:
        >>>  _tt1 = split_trial(block_pos_arr, start, window_length, pre)
        >>>  _tt1 = merge_in_swap(_tt1[0], _tt1[1], plot=False)
    """
    # swap
    c = init_arm_array[:, :, 14:27, :]
    pc = ip_array[:, :, 0:13, :]
    init_arm_array[:,:,14:27,:] = pc
    ip_array[:,:,0:13,:] = c

    # plot to view ranges
    if plot:
        for trials in range(0, 5):
            plt.plot(init_arm_array[trials, 0, 7, :])
        plt.show()

        for trials in range(0, init_arm_array.shape[0] - 40):
            plt.plot(c[trials, 0, :, :])
        plt.show()

        for trials in range(0, init_arm_array.shape[0] - 40):
            plt.plot(pc[trials, 0, :, :])
        plt.show()
        # init_arm_array[:,:,14:27,:] = pc
        # ip_array[:,:,0:13,:] = c
        for trials in range(0, init_arm_array.shape[0] - 40):
            plt.plot(ip_array[trials, 0, 0:13, :])
        plt.show()
        for trials in range(0, 5):
            plt.plot(init_arm_array[trials, 0, 7, :])
        plt.show()
    return np.array([init_arm_array, ip_array])


def pearson_features(ml_array_):
    feat_visualizer = Rank2D(algorithm="pearson")
    feat_visualizer.fit_transform(ml_array_)
    feat_visualizer.show()


def is_tug_no_tug():
    """
    Function to classify trials with post-reaching behavior from well-behaved handle release.
    """
    # ask if there is robot velocity after a trial ends (for around a second)
    tug_preds = []
    return tug_preds


# def is_reach_rewarded(lick_data, start, stop):
#    """
#    Function to simply classify trials as rewarded with water or not using sensor data from ReachMaster (lick detector).

#    """
#    reward_vector = []
#    return reward_vector

#########################################
# DLC Video Labels (unprocessed)
#########################################
nl1 = [
    [1, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # single left rew tug
    [2, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # single left rew tug
    [3, 0, 0, 1, 1, 'rla', 'noTug', 0, 0],  # rl assist, 1 , notug, rew
    [4, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l single rew notug
    [5, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew no tug
    [6, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lra 2 rew tug
    [7, 0, 0, 1, 4, 'lra', 'noTug', 0, 0],  # lra 4 rew notug
    [8, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lra 1 rew notug
    [9, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [10, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [11, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 2 rew tug
    [12, 0, 0, 1, 3, 'lra', 'Tug', 0, 0],  # lra 3 rew tug
    [13, 0, 0, 1, 3, 'lra', 'Tug', 0, 0],  # lra 3 rew tug
    [14, 0, 0, 1, 1, 'bi', 'Tug', 0, 0],  # bi 1 rew tug
    [15, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [16, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [17, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 1 rew tug
    [19, 0, 0, 1, 2, 'bi', 'Tug', 0, 0],  # bi 2 rew tug
    [20, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [21, 0, 0, 1, 1, 'bi', 'Tug', 0, 0],  # bi 1 r t
    [22, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [23, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 1 r t
    [24, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [25, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 r t
    [26, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [18, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 r t
    [0, 0, 0, 1, 2, 'rla', 'Tug', 1, 0]  # rl 2 r t hand switch

]

l18 = [
    [1, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [2, 0, 0, 1, 3, 'l', 'noTug', 0, 0],  # l 3 rew notug
    [4, 0, 0, 1, 3, 'lr', 'noTug', 1, 0],  # lr 3 switching rew notug
    [5, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew no tug
    [6, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew no tug
    [7, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew notug
    [9, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug (check b4)
    [10, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew notug
    [12, 0, 0, 1, 1, 'lr', 'noTug', 0, 0],  # lr 1 rew notug
    [14, 0, 0, 1, 1, 'lr', 'Tug', 0, 0],  # lr 1 rew tug
    [15, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [17, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew notug
    [0, 0, 0, 1, 4, 'l', 'noTug', 0, 0],  # l 4 norew notug
    [3, 0, 0, 1, 4, 'lra', 'noTug', 1, 0],  # lr 4 switching norew notug
    [8, 0, 0, 1, 11, 'lra', 'noTug', 1, 0],  # lr 11 switching norew notug
    [11, 0, 0, 1, 7, 'l', 'noTug', 0, 0],  # l 7 norew notug
    [13, 0, 0, 1, 7, 'l', 'noTug', 0, 0],  # l 7 norew notug
    [16, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 norew notug
    [18, 0, 0, 1, 6, 'l', 'noTug', 0, 0]
]

nl2 = [
    [1, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [2, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [3, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lr 2 rew tug
    [4, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [5, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [6, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lr 2 rew tug
    [7, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [8, 0, 0, 1, 2, 'l', 'Tug', 0, 0],  # l 2 rew tug
    [9, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew tug
    [10, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [11, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [12, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew nt
    [13, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew nt
    [14, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [15, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [16, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew nt
    [17, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew nt
    [18, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [19, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [20, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [21, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [22, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [23, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [24, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [25, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [26, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [27, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # lr 1 rew t
    [28, 0, 0, 1, 6, 'lra', 'noTug', 1, 0],  # lr 6 handswitch rew nt
    [30, 0, 0, 1, 15, 'lra', 'Tug', 0, 0],  # lr 15 rew t
    [31, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [32, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew nt
    [33, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew nt
    [34, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew t
    [35, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [36, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew t
    [0, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [29, 0, 0, 1, 3, 'l', 'noTug', 0, 0],  # l 3 nr nt
    [37, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
]

blist1 = [
    [1, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  # succ tugg, left hand, single reach,
    , [2, 1, 1, 3, 1, 'l', 'noTug', 0, 0]  # left handed reach, no tug of war, 1 reach, no switch,
    , [3, 1, 1, 3, 2, 'bi', 0, 1, 0]
    , [4, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  # tries to grab handle when moving
    , [6, 1, 1, 3, 1, 'l', 'Tug', 0, 0]  # reaching after handle moves but can't grasp
    , [7, 1, 1, 2, 2, 'l', 0, 0, 0]  # A mis-label!!
    , [8, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [10, 1, 1, 3, 2, 'l', 'Tug', 0, 0]
    , [17, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  #
    , [18, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0]  # added lbi for multiple reaching arms
    , [19, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0]  # lbi
    , [20, 1, 1, 4, 3, 'l', 'Tug', 0, 0]
    , [21, 1, 1, 4, 2, 'lbi', 'Tug', 0, 0]
    , [22, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [23, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [24, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [25, 1, 1, 2, 4, 'l', 0, 0, 0]
    , [26, 1, 1, 3, 3, 'lbi', 'noTug', 1, 0]
    , [27, 1, 1, 4, 1, 'lbi', 'Tug', 0, 0]
    , [28, 1, 1, 3, 2, 'l', 'noTug', 0, 0]
    , [30, 1, 1, 3, 2, 'l', 'noTug', 0, 0]
    , [31, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [32, 1, 1, 2, 4, 'l', 0, 0, 0]
    , [33, 1, 1, 4, 2, 'lr', 'noTug', 1, 0]
    , [34, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [35, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [37, 1, 1, 4, 1, 'lbi', 'Tug', 0, 0]
    , [38, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [39, 1, 1, 3, 3, 'l', 'noTug', 0, 0]
    , [40, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [43, 1, 1, 3, 1, 'bi', 'noTug', 0, 0]
    , [44, 1, 1, 4, 2, 'lbi', 'Tug', 0, 0]
    , [45, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [46, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [47, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [48, 1, 1, 3, 2, 'lbi', 'Tug', 0, 0]
    , [0, 1, 1, 0, 0, 0, 0, 0, 0]
    , [5, 1, 1, 2, 4, 'lr', 0, 0, 0]
    , [9, 1, 1, 1, 3, 'l', 0, 0, 0]
    , [11, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [12, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [13, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [14, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [15, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [16, 1, 1, 0, 0, 0, 0, 0, 0]
    , [29, 1, 1, 1, 4, 'l', 0, 0, 0]
    , [36, 1, 1, 1, 9, 'llr', 'Tug', 1, 0]  # lots of stuff going on here
    , [41, 1, 1, 1, 6, 'l', 0, 0, 0]
    , [42, 1, 1, 1, 8, 'llr', 'Tug', 1, 0]
    , [49, 1, 1, 2, 4, 'lr', 0, 0, 0]]

elist = [
    [0, 1723, 2284, 0, 0, 0, 'no_tug', 0, 30]  # null
    , [1, 5593, 6156, 0, 0, 0, 'no_tug', 0, 27]  # null
    , [2, 7866, 8441, 3, 2, 'l', 'no_tug', 0, 14]  # success
    , [3, 8873, 9426, 1, 7, 'l', 'no_tug', 0, 20]  # failed
    , [4, 10101, 10665, 1, 3, 'l', 'no_tug', 0, 15]  # failed
    , [5, 12962, 13524, 1, 8, 'l', 'no_tug', 0, 27]  # failed

    , [6, 14760, 15351, 3, 2, 'bi', 'no_tug', 1, 25]  # success ## bi # starts mid reach
    , [7, 15802, 16431, 3, 3, 'bi', 'no_tug', 1, 30]  # success ## bi # starts mid reach # post reaching activity
    , [8, 17400, 17964, 1, 3, 'l', 'no_tug', 0, 13]  # failed # starts mid reach
    , [9, 18923, 19485, 3, 4, 'l', 'no_tug', 0, 19]  # success
    , [10, 20044, 20604, 1, 5, 'l', 'no_tug', 0, 6]  # failed
    , [11, 24406, 24969, 1, 1, 'l', 'no_tug', 0, 6]  # failed # ends mid reach
    , [12, 26962, 27521, 3, 1, 'l', 'no_tug', 0, 5]  # success # starts mid reach
    , [13, 27980, 28536, 1, 12, 'l', 'no_tug', 0, 18]  # failed # ends mid reach # lots of reaches
    , [14, 29034, 29596, 3, 6, 'bi', 'no_tug', 1, 13]  # success ## bi
    , [15, 30106, 30665, 3, 1, 'l', 'no_tug', 0, 8]  # success # starts mid reach
    , [16, 38998, 39591, 1, 2, 'l', 'no_tug', 0, 4]  # failed
    , [17, 40033, 40594, 0, 0, 0, 'no_tug', 0, 32]  # null
    , [18, 45355, 45914, 3, 7, 'l', 'no_tug', 0, 6]  # success
    , [19, 46845, 47405, 3, 1, 'l', 'no_tug', 0, 7]  # success

    , [20, 50359, 50949, 3, 1, 'l', 'no_tug', 1, 8]  # success # post reaching activity with r
    , [21, 58229, 58793, 3, 2, 'l', 'tug', 1, 12]
    # success # post reaching activity with r # rat lets handle go before in reward zone
    , [22, 59596, 60427, 3, 2, 'l', 'no_tug', 0, 9]  # success
    , [23, 60903, 61466, 3, 1, 'l', 'no_tug', 0, 4]  # success
    , [24, 62233, 62790, 3, 2, 'l', 'tug', 0, 10]  # success # rat lets handle go before in reward zone
    , [25, 66026, 66600, 1, 9, 'l', 'no_tug', 0, 27]
    # classifed as success in py notebook, but is failed trial # ends mid reach
    , [26, 67473, 68046, 3, 1, 'l', 'no_tug', 1, 7]  # success # post reaching activity with r
    , [27, 68689, 69260, 3, 2, 'bi', 'no_tug', 1, 9]  # success # bi
    , [28, 70046, 70617, 3, 2, 'bi', 'no_tug', 1, 5]  # success # bi # starts mid reach

    , [29, 71050, 71622, 3, 11, 'bi', 'tug', 1, 7]
    # success # bi # starts mid reach # rat lets handle go before in reward zone # lots of reaches
    , [30, 72914, 73501, 3, 1, 'l', 'no_tug', 0, 10]  # success
    , [31, 74777, 75368, 3, 3, 'bi', 'no_tug', 1, 9]  # success # bi # post reaching activity with r
    , [32, 81538, 82106, 3, 9, 'l', 'no_tug', 1, 13]  # success # post reaching activity with r
    , [33, 82534, 83114, 3, 4, 'bi', 'tug', 1, 12]
    # success ## bi # starts mid reach # rat lets handle go before in reward zone # includes uncommon failed bi reach
    , [34, 83546, 84118, 3, 2, 'l', 'no_tug', 1, 4]  # success # starts mid reach # post reaching activity with r
    , [35, 85563, 86134, 3, 2, 'l', 'no_tug', 1, 5]  # success # starts mid reach # post reaching activity with r
    , [36, 86564, 87134, 1, 13, 'l', 'no_tug', 0, 5]  # fail # lots of reaches
    , [37, 87574, 88173, 3, 7, 'l', 'no_tug', 1, 8]  # success # post reaching activity with r
    , [38, 89012, 89584, 3, 4, 'bi', 'tug', 1, 5]
    # success ## bi # rat lets handle go before in reward zone # includes uncommon reach with r first then left in bi reach

    , [39, 90738, 91390, 3, 7, 'l', 'no_tug', 1, 9]  # success # post reaching activity with r
    , [40, 91818, 92387, 1, 7, 'l', 'no_tug', 0, 6]  # fail # starts mid reach
]
