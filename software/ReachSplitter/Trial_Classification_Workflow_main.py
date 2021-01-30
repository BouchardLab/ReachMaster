"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 12/8/2020
               Emily Nguyen, UC Berkeley

    This code is intended to create and implement structure supervised classification of coarsely
    segmented trial behavior from the ReachMaster experimental system.
    Functions are designed to work with a classifier of your choice.

    Edited: 12/8/2020
"""
import argparse
import os.path

from networkx.drawing.tests.test_pylab import plt
from scipy import ndimage
import pickle
from Analysis_Utils import preprocessing_df as preprocessing
from Analysis_Utils import query_df
import DataStream_Vis_Utils as utils
import Classification_Utils as CU
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Classification_Visualization import visualize_models
import numpy as np
import h5py


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
    d = CU.pkl_to_df(kin_name)

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

    Notes: check permissions
        so do not overwrite previously written data

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


#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.question == 1:
        # vectorize DLC labels into ML ready format
        # make_vectorized_labels returns list: vectorized list of labels,
        # e: vectorized array of reading indices (unused variables)
        elists, ev = CU.make_vectorized_labels(CU.elist)
        labellist, edddd = CU.make_vectorized_labels(CU.blist1)
        nl1lists, ev1 = CU.make_vectorized_labels(CU.nl1)
        nl2lists, ev2 = CU.make_vectorized_labels(CU.nl2)
        l18l, ev18 = CU.make_vectorized_labels(CU.l18)

        # save each vectorized label
        vectorized_labels = [elists, labellist, nl1lists, nl2lists, l18l]
        file_name = "vectorized_labels"
        key_names = ['elists', 'labellist', 'nl1lists', 'nl2lists', 'l18l']
        for i in np.arange(len(vectorized_labels)):
            save_to_hdf(file_name, key_names[i], vectorized_labels[i])


    elif args.question == 2:
        # load kinematic and experimental data
        kin_df, exp_df = load_kin_exp_data('tkd16.pkl', 'experimental_data.pickle')

        # get blocks
        #   rat (str): rat ID
        #   date (str): block date in robot_df_
        #   kdate (str): block date in kin_df_
        #   session (str): block session
        block_names = [
            ['RM16', '0190920', '0190920', 'S3'],
            ['RM16', '0190919', '0190919', 'S3'],
            ['RM16', '0190917', '0190917', 'S2'],
            ['RM16', '0190917', '0190917', 'S1'],
            ['RM16', '0190918', '0190918', 'S1']
        ]
        kin_blocks = []
        exp_blocks = []
        for i in np.arange(len(block_names)):
            # get blocks
            rat, kdate, date, session = block_names[i]
            kin_block_df = CU.get_kinematic_block(kin_df, rat, kdate, session)
            exp_block_df = utils.get_single_block(exp_df, date, session, rat)

            # append blocks
            kin_blocks.append(kin_block_df)
            exp_blocks.append(exp_block_df)

        # save kinematic blocks
        file_name = 'kin_block'
        for i in np.arange(len(block_names)):
            rat, kdate, date, session = block_names[i]
            key = rat + kdate + session
            kin_blocks[i].to_pickle(file_name + "_" + key)

        # save experimental blocks
        file_name = 'exp_block'
        for i in np.arange(len(block_names)):
            rat, kdate, date, session = block_names[i]
            key = rat + date + session
            exp_blocks[i].to_pickle(file_name + "_" + key)


    elif args.question == 3:
        # load vectorized labels
        l18l = load_hdf("vectorized_labels", 'l18l')
        nl1lists = load_hdf("vectorized_labels", 'nl1lists')
        elists = load_hdf("vectorized_labels", 'elists')
        labellist = load_hdf("vectorized_labels", 'labellist')
        nl2lists = load_hdf("vectorized_labels", 'nl2lists')

        # load saved block pickles
        exp_block_df = pd.read_pickle('exp_block_RM160190920S3')
        kin_block_df = pd.read_pickle('kin_block_RM160190920S3')

        exp_block_df3 = pd.read_pickle('exp_block_RM160190919S3')
        kin_block_df3 = pd.read_pickle('kin_block_RM160190919S3')

        exp_block_df1 = pd.read_pickle('exp_block_RM160190917S1')
        kin_block_df1 = pd.read_pickle('kin_block_RM160190917S1')

        exp_block_df2 = pd.read_pickle('exp_block_RM160190917S2')
        kin_block_df2 = pd.read_pickle('kin_block_RM160190917S2')

        exp_block_df18 = pd.read_pickle('exp_block_RM160190918S1')
        kin_block_df18 = pd.read_pickle('kin_block_RM160190918S1')

        # define params
        et = 0
        el = 0
        wv = 5
        window_length = 4
        pre = 2

        # trial-ize data
        hot_vector, tt, feats, e \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df, exp_block_df, et, el, wv, window_length, pre)
        hot_vector3, tt3, feats3, e3 \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df3, exp_block_df3, et, el, wv, window_length,
                                                  pre)  # Emily label trial list
        hot_vectornl1, ttnl1, featsnl1, enl1 \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df1, exp_block_df1, et, el, wv, window_length, pre)
        hot_vectornl2, ttnl2, featsnl2, enl2 \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df2, exp_block_df2, et, el, wv, window_length, pre)
        hot_vectorl18, ttl18, featsl18, el18 \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df18, exp_block_df18, et, el, wv, window_length, pre)

        # Match with trial labeled vectors
        # match_stamps args: kin array, label array, exp array
        matched_kin_b1, ez1 = CU.match_stamps(tt, labellist, e)  # Matched to blist1
        matched_kin_e1, ez4 = CU.match_stamps(tt3, elists, e3)  # Matched to blist4
        matched_kin_b1nl1, eznl1 = CU.match_stamps(ttnl1, nl1lists, enl1)  # Matched to blist1
        matched_kin_e1l18, ezl18 = CU.match_stamps(ttl18, l18l, el18)  # Matched to blist4
        matched_kin_b1nl2, eznl2 = CU.match_stamps(ttnl2, nl2lists, enl2)  # Matched to blist1

        # match kin and exp features
        # create_ML_array args: matched kin array, matched ez array
        c, c_prob = CU.create_ML_array(matched_kin_b1, ez1)
        c1, c1_prob = CU.create_ML_array(matched_kin_e1, ez4)
        c2, c2_prob = CU.create_ML_array(matched_kin_e1l18, ezl18)
        c3, c3_prob = CU.create_ML_array(matched_kin_b1nl1, eznl1)
        c4, c4_prob = CU.create_ML_array(matched_kin_b1nl2, eznl2)

        # Create final ML arrays
        final_ML_feature_array_XYZ, final_labels_array \
            = CU.stack_ML_arrays([c, c1, c2, c3, c4],
                                 [labellist, elists, l18l, nl1lists, nl2lists])
        final_ML_feature_array_prob, _ \
            = CU.stack_ML_arrays([c_prob, c1_prob, c2_prob, c3_prob, c4_prob],
                                 [labellist, elists, l18l, nl1lists, nl2lists])

        # concat horizontally XYZ and prob123 ml feature arrays
        # (total num labeled trials x (3*num kin feat)*2 +num exp feat = 174 for XYZ and prob, window_length+pre)
        final_ML_feature_array = np.concatenate((final_ML_feature_array_XYZ, final_ML_feature_array_prob), axis=1)

        print(final_ML_feature_array.shape, featsl18)

        # Save final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'w') as hf:
            hf.create_dataset("RM16", data=final_ML_feature_array)
            hf.create_dataset("RM16_labels", data=final_labels_array)

    elif args.question == 4:
        # Load final_ML_array and final_feature_array in h5 file
        f = h5py.File('ml_array_RM16.h5', 'r')
        final_ML_feature_array = f['RM16'][:]
        final_labels_array = f['RM16_labels'][:]


    # elif args.question == 5:
    # main_q5()
    else:
        raise ValueError("Cannot find specified question number")
