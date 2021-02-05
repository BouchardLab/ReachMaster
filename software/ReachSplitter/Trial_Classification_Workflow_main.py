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

from sklearn.ensemble import RandomForestClassifier

from Analysis_Utils import preprocessing_df as preprocessing
from Analysis_Utils import query_df
import DataStream_Vis_Utils as utils
import CU_SCRIPT.Classification_Utils as CU
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Classification_Visualization import visualize_models
import numpy as np
import h5py

# classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
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
            CU.save_to_hdf(file_name, key_names[i], vectorized_labels[i])


    elif args.question == 2:
        # load kinematic and experimental data
        kin_df, exp_df = CU.load_kin_exp_data('tkd16.pkl', 'experimental_data.pickle')

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
        l18l = CU.load_hdf("vectorized_labels", 'l18l')
        nl1lists = CU.load_hdf("vectorized_labels", 'nl1lists')
        elists = CU.load_hdf("vectorized_labels", 'elists')
        labellist = CU.load_hdf("vectorized_labels", 'labellist')
        nl2lists = CU.load_hdf("vectorized_labels", 'nl2lists')

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
        # TODO dont do this...
        final_ML_feature_array = np.concatenate((final_ML_feature_array_XYZ, final_ML_feature_array_prob), axis=1) # this causes issues with norm/zscore


        #print(final_ML_feature_array.shape, featsl18)

        # Save final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'w') as hf:
            hf.create_dataset("RM16_features", data=final_ML_feature_array)
            hf.create_dataset("RM16_labels", data=final_labels_array)

    elif args.question == 4:
        # Load final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'r') as f:
            final_ML_feature_array = f['RM16_features'][:]
            final_labels_array = f['RM16_labels'][:]
        # prepare classification data


        model = RandomForestClassifier(n_estimators=100, max_depth=5)  # default args


        # reshape features to be (num trials, num feat * num frames)
        final_ML_feature_array = final_ML_feature_array.reshape(final_ML_feature_array.shape[0],
                                                                final_ML_feature_array.shape[1] *
                                                                final_ML_feature_array.shape[2])


        # partition data into test, train
        X_train, X_test, y_train, y_test = CU.split_ML_array(final_ML_feature_array, final_labels_array, t=0.2)

        type_labels_y_train, num_labels_y_train, hand_labels_y_train, tug_labels_y_train, switch_labels_y_train \
            = CU.get_ML_labels(y_train)
        type_labels_y_test, num_labels_y_test, hand_labels_y_test, tug_labels_y_test, switch_labels_y_test \
            = CU.get_ML_labels(y_test)

        # norm XYZ but NOT prob
        # reshape to (cut trials, num features * frames)
        #X_train = CU.norm_and_zscore_ML_array(X_train, robust=False, decomp=False, gauss=False) # no need anymore

        # reshape prob to (cut trials, num features * frames)
        #X_train_p = X_train_p.reshape(X_train_p.shape[0], X_train_p.shape[1] * X_train_p.shape[2])

        # create final X_train feature array
       # X_train = np.concatenate((X_train, X_train_p), axis=1)

        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model)
        score = cross_val_score(classifier_pipeline, X_train, hand_labels_y_train, cv=3)
        print(np.mean(score))


        print(X_train.shape, X_train.shape)
        #print(np.concatenate((X_train, X_train), axis=1).shape)

        model.fit(X_train, hand_labels_y_train)
        print(model)

    # elif args.question == 5:
    # main_q5()
    else:
        raise ValueError("Cannot find specified question number")
