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

import sklearn
from networkx.drawing.tests.test_pylab import plt
from scipy import ndimage
import pickle

from sklearn.ensemble import RandomForestClassifier

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

# classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing


def main_1_vec_labels(save=False):
    # vectorize DLC labels into ML ready format
    # make_vectorized_labels returns list: vectorized list of labels,
    # e: vectorized array of reading indices (unused variables)
    elists, ev = CU.make_vectorized_labels(CU.elist)  # RM16, 09-19-2019, S3
    labellist, edddd = CU.make_vectorized_labels(CU.blist1)  # RM16, DATE 9-20, S3
    nl1lists, ev1 = CU.make_vectorized_labels(CU.nl1)  # RM16, 9-18, S1
    nl2lists, ev2 = CU.make_vectorized_labels(CU.nl2)  # RM16, 9-17, S2
    l18l, ev18 = CU.make_vectorized_labels(CU.l18)  # RM16, 9-17, S1

    vectorized_labels = [elists, labellist, nl1lists, nl2lists, l18l]

    if save:
        # save each vectorized label
        file_name = "vectorized_labels"
        key_names = ['elists', 'labellist', 'nl1lists', 'nl2lists', 'l18l']
        hf = h5py.File(file_name, 'w')
        for i in np.arange(len(vectorized_labels)):
            hf.create_dataset(key_names[i], data=vectorized_labels[i])
        hf.close()
        print("Saved vectorized labels.")

    print("Finished vectorizing labels.")
    return vectorized_labels


def main_2_kin_exp_blocks(save=False):
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
        ['RM16', '0190918', '0190918', 'S1'],
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

    if save:
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
        print("Saved kin & exp blocks.")

    print("Finished creating kin & exp blocks.")
    return kin_blocks, exp_blocks


def main_3_ml_feat_labels(save=False):
    # TODO make it easier to add more labels?

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
    window_length = 4  # TODO change to preferences, default = 250
    pre = 2  # TODO change to preferences, default = 10

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
    matched_kin_e1, ez4 = CU.match_stamps(tt3, elists, e3)  # Matched to
    matched_kin_b1nl1, eznl1 = CU.match_stamps(ttnl1, nl1lists, enl1)  # Matched to
    matched_kin_e1l18, ezl18 = CU.match_stamps(ttl18, l18l, el18)  # Matched to
    matched_kin_b1nl2, eznl2 = CU.match_stamps(ttnl2, nl2lists, enl2)  # Matched to

    # match kin and exp features
    # create_ML_array args: matched kin array, matched ez array
    c, c_prob = CU.create_ML_array(matched_kin_b1, ez1)
    c1, c1_prob = CU.create_ML_array(matched_kin_e1, ez4)
    c2, c2_prob = CU.create_ML_array(matched_kin_e1l18, ezl18)
    c3, c3_prob = CU.create_ML_array(matched_kin_b1nl1, eznl1)
    c4, c4_prob = CU.create_ML_array(matched_kin_b1nl2, eznl2)

    # Create final ML arrays
    vectorized_labels = [labellist, elists, l18l, nl1lists, nl2lists]
    final_ML_feature_array_XYZ, final_labels_array \
        = CU.stack_ML_arrays([c, c1, c2, c3, c4], vectorized_labels)
    final_ML_feature_array_prob, _ \
        = CU.stack_ML_arrays([c_prob, c1_prob, c2_prob, c3_prob, c4_prob], vectorized_labels)

    # concat horizontally XYZ and prob123 ml feature arrays
    # (total num labeled trials x (3*num kin feat)*2 +num exp feat = 174 for XYZ and prob, window_length+pre)
    final_ML_feature_array = np.concatenate((final_ML_feature_array_XYZ, final_ML_feature_array_prob),
                                            axis=1)  # this causes issues with norm/zscore

    if save:
        # Save final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'w') as hf:
            hf.create_dataset("RM16_features", data=final_ML_feature_array)
            hf.create_dataset("RM16_labels", data=final_labels_array)
        print("Saved final ml feat and label arrays.")
        with open('feat_names.npy', 'wb') as f:
            np.save(f, feats)

    print("Finished creating final ML feat and labels.")
    return final_ML_feature_array, final_labels_array


def classify(model, X, Y, k):
    """
    Classifies trials as null vs not null.
    Args:
        model: sklearn model
        X_train (array): features, shape (num trials, num feat*num frames)
        hand_labels_y_train (array): labels shape (num trials)
        k (int): number of kfolds for cross validation score

    Returns:
        classifier_pipeline (pipeline): trained model
        predictions (array): classifier trial predictions
        score (int): mean of cross validation scores
    """
    # create pipeline
    classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model)

    # fit to training data
    classifier_pipeline.fit(X, Y)

    # calculate mean of kfold cv
    score = np.mean(cross_val_score(classifier_pipeline, X, Y, cv=k))

    # predict X_train data
    predictions = classifier_pipeline.predict(X)

    return classifier_pipeline, predictions, score


def main_4_classify(save=False):
    # Load final_ML_array and final_feature_array in h5 file
    with h5py.File('ml_array_RM16.h5', 'r') as f:
        final_ML_feature_array = f['RM16_features'][:]
        final_labels_array = f['RM16_labels'][:]
    with open('feat_names.npy', 'rb') as f:
        feat_names = np.load(f)
    feat_names = [str(t[0]) for t in feat_names]  # un-nest


    # TODO feature engineering
    # TODO test set format

    ### prepare classification data ###

    # reshape features to be (num trials, num feat * num frames)
    num_frames = final_ML_feature_array.shape[2]
    final_ML_feature_array = final_ML_feature_array.reshape(final_ML_feature_array.shape[0],
                                                            final_ML_feature_array.shape[1] *
                                                            final_ML_feature_array.shape[2])

    # partition data into test, train
    X_train, X_test, y_train, y_test = CU.split_ML_array(final_ML_feature_array, final_labels_array, t=0.2)

    # type_labels_y_train, num_labels_y_train, hand_labels_y_train, tug_labels_y_train, switch_labels_y_train \
    y_train = CU.get_ML_labels(y_train)
    y_test = CU.get_ML_labels(y_test)

    ### classify ###

    # init basic variables
    k = 3

    # 1. NULL V NOT NULL
    model = RandomForestClassifier(n_estimators=100, max_depth=5)  # default args

    # 1a. feature selection
    keywords = ['Nose', 'Handle']
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    # 1b.
    type_labels_y_train = y_train[0]
    classifier_pipeline_null, predictions_null, score_null = classify(model, X_train_selected, type_labels_y_train, k)

    # print(predictions_null, score_null)

    # 1c. REMOVE NULL TRIALS
    toRemove = 1  # remove null trials # 1 if null, 0 if real trial
    print(np.array(X_train).shape, np.array(y_train).shape)
    X_train_null, y_train_null = CU.remove_trials(X_train, y_train, predictions_null, toRemove)
    print(X_train_null.shape, y_train_null.shape)

    # 2. NUM REACHES
    model = sklearn.svm.SVC()

    # 2a. feature selection
    keywords = ['Palm', 'Handle', 'Robot']
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train_null, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    # 2b. classify
    num_labels_y_train = y_train_null[1]
    classifier_pipeline_reaches, predictions_reaches, score_reaches = classify(model, X_train_selected, num_labels_y_train,
                                                                               k)
    # 2c. REMOVE >1 REACH TRIALS
    toRemove = 1  # remove >1 reaches # 0 if <1, 1 if > 1 reaches
    X_train_reaches, y_train_reaches = CU.remove_trials(X_train_null, y_train_null, predictions_reaches, toRemove)
    print(X_train_reaches.shape, y_train_reaches.shape)

    # 3. WHICH HAND
    model = RandomForestClassifier()

    # 3a. feature selection
    keywords = ['Robot', 'Palm']
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train_reaches, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    # 3b. classify
    hand_labels_y_train = y_train_reaches[2]
    classifier_pipeline_hand, predictions_hand, score_hand = classify(model, X_train_selected, hand_labels_y_train,
                                                                      k)
    # 3c. REMOVE lra/rla/bi HAND TRIALS
    toRemove = 1  # remove lra/rla/bi reaches # 1 if lra/rla/bi, 0 l/r reaches
    X_train_hand, y_train_hand = CU.remove_trials(X_train_reaches, y_train_reaches, predictions_hand, toRemove)
    print(X_train_hand.shape, y_train_hand.shape)

    print(score_null, score_hand, score_reaches)
    print("Finished classification.")


#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.function == 1:
        main_1_vec_labels(save=True)

    elif args.function == 2:
        main_2_kin_exp_blocks(save=True)

    elif args.function == 3:
        main_3_ml_feat_labels(save=True)

    elif args.function == 4:
        main_4_classify(save=True)

    elif args.function == 5:
        run_all_and_save = True  # change to preferences

        # MUST DELETE ALL OLD DATA FILES BEFORE RUNNING if NOT using default args
        # run all
        if run_all_and_save:
            main_1_vec_labels(save=True)
            main_2_kin_exp_blocks(save=True)
            main_3_ml_feat_labels(save=True)
            main_4_classify(save=True)
        else:
            main_1_vec_labels()
            main_2_kin_exp_blocks()
            main_3_ml_feat_labels()
            main_4_classify()

    else:
        raise ValueError("Cannot find specified question number")
