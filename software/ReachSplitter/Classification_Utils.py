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
import DataStream_Vis_Utils as utils
from networkx.drawing.tests.test_pylab import plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
# from yellowbrick.model_selection import CVScores
# from Classification_Visualization import visualize_model, print_preds
from scipy import ndimage
import pickle


def norm_coordinates(kin_three_vector, transform=True, filtering=False):
    xkin_three_vector = np.zeros(kin_three_vector.shape)
    if transform:
        xkin_three_vector[:, 0] = kin_three_vector[:, 0] * -2.5 + .25  # flip x-axis
        xkin_three_vector[:, 1] = kin_three_vector[:, 1] * -0.2 + .25  # flip y-axis
        xkin_three_vector[:, 2] = kin_three_vector[:, 2] * 1.5 + .5
    if filtering:
        xkin_three_vector[:, 0] = ndimage.median_filter(np.copy(xkin_three_vector[:, 0]), size=1)  # flip x-axis
        xkin_three_vector[:, 1] = ndimage.median_filter(np.copy(xkin_three_vector[:, 1]), size=1)  # flip y-axis
        xkin_three_vector[:, 2] = ndimage.median_filter(np.copy(xkin_three_vector[:, 2]), size=1)
    return np.copy(xkin_three_vector)


class ReachUtils:
    def __init__(self, rat, date, session, exp_data, kin_data, save_path):
        self.nose = []
        self.handle = []
        self.left_shoulder = []
        self.right_shoulder = []
        self.left_forearm = []
        self.right_forearm = []
        self.left_wrist = []
        self.right_wrist = []
        self.left_palm = []
        self.right_palm = []
        self.left_digits = []
        self.right_digits = []
        self.prob_left_arm = []
        self.prob_right_arm = []
        self.prob_nose = []
        self.right_shoulder = []
        self.prob_left_shoulder = []
        self.central_body_mass = []
        self.body_prob = []
        self.prob_right_shoulder = []
        self.left_palm_velocity = []
        self.right_palm_velocity = []
        self.handle_velocity = []
        self.robot_handle_speed = []
        self.block_exp_df = exp_data
        self.time_vector = []
        self.x_robot = []
        self.y_robot = []
        self.z_robot = []
        # self.rdata_path = sensor_data_path
        self.prob_left_index = 0
        self.total_lowp_vector = []
        # self.kinematic_data_path = kinematic_data_path
        self.right_prob_index = []
        self.sensors = []
        self.kinematic_block = kin_data
        self.h_moving_sensor = []
        self.d = []
        self.session = session
        self.date = date
        self.rat = rat
        self.lick = []
        self.lick_vector = []
        self.reward_zone_sensor = []
        self.exp_response_sensor = []
        self.left_prob_index = []
        self.trial_start_vectors = []
        self.trial_stop_vectors = []
        # self.load_data()
        self.get_block_data()
        self.get_start_stop()
        self.save_list = []
        self.sensor_array = []
        self.position_features = []
        self.velocity_features = []
        self.prob_features = []
        return

    def load_data(self):
        # self.sensors = self.import_robot_data().reset_index(drop=True)
        # with (open(self.kinematic_data_path, "rb")) as openfile:
        #    self.d = pickle.load(openfile)
        return

    def import_robot_data(self):
        # df = pd.read_pickle(self.rdata_path)
        # df = preprocessing(df)
        return  # df

    def get_start_stop(self):
        self.trial_start_vectors = self.block_exp_df['r_start'].values[0]
        self.trial_stop_vectors = self.block_exp_df['r_stop'].values[0]
        return

    def get_block_data(self):
        # for kin_items in self.d:
        #    sess = kin_items.columns.levels[1]
        #    date = kin_items.columns.levels[2]
        #    if sess[0] in self.session:
        #       if date[0][-2:] in self.date:
        #           print('Hooked block positions for date  ' + date[0] + '     and session  ' + sess[0])
        #           self.kinematic_block = kin_items
        # self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        # assert len(self.block_exp_df) != 0, "Block not found"
        return

    def extract_sensor_data_for_reaching_predictions(self, idxstrt, idxstp):
        if idxstrt > idxstp:  # handles case where start>stop
            temp = idxstp
            idxstp = idxstrt
            idxstrt = temp
        self.h_moving_sensor = np.copy(self.block_exp_df['moving'].values[0][idxstrt:idxstp])
        self.lick = np.copy(self.block_exp_df['lick'].values[0])  # Lick DIO sensor
        self.reward_zone_sensor = np.copy(self.block_exp_df['RW'].values[0][idxstrt:idxstp])
        self.time_vector = self.block_exp_df['time'].values[0][
                           idxstrt:idxstp]  # extract trial timestamps from SpikeGadgets
        assert len(self.time_vector) != 0, "time_vector is empty!"
        self.time_vector = list(np.around(np.array(self.time_vector), 2))  # round time's to ms
        self.exp_response_sensor = self.block_exp_df['exp_response'].values[0][idxstrt:idxstp]
        self.check_licktime()
        r, theta, phi, self.x_robot, self.y_robot, self.z_robot = utils.forward_xform_coords(
            self.block_exp_df['x_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['y_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['z_pot'].values[0][idxstrt:idxstp])
        try:
            self.sensor_array = np.vstack(
                (self.h_moving_sensor, self.lick_vector, self.reward_zone_sensor, self.time_vector,
                 self.exp_response_sensor, self.x_robot, self.y_robot, self.z_robot))
        except:
            print("issue 1 ")
            exit(-1)
        return

    def check_licktime(self):
        self.lick_vector = np.zeros((len(self.time_vector)))
        self.lick = list(np.around(np.array(self.lick), 2))
        for l in self.lick:
            if l >= self.time_vector[0]:
                for trisx, t in enumerate(self.time_vector):
                    if t in self.lick:
                        self.lick_vector[trisx] = 1
                if l <= self.time_vector[25]:  # is there a rapid reward? Bout-like interactions...
                    self.bout_flag = True
                if l <= self.time_vector[-1]:
                    self.trial_rewarded = True
                    break
        return

    def segment_kinematic_block_by_features(self, cl1, cl2):
        self.nose = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[6:9]].values[cl1:cl2, :])
        self.handle = np.mean(
            [norm_coordinates(self.kinematic_block[self.kinematic_block.columns[0:3]].values[cl1:cl2, :]),
             norm_coordinates(self.kinematic_block[self.kinematic_block.columns[3:6]].values[cl1:cl2, :])], axis=0)
        self.left_shoulder = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[9:12]].values[cl1:cl2, :])  # 21 end
        self.right_shoulder = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])  # 57 end
        self.left_forearm = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        self.right_forearm = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])  # 57 end
        self.left_wrist = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        self.right_wrist = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])  # 57 end
        self.left_palm = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[15:18]].values[cl1:cl2, :])
        self.right_palm = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        self.left_digits = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:45]].values[cl1:cl2, :])
        self.right_digits = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[57:81]].values[cl1:cl2, :])
        self.prob_right_arm = np.mean(
            self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[cl1:cl2, :], axis=1)
        self.prob_left_arm = np.mean(
            self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values[cl1:cl2, :], axis=1)
        self.prob_nose = np.mean(self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values[cl1:cl2, :],
                                 axis=1)
        self.prob_right_shoulder = self.kinematic_block[self.kinematic_block.columns[45 + 81:48 + 81]].values[cl1:cl2,
                                   :]
        self.prob_left_shoulder = self.kinematic_block[self.kinematic_block.columns[9 + 81:12 + 81]].values[cl1:cl2, :]
        self.central_body_mass = np.mean((self.left_shoulder, self.right_shoulder), axis=0)
        self.body_prob = np.mean((np.mean(self.prob_right_shoulder, axis=1),
                                  np.mean(self.prob_left_shoulder, axis=1)), axis=0)
        self.position_features = np.hstack(
            (self.nose, self.handle, self.right_shoulder, self.left_shoulder, self.right_forearm,
             self.left_forearm, self.right_wrist, self.left_wrist, self.right_palm, self.left_palm,
             self.central_body_mass))
        self.prob_features = np.vstack((self.prob_right_arm, self.prob_left_arm, self.prob_nose, self.body_prob))
        return

    def filter_all_kin_data_with_probabilities(self, gen_p_thresh=0.2):
        prob_nose_total = np.mean(self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values, axis=1)
        prob_right_arm_total = np.mean(self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values,
                                       axis=1)
        prob_left_arm_total = np.mean(self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values,
                                      axis=1)
        bad_nose = np.array(prob_nose_total) < gen_p_thresh
        bad_right_arm = np.array(prob_right_arm_total) < gen_p_thresh
        bad_left_arm = np.array(prob_left_arm_total) < gen_p_thresh
        self.total_lowp_vector = np.logical_and(bad_nose, bad_right_arm, bad_left_arm)
        self.kinematic_block.values[self.total_lowp_vector] = 0
        return

    def filter_trial_data_with_probabilities(self, c1, c2, gen_p_thresh=0.2):
        lowp_vector = self.total_lowp_vector[c1:c2]
        self.x_robot[lowp_vector] = 0
        self.y_robot[lowp_vector] = 0
        self.z_robot[lowp_vector] = 0
        self.reward_zone_sensor[lowp_vector] = 0
        self.lick_vector[lowp_vector] = 0
        self.h_moving_sensor[lowp_vector] = 0
        self.exp_response_sensor[lowp_vector] = 0
        return

    def compute_palm_velocities_from_positions(self):
        self.left_palm_velocity = np.zeros(self.left_palm.shape)
        self.right_palm_velocity = np.zeros(self.right_palm.shape)
        self.handle_velocity = np.zeros(self.handle.shape)
        self.robot_handle_speed = np.zeros(self.handle.shape)
        for ddx in range(0, self.right_palm_velocity.shape[0]):
            delta_time = self.time_vector[ddx] - self.time_vector[ddx - 1]
            if delta_time == 0:  # handles div by 0 case
                delta_time = 0.001
            self.robot_handle_speed[ddx, :] = (np.copy(self.x_robot[ddx] - self.x_robot[ddx - 1]) / (
                delta_time) +
                                               np.copy((self.y_robot[ddx] - self.y_robot[ddx - 1]) / (
                                                   delta_time) +
                                                       np.copy((self.z_robot[ddx] - self.z_robot[ddx - 1]) / (
                                                           delta_time) / 3)))
            self.handle_velocity[ddx, :] = np.copy(
                (self.handle[ddx, :] - self.handle[ddx - 1, :]) / (delta_time))
            self.left_palm_velocity[ddx, :] = np.copy((self.left_palm[ddx, :] - self.left_palm[ddx - 1, :]) / (
                delta_time))
            self.right_palm_velocity[ddx, :] = np.copy((self.right_palm[ddx, :] - self.right_palm[ddx - 1, :]) / (
                delta_time))
        np.nan_to_num(self.handle_velocity, 0)
        np.nan_to_num(self.right_palm_velocity, 0)
        np.nan_to_num(self.left_palm_velocity, 0)
        self.velocity_features = np.hstack(
            (self.handle_velocity, self.robot_handle_speed, self.right_palm_velocity, self.left_palm_velocity))
        return

    def create_and_save_classification_features(self):
        feature_names = ['time', 'moving', 'lick', 'rz', 'exp_response', 'xbot', 'ybot', 'zbot',
                         'nosex', 'nosey', 'nosez', 'handlex', 'handley', 'handlez', 'right_s', 'left_s', 'right_f',
                         'left_f', 'right_w', 'left_w',
                         'right_p', 'left_p', 'body_m', 'right_p_prob', 'left_p_prob', 'nose_prob', 'body_prob'
                                                                                                    'handle',
                         'robot_handle_speed', 'right_p_vel', 'left_p_vel']
        trial_numbers = np.linspace(0, len(self.trial_start_vectors), 1)
        self.filter_all_kin_data_with_probabilities(gen_p_thresh=0.4)
        for sd, isx in enumerate(self.trial_start_vectors):  # enumerate over all trial vectors
            stops = self.trial_stop_vectors[sd]
            self.extract_sensor_data_for_reaching_predictions(isx, stops)
            self.segment_kinematic_block_by_features(isx, stops)
            self.filter_trial_data_with_probabilities(isx, stops, gen_p_thresh=0.4)
            self.compute_palm_velocities_from_positions()
            try:
                save_vectors = np.vstack((self.sensor_array,
                                          self.position_features.reshape(self.position_features.shape[1],
                                                                         self.position_features.shape[0]),
                                          self.prob_features,
                                          self.velocity_features.reshape(self.velocity_features.shape[1],
                                                                         self.velocity_features.shape[0])))
            except:
                print("issue 2")
                exit(-1)
            self.save_list.append(save_vectors)  # might be able to use append kwarg..
        save_df = pd.DataFrame(self.save_list)
        save_df.to_csv('ClassifyTrials/Features' + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                       index=False,
                       header=False)
        return self.save_list


# To run ReachUtils
# R=ReachUtils(rat,date,session,kpath,exp_path,save_path) # init
# R.create_and_save_classification_features

####################################
# 1. Load data into pandas DataFrames
####################################


def unpack_pkl_df(rat_df1):
    """Formats a pandas DataFrame.
        rat,date,session,dim levels are converted to columns.
        Each column's values (except rat,date,session,dim) are stored in a single array.

    Attributes
    ------------
    rat_df1: dataframe, a multi-level DataFrame corresponding to an element in an un-pickled list

    Returns
    ----------
        new_df: a new formatted DataFrame with only 1 row for a specific rat,date,session,dim

    """
    # create a new DataFrame (copy of original), then removes levels corresponding to (rat, date, session, dim)
    df1 = rat_df1.droplevel([0, 1, 2, 3], axis=1)  # stores final row values for each rat,date,session
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
    """Converts a pickle files into a pandas DataFrame indexed by rat,date,session,dim

    Attributes
    ------------
    pickle_file: file path to pickle file that contains a list of DataFrames

    Returns
    ---------
    df_to_return: DataFrame indexed by rat,date,session,dim
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
            if not encountered_df:
                encountered_df = True
                df_to_return = unpack_pkl_df(rat_df1)
            # concat new df to existing df
            else:
                df_to_append = unpack_pkl_df(rat_df1)
                df_to_return = pd.concat([df_to_return, df_to_append], axis=0, sort=False)
        except:
            print("do nothing, not a valid df")
    df_to_return = df_to_return.set_index(['rat', 'date', 'session', 'dim'])
    return df_to_return


####################################
# 2. Label processing
####################################


def make_vectorized_labels(blist):
    """Vectorizes list of DLC video trial labels for use in ML-standard format
        Converts labels which hand and tug vs no tug string labels into numbers.

    Attributes
    -------------
    blist: list, of trial labels for a specific rat,date,session
            For more robust description please see github

    Returns
    ----------
    new_list: array, of lists of numeric labels.
            One list corresponds to one labeled trial.
    ind_total: array of lists of reaching indices .
            Currently all empty.

    """
    ll = len(blist)
    new_list = np.empty((ll, 9))
    ind_total = []
    for ix, l in enumerate(blist):
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


def onehot(r_df):
    """Returns one hot array for robot data.

    Attributes
    -----------
    r_df: dataframe, sensor dataframe for single block

    Returns
    -----------
    hot_vec: array, one-hot encoding array of length number of trials in block
        
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
    sf = r_df['SF'].values[0]
    hot_vec = np.zeros(m, dtype=int)
    hot_vec[sf] = 1
    return np.asarray(hot_vec)


###########################
# Generate Features and Aid Classification Functions
##########################

def is_tug_no_tug(moving_times):
    """Function to classify trials with post-reaching behavior from well-behaved handle release.
    Gives a simple estimate of if theres tug of war or not

    Attributes
    ------------
    moving_times: array, array of times the handle moves

    Returns
    ----------
    tug_preds: one-hot encoding of if there is a simple tug-of-war behavior recorded from handle sensor data
    ind_trial_times: int, time (in seconds) of trial from start of reward (lick) to beginning of reach

    """
    reward_end_times = np.argwhere(moving_times == 1)[0]  # take the first index when a robot movement command is issued
    movement_end_times = np.argwhere(moving_times == 1)[-1]
    move_time = 20  # parameter threshold needed to evoke TOW
    if movement_end_times - reward_end_times > move_time:
        tug_preds = 1  # tug of war
    else:
        tug_preds = 0  # no tug of war
    ind_trial_times = movement_end_times - reward_end_times
    return tug_preds, ind_trial_times


def is_reach_rewarded(lick_data_):
    """Function to simply classify trials as rewarded with water or not using sensor data from ReachMaster (lick detector).
    Tells if the reach was rewarded or not

    Attributes
    -----------
    lick_data_: array, times that our licking sensor is activated during a trial

    Returns
    ---------
    rew_lick_: int, flag to determine if a trial has been rewarded

    """
    rew_lick_ = 0
    if lick_data_.any():
        try:
            if np.where(lick_data_ == 1) > 3:  # filter bad lick noise
                rew_lick_ = 1
        except:
            rew_lick_ = 0
    return rew_lick_


#########################
# Classification_Structure helpers
#########################


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
    hand_type_label = np.zeros((hand_labels_.shape[0]))
    if simple:
        hand_type_label[np.where(hand_labels_ > 1)] = 1  # classify all non r,l reaches as 1
    else:
        hand_type_label[np.where(hand_labels_ > 2)] = 1  # classify 0 as r/l
    return hand_type_label


def get_ML_labels(fv):
    fv = fv[:, 3:-1]  # take label catagories necessary for trial classification
    type_labels = onehot_nulls(fv[:, 0])  # labels for trial type
    num_labels = onehot_num_reaches(fv[:, 1])  # labels for num reaches in trial
    hand_labels = hand_type_onehot(fv[:, 2])  # labels for which hand
    tug_labels = fv[:, 3]  # labels for tug/no tug, tug = 1
    switch_labels = fv[:, 4]  # labels for hand switching (y/n), 1 is switch
    return [type_labels, num_labels, hand_labels, tug_labels, switch_labels]


def run_classifier(_model, _X_train, _X_test, input_labels):
    """Function for manually running a given model, intended for hard-code/troubleshooting.
    """
    _model.fit(_X_train, input_labels)
    type_pred = _model.predict(_X_test)
    type_feature_imp = pd.Series(_model.feature_importances_).sort_values(ascending=True)
    return [type_pred, type_feature_imp]


def do_constant_split(model_, ml, feature):
    """classification_structure helper
    Attributes
    ------------

        ml : ML-ready feature vector containing experimental and kinematic data
        feature : labels for each class (vectorized using blist and get_ML_labels)
        model_ : classifier (sk-Learn compatible)

    Returns
    ---------

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
    """classification_structure helper
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
    """helper for structured_classification
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


def split_ML_array(Ml_array, labels_array, t=0.2):
    """Function to split our sensor and positional data into trialized blocks
    Attributes
    -------------

    Ml_array: array, sensor and positional data
    labels_array:
    t:

    Returns
    ---------
    X_train:
    X_test:
    y_train:
    y_test:

    """
    X_train, X_test, y_train, y_test = train_test_split(Ml_array, labels_array, test_size=t, random_state=0)
    return X_train, X_test, y_train, y_test


def norm_and_zscore_ML_array(ML_array, robust=False, decomp=False, gauss=False):
    """ Function to manually norm/Zscore classifier raw data

    Attributes
    ------------
    ML_array: array, (Cut Trials, Features, Frames)   where Cut Trials refers to either the number of Trials
        inside the testing data or training data (Don't call this function for just the total ML data, split beforehand..)
    robust: boolean, use sci-kit learn robust scaling to normalize our data
    decomp: boolean, post-processing step used to return first whitened 20 PCA components to remove linear dependence
    gauss: boolean, use sci-kit learn gaussian distribution scaling to normalize our data

    Returns
    ---------
    r_ML_array: array, shape (cut trials, num features * frames)
    """
    if robust:
        pt = preprocessing.robust_scale()
    elif gauss:
        pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    else:
        pt = preprocessing.MinMaxScaler()
    r_ML_array = pt.fit_transform(ML_array.reshape(ML_array.shape[0], ML_array.shape[1] * ML_array.shape[2]))
    if decomp:  # used to decomp linear correlations, if they exist.
        pca = decomposition.PCA(n=20, whiten=True)
        r_ML_array = pca.fit(r_ML_array)
    return r_ML_array


def structured_classification(ml, feature, model_, train_labels, classifier_pipeline,
                              kFold, pred, disc, bal, conf):
    """classification_structure helper
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

            # TODO
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

            # TODO
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

            # TODO
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


def classification_structure(ml, feature, model_, kFold=False, LOO=False, PCA_data=False, structured=True, pred=False,
                             disc=True, bal=True, conf=True):
    """Classification structure for use in behavioral characterization.
    Attributes
    -----------
        ml : ML-ready feature vector containing experimental and kinematic data
            Shape (Cut Trials, Features, Frames)
        feature : labels for each class (vectorized using blist and get_ML_labels)
        model_ : classifier (sk-Learn compatible)
        kFold : int, number of folds if using kFold cross-validation from sk-Learn
        LOO : boolean flag, set True if using LOO cross-validation from sk-Learn
        PCA : boolean flag, set True if using PCA to reduce dimensions of feature vectors
        constant_split: boolean flag, set True if comparing results between classifiers
        structured: boolean flag, set True to do multiple binary classifications

    Variables
    ---------------
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

     Returns
     ----------
         preds: list of (3 arrays of 5 elems for each classifier in hierarchy) arrays of classifier predictions
         model_: array, final ML model

      Notes:
        kfold boolean arg vs KFold for sklearn.model_selection._split.KFold

    """
    X_train, X_test, y_train, y_test = split_ML_array(ml, feature, t=0.2)
    train_labels = get_ML_labels(y_train)
    X_train = norm_and_zscore_ML_array(X_train, robust=False, decomp=False, gauss=False)
    X_test = norm_and_zscore_ML_array(X_test, robust=False, decomp=False, gauss=False)
    if PCA_data:
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(),
                                            decomposition.PCA(n_components=int(PCA_data)), model_)
    else:
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model_)
    if structured:
        return structured_classification(ml, feature, model_,
                                         X_train, X_test, y_train, y_test, train_labels, classifier_pipeline,
                                         kFold, pred, disc, bal, conf)
    else:
        return simple_classification_verification(train_labels, classifier_pipeline, ml, feature, kFold, model_, LOO,
                                                  X_train, X_test)


###################################
# Convert Nested lists/arrays into pandas DataFrames
##################################

def make_vectorized_labels_to_df(labels):
    """Convert return value from make_vectorized_labels into a pandas df
    
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


def import_experiment_features_to_df(exp_features):
    """Converts return value of import_experiment_features to pandas dataframe.
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


pos_names = ['Handle', 'Back Handle', 'Nose',
             'Left Shoulder', 'Left Forearm', 'Left Wrist', 'Left Palm', 'Left Index Base', 'Left Index Tip',
             'Left Middle Base', 'Left Middle Tip', 'Left Third Base',
             'Left Third Tip', 'Left Fourth Finger Base', 'Left Fourth Finger Tip',
             'Right Shoulder', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Right Index Base',
             'Right Index Tip', 'Right Middle Base', 'Right Middle Tip', 'Right Third Base',
             'Right Third Tip', 'Right Fourth Finger Base', 'Right Fourth Finger Tip']


###############################
# Classification Hierarchy
###############################

def remove_trials(X, Y, preds, toRemove):
    """Removes trials from labels after classification. Used to prepare data for next classification in hierarchy.
    Attributes
    -------------
        X (array): features, shape (num trials, num feat*num frames)
        Y (array): labels
        shape # type_labels_y_train, num_labels_y_train, hand_labels_y_train, tug_labels_y_train, switch_labels_y_train
        preds (array): classifier trial predictions
        toRemove: 0 to remove trials classified as 0, 1 otherwise

    Returns
    ----------
        X (array): filtered
        Y (array): filtered

    Notes
    ----------
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


def select_feat_by_keyword(feat_df, keywords):
    """Returns data from selected features.
    Attributes
    -----------
    feat_df(df): df of features
    keywords(list of str): list of feature column names to select

    Returns
    ----------
    df: of selected features
    arr: same data as df, just in array form

    """
    feat_names_arr = np.array(feat_df.columns)
    selected_data = []

    # select data
    for keyword in keywords:
        selected_cols = np.array([s for s in feat_names_arr if keyword in s])
        new_feat_arr = feat_df[selected_cols]
        selected_data.append(new_feat_arr)

    df = pd.concat(selected_data, axis=1)
    arr = df.values
    return df, arr


def is_tug_no_tug():
    """Function to classify trials with post-reaching behavior from well-behaved handle release.
    """
    # ask if there is robot velocity after a trial ends (for around a second)
    tug_preds = []
    return tug_preds


###############################
# DLC Video Labels
###############################

rm16_9_17_s1_label = [
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

# RM16, 9-18, S1
# l18
rm16_9_18_s1_label = [
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

# RM16, 9-17, S2
# nl2
rm16_9_17_s2_label = [
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

# RM16, DATE 9-20, S3
# blist1
rm16_9_20_s3_label = [
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

# RM16, 09-19-2019, S3
# elist
rm16_9_19_s3_label = [
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
    , [40, 91818, 92387, 1, 7, 'l', 'no_tug', 0, 6]]  # fail # starts mid reach

# RM15, 25, S3
# blist2
rm15_9_25_s3_label = [
    [3, 1, 1, 3, 1, 'r', 'noTug', 0, 0, 16360, 16400],
    [2, 1, 1, 3, 1, 'r', 'noTug', 0, 0, 15375, 15470],
    [4, 1, 1, 3, 1, 'r', 'Tug', 0, 0, 20555, 20600],
    [5, 1, 1, 3, 1, 'l', 'noTug', 0, 0, 23930, 24000],
    [6, 1, 1, 3, 1, 'r', 'Tug', 0, 0, 27145, 27205],
    [8, 1, 1, 3, 1, 'r', 'Tug', 0, 0, 32215, 32300],
    [9, 1, 1, 0, 0, 0, 0, 0, 0],
    [10, 1, 1, 3, 1, 'r', 'Tug', 0, 0, 34415, 34495],
    [11, 1, 1, 2, 3, 'r', 'noTug', 0, 0, 35300, 35333, 36365, 35330, 35360, 35405],
    [14, 1, 1, 3, 1, 'r', 'noTug', 0, 0, 72470, 72505],
    [16, 1, 1, 3, 1, 'r', 'noTug', 0, 0, 75000, 75090],
    [17, 1, 1, 3, 1, 'r', 'noTug', 0, 0, 86570, 86610],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],  # null
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # null
    [7, 1, 1, 0, 0, 0, 0, 0, 0],  # null
    [12, 1, 1, 0, 0, 0, 0, 0, 0],  # null
    [13, 1, 1, 0, 0, 0, 0, 0, 0],  # null
    [15, 1, 1, 0, 0, 0, 0, 0, 0],  # null rear
    [18, 1, 1, 2, 1, 'r', 'Tug', 0, 0, 87495, 87555]]

# RM15, 17, S4
# blist3
rm15_9_17_s4_label = [
    [0, 1, 1, 3, 1, 'lbi', 'tug', 0, 0, 9470, 9610],
    [1, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0, 10605, 10880, 10675, 10940],
    [2, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0, 11630, 11675, 11676, 11720],
    [3, 1, 1, 3, 3, 'lbi', 'noTug', 0, 0, 12770, 13035, 13090, 13165, 12830, 13085, 13155, 13185],
    [4, 1, 1, 3, 1, 'l', 'noTug', 0, 0, 14635, 14700],
    [5, 1, 1, 3, 3, 'lbi', 'Tug', 0, 0, 19105, 19125, 19200, 19120, 19185, 19280],
    [6, 1, 1, 3, 3, 'lbi', 'noTug', 0, 0, 20730, 20763, 20845, 20762, 20830, 20915],
    [7, 1, 1, 3, 2, 'lbi', 'Tug', 0, 0, 21930, 21985, 21980, 22040]]

# 2019-09-20-S1-RM14_cam2
rm14_9_20_s1_label = [
    [0, 1358, 1921, 0, 0, 0, 'no_tug', 0, 0],  # null
    [1, 3092, 3679, 3, 1, 'r', 'no_tug', 0, 0],  # success
    [2, 4104, 4668, 1, 1, 'r', 'no_tug', 0, 0],  # failed
    [3, 5418, 5919, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [4, 7127, 7584, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [5, 12048, 12544, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [6, 13145, 13703, 1, 4, 'r', 'no_tug', 1, 0],  # failed, reach 2 l, 2 r
    [7, 14120, 14609, 3, 1, 'l', 'no_tug', 0, 0],  # success, interupted
    [8, 16153, 16726, 3, 4, 'lra', 'no_tug', 1, 0],  # success, lra
    [9, 17276, 17838, 0, 0, 0, 'no_tug', 0, 0],  # null
    [10, 18557, 19083, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [11, 19543, 20104, 1, 4, 'r', 'no_tug', 1, 0],  # failed, reach 2 l, 2 r
    [12, 25112, 25756, 3, 3, 'bi', 'no_tug', 1, 0],  # success, bi
    [13, 26288, 26847, 1, 4, 'r', 'no_tug', 0, 0],  # failed, reach 2 l, 2 r
    [14, 30104, 30667, 1, 1, 'l', 'no_tug', 0, 0],  # failed
    [15, 31876, 32426, 0, 0, 0, 'no_tug', 0, 0],  # null
    [16, 33928, 34448, 3, 2, 'lra', 'no_tug', 1, 0],  # success, lra
    [17, 34880, 35461, 3, 2, 'lra', 'no_tug', 1, 0],  # success, lra
    [18, 36083, 36707, 3, 3, 'lra', 'no_tug', 1, 0],  # success, lra
    [19, 37190, 37781, 1, 3, 'l', 'no_tug', 0, 0],  # failed
    [20, 38580, 39172, 3, 2, 'l', 'no_tug', 1, 0],  # success
    [21, 42519, 43217, 3, 2, 'r', 'no_tug', 1, 0],  # success
    [22, 44318, 44887, 3, 6, 'rla', 'no_tug', 1, 0],  # success
    [23, 45311, 45784, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [24, 46207, 46755, 3, 2, 'l', 'no_tug', 0, 0],  # success
    [25, 47341, 47885, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [26, 48442, 49004, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [27, 49425, 49987, 3, 1, 'l', 'no_tug', 0, 0],  # success, interrupted
    [28, 50410, 50970, 3, 6, 'r', 'no_tug', 1, 0],  # success, not quite in reward zone, but there's licking
    [29, 55926, 56488, 3, 1, 'r', 'no_tug', 0, 0],  # success
    [30, 56911, 57404, 3, 2, 'lra', 'no_tug', 1, 0],  # success, interrupted
    [31, 58700, 59261, 3, 5, 'r', 'no_tug', 1, 0],  # success, in reward zone, no licking
    [32, 59708, 60271, 1, 3, 'r', 'no_tug', 1, 0],  # failed
    [33, 68042, 68618, 1, 3, 'r', 'no_tug', 0, 0],  # failed
    [34, 69121, 69697, 3, 3, 'rla', 'no_tug', 1, 0],  # success
    [35, 70242, 70816, 3, 4, 'l', 'no_tug', 1, 0],  # success
    [36, 71549, 72109, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [37, 72541, 73115, 3, 1, 'l', 'no_tug', 0, 0],  # success, interrupted
    [38, 75805, 76325, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [39, 76756, 77326, 3, 4, 'lra', 'no_tug', 1, 0],  # success, interrupted
    [40, 78866, 79439, 3, 2, 'l', 'no_tug', 0, 0],  # success
    [41, 80692, 811893, 3, 2, 'l', 'no_tug', 1, 0],  # success
    [42, 82560, 83181, 3, 2, 'bi', 'no_tug', 1, 0],  # success, is bi reach
    [43, 83612, 84096, 3, 1, 'l', 'no_tug', 0, 0],  # success, interupted
    [44, 84527, 85046, 3, 1, 'l', 'no_tug', 0, 0],  # success, interupted
    [45, 85475, 85992, 3, 1, 'r', 'no_tug', 0, 0],  # success, interupted
    [46, 87271, 87844, 3, 2, 'bi', 'no_tug', 1, 0],  # success, is bi
    [47, 88275, 88849, 1, 4, 'r', 'no_tug', 1, 0],  # failed, interrupted
    [48, 90043, 90620, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [49, 91675, 92251, 3, 2, 'l', 'no_tug', 0, 0]]  # success

# 2019-09-18-S2-RM14-cam2
rm14_9_18_s2_label = [
    [0, 3144, 3708, 0, 0, 0, 'no_tug', 0, 0],  # null
    [1, 6876, 7735, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [2, 9931, 10499, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [3, 12640, 13206, 1, 1, 'l', 'no_tug', 0, 0],  # failed
    [4, 13703, 14446, 3, 3, 'lra', 'no_tug', 1, 0],  # success, interrupted
    [5, 14895, 15445, 3, 3, 'bi', 'no_tug', 1, 0],  # success, bi
    [6, 15893, 16457, 0, 0, 0, 'no_tug', 0, 0],  # null
    [7, 16874, 17465, 1, 2, 'r', 'no_tug', 1, 0],  # failed, reach 1 r, 1 l
    [8, 21293, 21792, 3, 2, 'rla', 'no_tug', 1, 0],  # success
    [9, 22507, 23014, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [10, 23638, 24203, 1, 2, 'l', 'no_tug', 0, 0],  # failed
    [11, 24730, 25296, 3, 1, 'r', 'no_tug', 0, 0],  # success, not quite in reward zone
    [12, 26751, 27319, 0, 0, 0, 'no_tug', 0, 0],  # null
    [13, 29469, 30040, 0, 0, 0, 'no_tug', 0, 0],  # null
    [14, 30505, 31091, 0, 0, 0, 'no_tug', 0, 0],  # null
    [15, 31689, 32276, 3, 1, 'l', 'no_tug', 0, 0],  # success
    [16, 35532, 36112, 1, 5, 'r', 'no_tug', 1, 0],  # failed
    [17, 36593, 37174, 0, 0, 0, 'no_tug', 0, 0],  # null
    [18, 38288, 38927, 3, 2, 'r', 'no_tug', 0, 0],  # success
    [19, 39478, 40047, 1, 2, 'l', 'no_tug', 0, 0],  # failed
    [20, 43806, 44379, 0, 0, 0, 'no_tug', 0, 0],  # null
    [21, 46307, 46888, 1, 1, 'r', 'no_tug', 0, 0],  # failed
    [22, 47932, 48511, 0, 0, 0, 'no_tug', 0, 0],  # null
    [23, 48947, 49591, 3, 2, 'bi', 'no_tug', 1, 0],  # success, bi
    [24, 54297, 54875, 1, 3, 'l', 'no_tug', 1, 0],  # failed
    [25, 55317, 55894, 0, 0, 0, 'no_tug', 0, 0],  # null
    [26, 68274, 69141, 3, 3, 'l', 'no_tug', 1, 0],  # success
    [27, 69617, 70207, 3, 2, 'r', 'no_tug', 0, 0],  # success
    [28, 72591, 73164, 3, 2, 'rla', 'no_tug', 1, 0],  # success
    [29, 73719, 74223, 3, 2, 'rla', 'no_tug', 1, 0],  # success
    [30, 79481, 80078, 3, 2, 'bi', 'no_tug', 1, 0],  # success
    [31, 80529, 81127, 0, 0, 0, 'no_tug', 0, 0],  # null
    [32, 83369, 83963, 1, 4, 'r', 'no_tug', 1, 0],  # failed, reach 2 l , 2 r
    [33, 84447, 85042, 1, 4, 'r', 'no_tug', 1, 0],  # failed, interrupted
    [34, 85491, 86088, 1, 3, 'r', 'no_tug', 0, 0],  # failed
    [35, 86539, 87139, 3, 5, 'bi', 'no_tug', 1, 0],  # success, interrupted bi
    [36, 88801, 89422, 3, 4, 'lra', 'no_tug', 1, 0],  # success, interrupted
    [37, 89877, 90689, 3, 3, 'l', 'no_tug', 1, 0],  # success
    [38, 91302, 91901, 3, 1, 'r', 'no_tug', 1, 0],  # success, interrupted
    [39, 96013, 96611, 1, 4, 'r', 'no_tug', 1, 0]]  # failed, reach 2 l , 2 r

# Guang's labels
# 2019-09-20-S3-RM13-cam2
# TODO  cant assing to literal error and weird extra bracket needed?
"""rm13_9_20_s3_label = [
                      [0, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [1, 0, 0, 1, 2, 'l', 'no_tug', 0, 0, 1], #failed
                      [2, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [3, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], #null
                      [4, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0], #success
                      [5, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [6, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [7, 0, 0, 1, 4, 'l', 'no_tug', 0, 0, 0], #failed
                      [8, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [9, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 0], #success
                      [10, 0, 0, 1, 2, 'l', 'no_tug', 0, 0, 1], #failed
                      [11, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [12, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [13, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [14, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [15, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [16, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0], #success
                      [17, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [18, 0, 0, 3, 6, 'r', 'no_tug', 1, 0, 0], #success #bi
                      [19, 0, 0, 1, 13, 'bi', 'no_tug', 1, 0, 0], #failed #bi #lots of reaches
                      [20, 0, 0, 1, 5, 'l', 'no_tug', 0, 0, 0], #failed
                      [21, 0, 0, 1, 3, 'l', 'no_tug', 0, 0, 0], #failed
                      [22, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [23, 0, 0, 1, 2, 'r', 'no_tug', 0, 0, 0], #failed
                      [24, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 1, #success #starts mid reach
                      [25, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [26, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [27, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [28, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [29, 0, 0, 1, 2, 'l', 'no_tug', 0, 0, 0], #failed
                      [30, 0, 0, 3, 2, 'bi', 'no_tug', 1, 0, 0], #success #bi
                      [31, 0, 0, 3, 6, 'r', 'no_tug', 1, 0, 0], #success #bi
                      [32, 0, 0, 1, 4, 'l', 'no_tug', 0, 0, 0], #failed
                      [33, 0 ,0, 3, 2, 'r', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [34, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [35, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [36, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [37, 0, 0, 3, 5, 'r', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [38, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [39, 0, 0, 1, 4, 'r', 'no_tug', 0, 0, 0], #failed
                      [40, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [41, 0, 0, 1, 3, 'l', 'no_tug', 0, 0, 0], #failed
                      [42, 0, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [43, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [44, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 0], #success
                      [45, 0, 0, 1, 6, 'r', 'no_tug', 1, 0, 0], #failed #hand-swith at the last reach
                      [46, 0, 0, 3, 10, 'bi', 'no_tug', 1, 0, 1], #success #starts mid reach #bi #lots of reaches
                      [47, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [48, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [49, 0, 0, 1, 7, 'l', 'no_tug', 0, 0, 1], #failed with lots of reaches
                      [50, 0, 0, 3, 6, 'r', 'no_tug', 0, 0, 0], #success
                      [51, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0], #success
                      [52, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 0], #success
                      [53, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 1], #success #starts mid reach
                      [54, 0, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [55, 0, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0], # null
                      [56, 0, 0, 3, 3, 'r', 'no_tug', 0, 0, 0], #success
                      ]"""


# 2019-09-18-S4-RM11-cam2
rm11_9_18_s4_label = [
    [0, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [1, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [2, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [3, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [4, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [5, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [6, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [7, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [8, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [9, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [10, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [11, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [12, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [13, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [14, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [15, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [16, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [17, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success #mouth reach #hand reach failed
    [18, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success #mouth reach #hand reach failed
    [19, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 1],  # success #mouth reach #starts mid reach
    [20, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [21, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [22, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [23, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 1],  # success #mouth reach #starts mid reach
    [24, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [25, 0, 0, 3, 1, 0, 'no_tug', 0, 0, 0],  # success #mouth reach without hand
    [26, 0, 0, 3, 2, 'r', 'no_tug', 0, 0, 0],  # success
    [27, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [28, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0]  # null
]

# 2019-09-19-S3-RM9-cam2
rm9_9_19_s3_label = [
    [0, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [1, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [2, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [3, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [4, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 1],  # success #starts mid reach
    [5, 0, 0, 3, 2, 'l', 'no_tug', 0, 0, 0],  # success
    [6, 0, 0, 1, 1, 'l', 'no_tug', 0, 0, 0],  # failed
    [7, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 1],  # success #starts mid reach
    [8, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [9, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [10, 0, 0, 3, 2, 'l', 'no_tug', 0, 0, 0],  # success
    [11, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [12, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [13, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [14, 0, 0, 3, 2, 'bi', 'no_tug', 1, 0, 0],  # success #bi
    [15, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [16, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [17, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [18, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [19, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [20, 0, 0, 3, 1, 'bi', 'no_tug', 0, 0, 1],  # success #bi #starts mid reach
    [21, 0, 0, 3, 2, 'bi', 'no_tug', 1, 0, 0],  # success #bi
    [22, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [23, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [24, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [25, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [26, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [27, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [28, 0, 0, 3, 1, 'r', 'no_tug', 0, 0, 0],  # success
    [29, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [30, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [31, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [32, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [33, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [34, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [35, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [36, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [37, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [38, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [39, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [40, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [41, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [42, 0, 0, 3, 2, 'l', 'no_tug', 0, 0, 0],  # success
    [43, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [44, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [45, 0, 0, 3, 2, 'l', 'no_tug', 0, 0, 1],  # success #starts mid reach
    [46, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [47, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [48, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [49, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [50, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
]

# 2019-09-17-S2-RM10-cam2
rm10_9_17_s2_label = [
    # [0, 0, 0, ] # can't open the file
    [1, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success
    [2, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success # starts mid reach
    [3, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success # starts mid reach
    [4, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success # starts mid reach
    [5, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success # starts mid reach
    [6, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success #
    [7, 0, 0, 3, 1, 'l', 'no_tug', 0, 0, 0],  # success # starts mid reach
]

# 2019-09-19-S1-RM9-cam2
rm12_9_19_s1_label = [
    [0, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [1, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [2, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [3, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
    [4, 0, 0, 0, 0, 0, 'no_tug', 0, 0, 0],  # null
]
