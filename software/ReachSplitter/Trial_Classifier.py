"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab
               Emily Nguyen, UC Berkeley

    This code is intended to create and implement structure supervised classification of coarsely
    segmented trial behavior from the ReachMaster experimental system.
    Functions are designed to work with a classifier of your choice.
    Operates on a single block.

    Edited: 6/4/2021
"""
import argparse
import os
import sklearn
from scipy import ndimage
import Classification_Utils as CU
import pandas as pd
import numpy as np
import h5py
import random

# set global random seed for reproducibility #
random.seed(246810)
np.random.seed(246810)

# Create folder in CWD to save data and plots #
current_directory = os.getcwd()
folder_name = 'ClassifyTrials'
final_directory = os.path.join(current_directory, folder_name)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)


class IsReachClassifier:
    random.seed(246810)
    np.random.seed(246810)

    def __init__(self):
        self.model = sklearn.ensemble.RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)


class ClassificationHierarchy:
    random.seed(246810)
    np.random.seed(246810)

    def __init__(self, models):
        self.models = models

    def split(self, X, y, model):
        # sort
        mask_left, mask_right = self.split_test(X, model)
        X0, y0, X1, y1 = 0, 0, 0, 0
        return X0, y0, X1, y1

    def split_test(self, X, model):
        # classify
        preds = model.predict(X)
        mask_left, mask_right = preds, preds
        return mask_left, mask_right

    def fit(self, X, y):
        for model in self.models:
            # model.fit(X), predict(X, y), split(X, y, model)
            pass

    def trace_datapoint(self, X, arr=[]):
        """ Q3.2
        for a data point from the spam dataset, prints splits and thresholds
        as it is classified down the tree.
        """
        pass


class Preprocessor:
    def __init__(self):
        """
        Trial-izes data into a ML compatible format.
        """
        self.kin_data = None
        self.exp_data = None
        self.label = None  # CU.make_vectorized_labels(label)
        self.kin_block = None
        self.formatted_kin_block = None
        self.exp_block = None
        self.all_exp_blocks = []
        self.all_kin_blocks = []

    def set_kin_data(self, data):
        self.kin_data = data

    def set_exp_data(self, data):
        self.exp_data = data

    def set_kin_block(self, data):
        self.kin_block = data
        self.set_formatted_kin_block()

    def set_formatted_kin_block(self):
        # rm ID levels
        index = self.kin_block.columns[0]
        self.formatted_kin_block = self.kin_block[index[0]][index[1]][index[2]][index[3]]
        # filter bodypart columns
        Preprocessor.apply_median_filter(self.kin_block)  # uses default wv

    def set_exp_block(self, data):
        self.exp_block = data

    def set_label(self, data):
        self.label = data

    @staticmethod
    def load_data(filename, file_type='pkl'):
        """
        Loads FILENAME as pandas DataFrame.
        Args:
            filename: (str) path to file to load
            file_type: (str) file type to load

        Returns: (df) pandas DataFrame

        """
        assert file_type == 'pkl' or file_type == 'h5' or file_type == 'csv', f'{file_type} not a valid file type'
        if file_type == 'pkl':
            return pd.read_pickle(filename)
        elif file_type == 'h5':
            # get h5 key
            with h5py.File(filename, "r") as f:
                key = list(f.keys())[0]
            return pd.read_hdf(filename, key)
        elif file_type == 'csv':
            return pd.read_csv(filename)

    @staticmethod
    def save_data(df, filename, file_type='csv'):
        """
        Saves FILENAME.
        Args:
            df: (df) to save
            filename: (str) path to file
            file_type: (str) file type

        Returns: None

        """
        assert file_type == 'csv' or file_type == 'pkl', f'{file_type} not a valid file type'
        if file_type == 'csv':
            df.to_csv(filename)
        if file_type == 'pkl':
            df.to_pickle(filename)

    @staticmethod
    def get_single_block(df, date, session, rat, save_as=None, format='exp'):
        """
        Returns DataFrame from data with matching rat, date, session.
        Args:
            df: (df) DataFrame with all blocks
            date: (str) date
            session: (str) session number
            rat: (str) rat name
            save_as: (bool) True to save as csv file, else default None
            format: (str) specifies which type of block to retrieve (kin or exp)
        Returns: new_df: (df) with specified rat, date, session

        """
        if format == 'exp':
            rr = df.loc[df['Date'] == date]
            rr = rr.loc[rr['S'] == session]
            new_df = rr.loc[rr['rat'] == rat]
        else:  # kin case
            new_df = pd.DataFrame()
            for block in df:
                index = block.columns[0]
                if rat == index[0] and session == index[1] and date == index[2]:
                    new_df = pd.DataFrame(block)
        assert (len(new_df.index) != 0), "block does not exist in data!"
        if save_as:
            Preprocessor.save_data(new_df, save_as, file_type='pkl')
        return new_df

    @staticmethod
    def apply_median_filter(df, wv=5):
        """
        Applies a multidimensional median filter to DF columns.
        Args:
            df: (df)
            wv: (int) the wavelet # for the median filter applied to the positional data (default 5)

        Returns: Filtered df. Has the same shape as input.
        """
        # iterate across columns
        for (columnName, columnData) in df.iteritems():
            # Apply median filter to column array values (bodypart, pos or prob)
            df[columnName] = ndimage.median_filter(columnData.values, size=wv)
        return df

    @staticmethod
    def stack(df):
        """
        Reshapes DF. Stack the prescribed level(s) from columns to index.
        Args:
            df: (df)

        Returns: stacked df
        """
        df_out = df.stack()
        df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
        if isinstance(df_out, pd.Series):
            df_out = df_out.to_frame()
        return df_out

    @staticmethod
    def split_trial(kin_block, exp_block, window_length=250, pre=10):
        """
        Partitions kinematic data into trials
        Args:
            kin_block: (df)
            exp_block: (df)
            window_length (int): trial splitting window length, the number of frames to load data from (default 250)
                    Set to 4-500. 900 is too long.
            pre: int, pre cut off before a trial starts, the number of frames to load data from before start time
                    For trial splitting, set to 10. 50 is too long. (default 10)

        Returns: (list of dfs) of length number of trials with index trial number

        """
        start = exp_block['r_start'].values[0]
        trials = []
        for frame_num in start:
            trials.append(kin_block.loc[frame_num - pre:frame_num + window_length])
        return trials

    @staticmethod
    def match_to_label(trials, label_array):
        """

        Args:
            trials: (list of df) trialized data
            label_array: (list of lists) vectorized labels

        Returns: (df)

        """
        #assert(len(trials) == len(label_array)), "Number of trials in data and labels must match!" TODO test vectorize
        # iterate over labels and trials
        ftrials = []
        for i, label in enumerate(label_array):
            trial = trials[i]
            label_trial_num = int(label[0])
            # match bodypart names
            trial_size = len(trial.index)
            trial.index = np.arange(trial_size)
            # reshape
            formatted_trial = Preprocessor.stack(Preprocessor.stack(trial))
            # rename column of block df to trial num
            formatted_trial.columns = [label_trial_num]
            # transpose so each row represents a trial
            formatted_trial = formatted_trial.T
            ftrials.append(formatted_trial)
        return ftrials

    @staticmethod
    def create_feat_df(formatted_trials):
        """
        Appends all formatted trials into a single DataFrame.
        Args:
            formatted_trials: (list of dfs)

        Returns: (df) where row represents trial num and columns are features.

        """
        df = formatted_trials[0]
        for trial in formatted_trials[1:]:
            df = df.append(trial, ignore_index=True)
        return df





    def make_ml_feat_labels(self, kin_block, exp_block, label,
                              et, el, window_length=250, pre=10):
        """
        Returns ml feature and label arrays.
        Args:
            kin_block: (df)
            exp_block: (df)
            et: int, coordinate change variable
                    Will take the positional coordinates and put them into the robot reference frame.
            el: int, coordinate change variable
                    Will take the positional coordinates and put them into the robot reference frame.
            window_length (int): trial splitting window length, the number of frames to load data from (default 250)
                    Set to 4-500. 900 is too long.
            pre: int, pre cut off before a trial starts, the number of frames to load data from before start time
                    For trial splitting, set to 10. 50 is too long. (default 10)

        Notes:
            labels and blocks must match!
        """
        # trial-ize data
        #hot_vector, trialized_kin_data, feat_names, trialized_exp_data =\
        #    CU.make_s_f_trial_arrays_from_block(kin_block, exp_block, et, el, wv, window_length, pre)
        self.set_kin_block(kin_block)
        self.set_exp_block(exp_block)
        self.set_label(label)



        # Match with label
        matched_kin_data, matched_exp_data = CU.match_stamps(trialized_kin_data, block_label, exp_data)

        # match kin and exp features
            # create_ML_array args: matched kin array, matched ez array
        c_pos, c_prob = CU.create_ML_array(matched_kin_data, matched_exp_data)

            # append results
         #   c_positions.append(c_pos)
        #    c_probabilities.append(c_prob)

        # resize data
        final_ML_feature_array_XYZ, final_labels_array \
            = CU.stack_ML_arrays(c_positions, vectorized_labels)
        #final_ML_feature_array_prob, _ \
        #    = CU.stack_ML_arrays(c_probabilities, vectorized_labels)

        # concat horizontally XYZ and prob123 ml feature arrays
        # (total num labeled trials x (3*num kin feat)*2 +num exp feat = 174 for XYZ and prob, window_length+pre)
        #final_ML_feature_array = np.concatenate((final_ML_feature_array_XYZ, final_ML_feature_array_prob),
         #                                       axis=1)  # this causes issues with norm/zscore




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.function == 1:
        preprocessor = Preprocessor()
