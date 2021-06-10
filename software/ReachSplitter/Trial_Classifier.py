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
        self.label = None  # usage: CU.make_vectorized_labels(label)
        self.kin_block = None

        self.exp_block = None
        self.all_exp_blocks = []
        self.all_kin_blocks = []

        # kin block
        self.wv = None
        self.window_length = None
        self.pre = None

        # ML dfs
        self.formatted_kin_block = None  # kinematic feature df
        self.formatted_exp_block = None  # robot feature df

    def set_kin_data(self, data):
        self.kin_data = data

    def set_exp_data(self, data):
        self.exp_data = data

    def set_kin_block(self, data):
        self.kin_block = data
        self.format_kin_block()

    def set_formatted_kin_block(self, data):
        self.formatted_kin_block = data

    def set_exp_block(self, data):
        self.exp_block = data

    def set_formatted_exp_block(self, data):
        self.formatted_exp_block = data

    def set_label(self, data):
        self.label = data

    def set_wv(self, data):
        self.wv = data

    def set_window_length(self, data):
        self.window_length = data

    def set_pre(self, data):
        self.pre = data

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

    def format_kin_block(self):
        """
        Removes rat ID levels of a block df and applies median filter to column values.
        Sets formatted_kin_block to (df) two level multi-index df with filtered values.

        Returns: None

        """
        # rm ID levels
        index = self.kin_block.columns[0]
        rm_levels_df = self.kin_block[index[0]][index[1]][index[2]][index[3]]
        # filter bodypart columns
        filtered_df = Preprocessor.apply_median_filter(rm_levels_df, wv=self.wv)
        # update attribute
        self.set_formatted_kin_block(filtered_df)

    @staticmethod
    def split_trial(formatted_kin_block, exp_block, window_length, pre):
        """
        Partitions kinematic data into trials.

        Args:
            formatted_kin_block: (df) formatted kin block
            exp_block: (df)
            window_length (int): trial splitting window length, the number of frames to load data from (default 250)
                    Set to 4-500. 900 is too long.
            pre: int, pre cut off before a trial starts, the number of frames to load data from before start time
                    For trial splitting, set to 10. 50 is too long. (default 10)

        Returns: trials: (list of dfs) of length number of trials with index trial number

        """
        assert(window_length > pre), "invalid slice!"
        starting_frames = exp_block['r_start'].values[0]
        trials = []
        # iterate over starting frames
        for frame_num in starting_frames:
            start = frame_num - pre
            # negative indices case
            if (frame_num-pre) <= 0:
                start = 0
            # slice trials
            trials.append(formatted_kin_block.loc[start:frame_num + window_length])
        return trials

    @staticmethod
    def trialize_kin_blocks(formatted_kin_block):
        """
        Returns a list of one row dfs, each representing a trial
        Args:
            formatted_kin_block: (list of dfs) split trial data

        Returns: ftrials: (list of one row dfs)

        """
        # iterate over trials
        ftrials = []
        for trial in formatted_kin_block:
            # match bodypart names
            trial_size = len(trial.index)
            trial.index = np.arange(trial_size)
            # reshape df into one row for one trial
            formatted_trial = Preprocessor.stack(Preprocessor.stack(trial))
            ftrials.append(formatted_trial)
        return ftrials

    @staticmethod
    def match_kin_to_label(formatted_kin_block, label):
        """
        Selects labeled trials and matches them to their labels.
        Args:
            formatted_kin_block: (list of dfs) trialized data
            label: (list of lists) vectorized labels

        Returns: labeled_trials: (list of one row dfs) matched to labels

        Note:
            If a trial is not labeled, the trial is dropped and unused.
            Trial numbers are zero-indexed.

        """
        assert(len(label) <= len(formatted_kin_block)),\
            f"More labels {len(label)} than trials {len(formatted_kin_block)}!"
        # iterate over labels and trials
        labeled_trials = []
        for i, label in enumerate(label):
            label_trial_num = int(label[0])
            trialized_df = formatted_kin_block[label_trial_num]  # trial nums are 0-indexed
            # rename column of block df to trial num
            trialized_df.columns = [label_trial_num]
            # transpose so each row represents a trial
            trialized_df = trialized_df.T
            labeled_trials.append(trialized_df)
        return labeled_trials

    @staticmethod
    def create_kin_feat_df(formatted_kin_block):
        """
        Appends all formatted trials into a single DataFrame.

        Args:
            formatted_kin_block: list of formatted dfs

        Returns: df: (df) where row represents trial num and columns are features.

        """
        df = formatted_kin_block[0]
        for trial in formatted_kin_block[1:]:
            df = df.append(trial, ignore_index=True)
        return df

    def make_kin_feat_df(self):
        """
        Given a kinematic block df, returns a ML ready feature df
        Returns: (df)  where row represents trial num and columns are features.

        """
        trials = Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        ftrials = Preprocessor.trialize_kin_blocks(trials)
        labeled_trials = Preprocessor.match_kin_to_label(ftrials, self.label)
        df = Preprocessor.create_kin_feat_df(labeled_trials)
        self.set_formatted_kin_block(df)
        return df

    @staticmethod
    def match_exp_to_label(exp_feat_df, label):
        """
        Selects labeled trials and matches them to their labels.
        Args:
            exp_feat_df: (df) exp df
            label: (list of lists) vectorized labels

        Returns: masked_exp_feat_df: (df) exp feature df matched with labels

        Note:
            If a trial is not labeled, the trial is dropped and unused.
            Trial numbers are zero-indexed.

        """
        assert(len(label) <= len(exp_feat_df)),\
            f"More labels {len(label)} than trials {len(exp_feat_df)}!"
        # match to labels
        labeled_trial_nums = []
        for i, label in enumerate(label):
            labeled_trial_nums.append(int(label[0]))
        # apply mask
        masked_exp_feat_df = exp_feat_df.iloc[labeled_trial_nums]
        return masked_exp_feat_df

    def make_exp_feat_df(self):
        """
        Given a robot block df, returns a ML ready feature df
        Returns: (df)  where row represents trial num and columns are features.

        """
        # create exp features
        start_frames = self.exp_block['r_start'].values[0]
        exp_features = CU.import_experiment_features(self.exp_block, start_frames, self.window_length, self.pre)
        hot_vector = CU.onehot(self.exp_block)  # unused
        exp_feat_df = CU.import_experiment_features_to_df(exp_features)

        masked_exp_feat_df = Preprocessor.match_exp_to_label(exp_feat_df, self.label)

        # update attribute
        self.set_formatted_exp_block(masked_exp_feat_df)
        return self.formatted_exp_block

    def make_ml_feat_labels(self, kin_block, exp_block, label,
                              et, el, window_length=250, pre=10, wv=5):
        """
        Returns ml feature and label arrays.
        Args:
            kin_block: (df)
            exp_block: (df)
            label: (list of list)
            et: int, coordinate change variable
                    Will take the positional coordinates and put them into the robot reference frame.
            el: int, coordinate change variable
                    Will take the positional coordinates and put them into the robot reference frame.
            window_length (int): trial splitting window length, the number of frames to load data from (default 250)
                    Set to 4-500. 900 is too long.
            pre: int, pre cut off before a trial starts, the number of frames to load data from before start time
                    For trial splitting, set to 10. 50 is too long. (default 10)
            wv: (int) the wavelet # for the median filter applied to the positional data

        Notes:
            labels and blocks must match!

            hot_vector: (array) one hot array of robot block data of length num trials
            exp_features: (list) experimental features with shape (Num trials X Features X pre+window_length)
        """
        # init instance attributes

        self.set_exp_block(exp_block)
        self.set_wv(wv)  # must be set first
        self.set_window_length(window_length)
        self.set_pre(pre)
        self.set_kin_block(kin_block)

        # vectorize label
        vectorized_label, _ = CU.make_vectorized_labels(label)
        self.set_label(vectorized_label)

        # create kin features
        kin_feat_df = self.make_kin_feat_df()

        # create exp features
        exp_feat_df = self.make_exp_feat_df()

        # concat results
        assert(kin_feat_df.shape[0] == exp_feat_df.shape[0]), f'{kin_feat_df.shape} {exp_feat_df.shape} rows must match!'
        features = pd.concat([kin_feat_df, exp_feat_df], axis=1)
        return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.function == 1:
        pass
