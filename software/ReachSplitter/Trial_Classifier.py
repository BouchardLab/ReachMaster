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
import matplotlib.pyplot as plt
import sklearn

from scipy import ndimage
import Classification_Utils as CU
import pandas as pd
import numpy as np
import h5py
import random
import joblib  # for saving sklearn models
from imblearn.over_sampling import SMOTE  # for adjusting class imbalances
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imblearn_Pipeline
from collections import Counter
# classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline
# from imblearn.pipeline import Pipeline as imblearnPipeline
from sklearn.feature_selection import SelectKBest  # feature selection
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# set global random seed for reproducibility #
random.seed(246810)
np.random.seed(246810)

# Create folder in CWD to save data and plots #
current_directory = os.getcwd()
folder_name = 'ClassifyTrials'
final_directory = os.path.join(current_directory, folder_name)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)


class ReachClassifier:
    # set random set for reproducibility
    random.seed(246810)
    np.random.seed(246810)

    def __init__(self, model=None):
        self.model = model
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.fs = None

    def set_model(self, data):
        self.model = data

    def set_X(self, data):
        self.X = data

    def set_y(self, data):
        self.y = data

    def set_X_train(self, data):
        self.X_train = data

    def set_y_train(self, data):
        self.y_train = data

    def set_X_val(self, data):
        self.X_val = data

    def set_y_val(self, data):
        self.y_val = data

    def set_fs(self, data):
        self.fs = data

    def fit(self, X, y):
        """
        Fits model to data.
        Args:
            X: features
            y: labels

        Returns: None

        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Returns trained model predictions.
        Args:
            X: features
            y: labels

        Returns: preds

        """
        return self.model.predict(X)

    @staticmethod
    def partition(X, y):
        """
        Partitions data.
        Args:
            X: features
            y: labels

        Returns: X_train, X_val, y_train, y_val

        """
        # partition into validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        return X_train, X_val, y_train, y_val

    @staticmethod
    def evaluate(model, X, y):
        """
        Performs 5-fold cross-validation and returns accuracy.
        Args:
            model: sklearn model
            X: features
            y: labels

        Returns: avg_train_accuracy, avg_test_accuracy

        """
        print("Cross validation:")
        cv_results = cross_validate(model, X, y, cv=5, return_train_score=True)
        train_results = cv_results['train_score']
        test_results = cv_results['test_score']
        avg_train_accuracy = sum(train_results) / len(train_results)
        avg_test_accuracy = sum(test_results) / len(test_results)

        print('averaged train accuracy:', avg_train_accuracy)
        print('averaged validation accuracy:', avg_test_accuracy)

        return avg_train_accuracy, avg_test_accuracy

    @staticmethod
    def adjust_class_imbalance(X, y):
        """
        Adjusts for class imbalance.
            Object to over-sample the minority class(es) by picking samples at random with replacement.
            The dataset is transformed, first by oversampling the minority class, then undersampling the majority class.
        Returns: new samples
        References: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
        """
        oversampler = SMOTE(random_state=42)
        # undersampler = RandomUnderSampler(random_state=42)
        steps = [('o', oversampler)]  # , ('u', undersampler)]
        pipeline = imblearn_Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(X, y)
        return X_res, y_res

    @staticmethod
    def hyperparameter_tuning(X_train, X_val, y_train, y_val, model, param_grid, fullGridSearch=False):
        """
        Performs hyperparameter tuning and returns best trained model.
        Args:
            model: sklearn
            param_grid: grid of models and hyperparameters
            fullGridSearch: True to run exhaustive param search, False runs RandomizedSearchCV

        Returns:
            tuned model
            parameters found through search
            accuracy of tuned model

        Reference: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        """

        # Use the random grid to search for best hyperparameters
        if fullGridSearch:
            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                       cv=3, n_jobs=-1, verbose=2)

        else:
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=2, cv=5,
                                             random_state=42, verbose=2, n_jobs=-1)

        # Fit the random search model
        grid_search.fit(X_train, y_train)

        base_model = RandomForestClassifier()
        base_model.fit(X_train, y_train)
        base_train_accuracy, base_test_accuracy = ReachClassifier.evaluate(base_model, X_val, y_val)

        best_grid = grid_search
        best_model = grid_search.best_estimator_
        best_train_accuracy, best_test_accuracy = ReachClassifier.evaluate(best_model, X_val, y_val)

        print('Improvement % of', (100 * (best_test_accuracy - base_test_accuracy) / base_test_accuracy))

        return best_model, best_grid.best_params_, best_test_accuracy

    @staticmethod
    def mean_df(df):
        """
        Maps np.mean to all cells in df. For generating features.
        Args:
            df: (df)

        Returns: df with mean of each cell as its values

        """
        mean_df = df.applymap(np.mean)
        return mean_df

    @staticmethod
    def do_feature_selection(X, y, k):
        """
        Defines the feature selection and applies the feature selection procedure to the dataset.
        Fit to data, then transform it.
        Args:
            k: top number of features to select

        Returns: (array shape trials x k features) subset of the selected input features and feature estimator

        references: https://machinelearningmastery.com/feature-selection-with-numerical-input-data/

        """
        # configure to select a subset of features
        fs = SelectKBest(score_func=f_classif, k=k)
        # learn relationship from training data
        fs.fit(X, y)
        # transform train input data
        X_train_fs = fs.transform(X)
        return X_train_fs, fs

    @staticmethod
    def plot_features(fs, X):
        """
        Plots and saves feature importances.
        Returns: None

        """
        for i in range(len(fs.scores_)):
            print('Feature %d: %f' % (i, fs.scores_[i]))
        # plot the scores
        # x = [i for i in range(len(fs.scores_))]
        x = X.columns
        plt.bar(x, fs.scores_)
        # rotate x axis to avoid overlap
        plt.xticks(rotation=45)
        plt.yticks(rotation=90)
        plt.title("Input Features vs. Feature Importance")
        plt.ylabel("Mutual Information Feature Importance")
        plt.xlabel("Input Features")
        plt.savefig(f'{folder_name}/feat_importance.png')

    @staticmethod
    def pre_classify(X, y, k=10):
        """
        Partitions, adjusts class imbalance, and performs feature selection.
        Args:
            X: features
            y: labels
            k: (int) number of features to select

        Returns: data ready for ML classification

        """
        # adjust class imbalance
        X_res, y_res = ReachClassifier.adjust_class_imbalance(X, y)
        # feat selection
        X_selected, fs = ReachClassifier.do_feature_selection(X_res, y_res, k)
        return X_selected, y_res, fs

    @staticmethod
    def train_and_validate(X, y, param_grid, save=True, filename=None):
        """
        Trains and Validates.
        Args:
            X: features
            y: labels
            param_grid: model and hyperparameters to search over
            save: (bool) True to save model
            filename: (str) name of model to save as

        Returns: trained model, train model's CV score

        """
        # partition
        X_train, X_val, y_train, y_val = ReachClassifier.partition(X, y)
        # hyperparameter and model tuning
        base_model = Pipeline(steps=[('standardscaler', StandardScaler()),
                                     ('classifier', RandomForestClassifier())])
        best_model, best_params_, best_test_accuracy = ReachClassifier.hyperparameter_tuning(
            X_train, X_val, y_train, y_val, base_model, param_grid, fullGridSearch=False)

        # fit and validate
        best_model.fit(X_train, y_train)
        _, val_score = ReachClassifier.evaluate(best_model, X_val, y_val)

        # fit on all training data
        best_model.fit(X, y)

        # save model
        if save:
            joblib.dump(best_model, f"{filename}.joblib")

        # print("MODEL SCORE", best_model.score(X_val_selected, y_val))
        print("BEST MODEL", best_model)
        print("CV SCORE", val_score)
        return best_model, val_score


class ClassificationHierarchy:
    random.seed(246810)
    np.random.seed(246810)

    def __init__(self):
        pass

    def split(self, preds, X, y, onesGoLeft=True):
        """
        Splits X and y based on predictions.
        Args:
            preds: (list of ints) predictions of ones and zeros
            X: features
            y: labels
            onesGoLeft: (bool) True for labels with prediction 1 to be on LHS.

        Returns: split X, y data

        """
        row_mask = list(map(bool, preds))  # True for 1, False otherwise
        negate_row_mask = ~np.array(row_mask)  # True for 0, False otherwise

        if onesGoLeft:
            X_left = X[row_mask]
            y_left = y[row_mask]
            X_right = X[negate_row_mask]
            y_right = y[negate_row_mask]
        else:
            X_right = X[row_mask]
            y_right = y[row_mask]
            X_left = X[negate_row_mask]
            y_left = y[negate_row_mask]

        return X_left, y_left, X_right, y_right

    def run_hierarchy(self, X, y, param_grid, models, save_models):
        """
        Makes predictions through the whole classification hierarchy.
        Args:
            X: features
            y: labels (Trial Type	Num Reaches	Which Hand)
            param_grid: grid
            models: (list) list of trained models or None
            save_models: (bool) True to save

        Returns:

        """
        # load models
        # model_0, model_1, model_2 = None, None, None
        # if models:
        #    model_0 = joblib.load(models[0])
        #    model_1 = joblib.load(models[1])
        #    model_2 = joblib.load(models[2])

        # TRIAL TYPE
        classifier = ReachClassifier()
        y_0 = y['Trial Type'].values  # 0 for not null
        y_0 = CU.onehot_nulls(y_0)
        model_0, val_score_0 = self.fit(classifier, X, y_0, param_grid, save_models,
                                        f'{folder_name}/TrialTypeModel')

        # SPLIT
        # X_null, y_null, X_NotNull, y_NotNull = self.split(preds_0, X, y, onesGoLeft=True)  # 1 if null, 0 if real trial

        # NUM REACHES
        y_1 = y['Num Reaches'].values
        y_1 = CU.onehot_num_reaches(y_1)  # 0 if <1, 1 if > 1 reaches
        classifier = ReachClassifier()
        model_1, val_score_1 = self.fit(classifier, X, y_1, param_grid, save_models,
                                        f'{folder_name}/NumReachesModel')

        # SPLIT
        # X_greater, y_greater, X_less, y_less = self.split(preds_1, X_NotNull, y_NotNull, onesGoLeft=True)  # 0 if <1, 1 if > 1 reaches

        # WHICH HAND
        classifier = ReachClassifier()
        y_2 = y['Which Hand'].values  # # classify 0 as r/l
        y_2 = CU.hand_type_onehot(y_2)
        model_2, val_score_2 = self.fit(classifier, X, y_2, param_grid, save_models,
                                        f'{folder_name}/WhichHandModel')

        # X_bi, y_bi, X_rl, y_rl = self.split(preds_2, X_less, y_less, onesGoLeft=True)  # classify 0 as r/l, 1 or non r/l

        return [val_score_0, val_score_1, val_score_2]

    def fit(self, classifier, X, y, param_grid, save, filename):
        """
        Trains, validates, and/or makes predictions.
        Args:
            classifier: ReachClassifier object
            X: features
            y: labels
            param_grid: grid
            save: (bool) True to save
            filename: (str) file name to save model as
            best_model: model
            doFit: (bool) True to train

        Returns: model, validation score, predicitons

        """
        # adjust class imbalance, feature selection
        X_selected, y_res, fs = classifier.pre_classify(X, y)
        # train and validate
        assert (y is not None)
        best_model, val_score = classifier.train_and_validate(X_selected, y_res, param_grid, save=save,
                                                              filename=filename)
        return best_model, val_score

    def run_hierarchy_pretrained(self, X, y, models):
        """
        Makes predictions through the whole classification hierarchy.
        Args:
            X: features
            y: labels (Trial Type	Num Reaches	Which Hand)
            models: (list of str) list of trained models
        Returns: list of validation accuracies

        """
        # load models
        model_0 = joblib.load(models[0])
        model_1 = joblib.load(models[1])
        model_2 = joblib.load(models[2])

        # TRIAL TYPE
        classifier = ReachClassifier()
        y_0 = y['Trial Type'].values  # 0 for not null
        y_0 = CU.onehot_nulls(y_0)
        val_score_0 = self.predict(X, y_0, model_0)

        # SPLIT
        # X_null, y_null, X_NotNull, y_NotNull = self.split(preds_0, X, y, onesGoLeft=True)  # 1 if null, 0 if real trial

        # NUM REACHES
        y_1 = y['Num Reaches'].values
        y_1 = CU.onehot_num_reaches(y_1)  # 0 if <1, 1 if > 1 reaches
        classifier = ReachClassifier()
        val_score_1 = self.predict(X, y_1, model_1)

        # SPLIT
        # X_greater, y_greater, X_less, y_less = self.split(preds_1, X_NotNull, y_NotNull, onesGoLeft=True)  # 0 if <1, 1 if > 1 reaches

        # WHICH HAND
        classifier = ReachClassifier()
        y_2 = y['Which Hand'].values  # # classify 0 as r/l
        y_2 = CU.hand_type_onehot(y_2)
        val_score_2 = self.predict(X, y_2, model_2)

        # X_bi, y_bi, X_rl, y_rl = self.split(preds_2, X_less, y_less, onesGoLeft=True)  # classify 0 as r/l, 1 or non r/l

        return [val_score_0, val_score_1, val_score_2]

    def predict(self, X, y, model):
        # let
        k = 5
        X_selected, fs = ReachClassifier.do_feature_selection(X, y, k)
        _, val_score = ReachClassifier.evaluate(model, X_selected, y)
        return val_score

    def trace_datapoint(self, X, arr=[]):
        """ Q3.2
        for a data point from the spam dataset, prints splits and thresholds
        as it is classified down the tree.
        """
        pass


class MakeFeatures:
    # Operates on a single trial.

    pos_names = ['Handle', 'Back Handle', 'Nose',
                 'Left Shoulder', 'Left Forearm', 'Left Wrist', 'Left Palm', 'Left Index Base', 'Left Index Tip',
                 'Left Middle Base', 'Left Middle Tip', 'Left Third Base',
                 'Left Third Tip', 'Left Fourth Finger Base', 'Left Fourth Finger Tip',
                 'Right Shoulder', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Right Index Base',
                 'Right Index Tip', 'Right Middle Base', 'Right Middle Tip', 'Right Third Base',
                 'Right Third Tip', 'Right Fourth Finger Base', 'Right Fourth Finger Tip']

    def __init__(self, trial_arr):
        # partition coords and probabilities
        self.num_bodyparts = 27
        self.num_coords = 3
        self.split_index = self.num_bodyparts * self.num_coords  # 27 bodyparts * 3 XYZ coordinates for each = 81
        self.coords = trial_arr[:self.split_index]  # all XYZ coords of all bodyparts (81 rows of first half of array)
        self.prob = trial_arr[self.split_index:]  # all probability columns (81 rows of second half of array)

        # display(coords, prob)

    def calc_position(self):
        # calculate position of each bodypart (x+y+z/3)
        positions = []  # 2D array with rows are bodyparts, cols are frame nums
        for i in np.arange(0, len(self.coords), self.num_coords):  # for every bodypart
            X = self.coords[i]
            Y = self.coords[i + 1]
            Z = self.coords[i + 2]
            pos = (X + Y + Z) / self.num_coords  # 1D array
            positions.append(pos)

        assert (len(positions) == self.num_bodyparts)
        return positions

    def calc_velocity_speed(self, time):
        """
        Time is sliced from exp block 'time' column
        """
        # calculate velocity for each XYZ bodypart (x1-x0/t0-t1)
        velocities = []  # 2D array with rows are XYZ bodyparts, cols are frame nums
        for i in np.arange(0, self.split_index, self.num_coords):  # for every bodypart
            X = self.coords[i]
            Y = self.coords[i + 1]
            Z = self.coords[i + 2]

            for arr in [X, Y, Z]:
                vel = []
                for j in np.arange(len(arr) - 1):
                    x_0 = arr[j]
                    x_1 = arr[j + 1]
                    t_0 = time[j]
                    t_1 = time[j + 1]
                    vel.append(x_1 - x_0 / t_1 - t_0)
                velocities.append(vel)

        assert (len(velocities) == self.split_index)

        # calculate speed of each bodypart (vel_x+vel_y+vel_z/3)
        speeds = []  # 1D array with rows are bodyparts, cols are frame nums
        for i in np.arange(0, self.split_index, self.num_coords):
            x_vel = velocities[i]
            y_vel = velocities[i + 1]
            z_vel = velocities[i + 2]

            x_squared = np.dot(x_vel, x_vel)
            y_squared = np.dot(y_vel, y_vel)
            z_squared = np.dot(z_vel, z_vel)

            speed = (x_squared + y_squared + z_squared) / 3  # int
            speeds.append(speed)

        assert (len(speeds) == self.num_bodyparts)
        return velocities, speeds

    @staticmethod
    def calc_all(trial_arr, time):
        # Calculate
        f = MakeFeatures(trial_arr)
        positions = f.calc_position()
        velocities, speeds = f.calc_velocity_speed(time)

        # take mean & median of each bodypart for 2D arrays
        mean_vel = np.mean(velocities, axis=1)  # len = 81
        median_vel = np.median(velocities, axis=1)

        mean_pos = np.mean(positions, axis=1)  # len = 27
        median_pos = np.median(positions, axis=1)

        # Create df
        # concat all arrays
        speeds.extend(mean_pos)
        speeds.extend(median_pos)
        speeds.extend(mean_vel)
        speeds.extend(median_vel)
        # create col names
        col_names = [bodypart + ' speed' for bodypart in f.pos_names]
        col_names.extend([bodypart + ' mean pos' for bodypart in f.pos_names])
        col_names.extend([bodypart + ' median pos' for bodypart in f.pos_names])
        xzy_pos_names = [bodypart + ' X' for bodypart in f.pos_names] + [bodypart + ' Y' for bodypart in
                                                                         f.pos_names] + [bodypart + ' Z' for bodypart in
                                                                                         f.pos_names]
        col_names.extend([bodypart + ' mean vel' for bodypart in xzy_pos_names])
        col_names.extend([bodypart + ' median vel' for bodypart in xzy_pos_names])
        # create df
        df = pd.DataFrame([speeds], columns=col_names)
        return df

    @staticmethod
    def make_block_features(trials, times):
        df = pd.DataFrame()
        for i in range(len(trials)):
            # take trial
            trial = trials[i]
            time = times[i]

            trial_arr = trial.values  # convert df to array where rows are frame numbers, cols are bodyparts
            trial_arr = trial_arr.T  # array with rows are XYZ bodyparts, cols are frame nums

            df = pd.concat([df, MakeFeatures.calc_all(trial_arr, time)])
        df.reset_index(drop=True, inplace=True)

        # rows are trials, cols are features
        return df

    @staticmethod
    def match_labels(df, vec_label):
        # create mask of labeled trials
        labeled_trials_mask = []
        for i, label in enumerate(vec_label):
            label_trial_num = int(label[0])
            labeled_trials_mask.append(label_trial_num)

        return df.T[labeled_trials_mask].T

    @staticmethod
    def sel_feat_by_keyword(df):
        """
        reference: https://towardsdatascience.com/interesting-ways-to-select-pandas-dataframe-columns-b29b82bbfb33
        """
        return df.loc[:, [('Palm' in i) or ('Wrist' in i) for i in df.columns]]

    @staticmethod
    def randomize_feat(df):
        return df.sample(n=len(df), replace=False, axis=0, random_state=42)  # shuffles rows w.o repl


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
        assert file_type == 'csv' or file_type == 'pkl' or file_type == 'h5', f'{file_type} not a valid file type'
        if file_type == 'csv':
            df.to_csv(filename)
        if file_type == 'pkl':
            df.to_pickle(filename)
        if file_type == 'h5':
            df.to_hdf(filename, key='df')

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
        assert (window_length > pre), "invalid slice!"
        starting_frames = exp_block['r_start'].values[0]
        trials = []
        times = []
        # iterate over starting frames
        for frame_num in starting_frames:
            start = frame_num - pre
            # negative indices case
            if (frame_num - pre) <= 0:
                start = 0
            # slice trials
            trials.append(formatted_kin_block.loc[start:frame_num + window_length])
            times.append(
                exp_block['time'][0][start:frame_num + window_length + 1])  # plus 1 to adjust size diff with trial size
        return trials, times

    @staticmethod
    def trialize_kin_blocks(formatted_kin_block, times):
        """
        Returns a list of one column dfs, each representing a trial
        Args:
            formatted_kin_block: (list of dfs) split trial data
            times: (list of arrays of ints) sliced time from exp block

        Returns: ftrials: (list of one column dfs)
        """
        # iterate over trials
        ftrials = []
        for trial in formatted_kin_block:
            # match bodypart names
            trial_size = len(trial.index)
            trial.index = np.arange(trial_size)
            # reshape df into one column for one trial
            formatted_trial = Preprocessor.stack(Preprocessor.stack(trial))
            ftrials.append(formatted_trial)
        return ftrials

    @staticmethod
    def match_kin_to_label(formatted_kin_block, label):
        """
        Selects labeled trials and matches them to their labels.
        Args:
            formatted_kin_block: (list of one column dfs) trialized data
            label: (list of lists) vectorized labels

        Returns: labeled_trials: (list of one row dfs) matched to labels

        Note:
            If a trial is not labeled, the trial is dropped and unused.
            Trial numbers are zero-indexed.

        """
        assert (len(label) <= len(formatted_kin_block)), \
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
        trials, times = Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        ftrials = Preprocessor.trialize_kin_blocks(trials)
        labeled_trials = Preprocessor.match_kin_to_label(ftrials, self.label)
        df = Preprocessor.create_kin_feat_df(labeled_trials)
        self.set_formatted_kin_block(df)
        return df

    def make_kin_psv_feat_df(self, randomize=False):
        """

        Returns: feature df of position, speed, and velocity

        """
        trials, times = Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        df = MakeFeatures.make_block_features(trials, times)
        df = MakeFeatures.match_labels(df, self.label)
        ret_df = MakeFeatures.sel_feat_by_keyword(df)  # select just wrist and palms
        if randomize:
            return MakeFeatures.randomize_feat(ret_df)
        return ret_df

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
        assert (len(label) <= len(exp_feat_df)), \
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

        # match and expand
        masked_exp_feat_df = Preprocessor.match_exp_to_label(exp_feat_df, self.label)

        # update attribute
        self.set_formatted_exp_block(masked_exp_feat_df)
        return self.formatted_exp_block

    @staticmethod
    def concat(dfs, row=True):
        """
        Concats a list of dataframes row or col-wise
        Args:
            dfs: (list of dfs) to concat
            row: (bool) True to concat by row

        Returns: new df

        """
        assert (len(dfs) >= 2), "Must concat at least 2 dfs!"
        if row:
            df_0 = dfs[0]
            for df in dfs[1:]:
                assert (df_0.shape[1] == df.shape[1]), f'{df_0.shape} {df.shape} cols must match!'
                df_0 = pd.concat([df_0, df], axis=0)
        else:
            df_0 = dfs[0]
            for df in dfs[1:]:
                assert (df_0.shape[0] == df.shape[0]), f'{df_0.shape} {df.shape} rows must match!'
                df_0 = pd.concat([df_0, df], axis=1)
        return df_0

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
        self.set_wv(wv)  # must be set before kin block
        self.set_window_length(window_length)
        self.set_pre(pre)
        self.set_kin_block(kin_block)

        # vectorize label
        vectorized_label, _ = CU.make_vectorized_labels(label)
        self.set_label(vectorized_label)

        # create kin features
        # kin_feat_df = self.make_kin_feat_df()
        kin_feat_df = self.make_kin_psv_feat_df()  # todo randomize=True to change features

        # create exp features
        exp_feat_df = self.make_exp_feat_df()

        return kin_feat_df, exp_feat_df


def main_run_all():
    # LOAD DATA
    preprocessor = Preprocessor()
    exp_data = preprocessor.load_data('experimental_data.pickle')
    tkdf_16 = preprocessor.load_data('tkdf16_f.pkl')
    tkdf_15 = preprocessor.load_data('3D_positions_RM15_f.pkl')
    tkdf_14 = preprocessor.load_data('3D_positions_RM14_f.pkl')

    # GET and SAVE BLOCKS
    # (df, date, session, rat, save_as=None, format='exp')
    exp_lst = [
        preprocessor.get_single_block(exp_data, '0190917', 'S1', 'RM16', format='exp',
                                      save_as=f'{folder_name}/exp_rm16_9_17_s1.pkl'),
        preprocessor.get_single_block(exp_data, '0190918', 'S1', 'RM16', format='exp',
                                      save_as=f'{folder_name}/exp_rm16_9_18_s1.pkl'),
        preprocessor.get_single_block(exp_data, '0190917', 'S2', 'RM16', format='exp',
                                      save_as=f'{folder_name}/exp_rm16_9_17_s2.pkl'),
        preprocessor.get_single_block(exp_data, '0190920', 'S3', 'RM16', format='exp',
                                      save_as=f'{folder_name}/exp_rm16_9_20_s3.pkl'),
        preprocessor.get_single_block(exp_data, '0190919', 'S3', 'RM16', format='exp',
                                      save_as=f'{folder_name}/exp_rm16_9_19_s3.pkl'),
        preprocessor.get_single_block(exp_data, '0190925', 'S3', 'RM15', format='exp',
                                      save_as=f'{folder_name}/exp_rm15_9_25_s3.pkl'),
        preprocessor.get_single_block(exp_data, '0190917', 'S4', 'RM15', format='exp',
                                      save_as=f'{folder_name}/exp_rm15_9_17_s4.pkl'),
        preprocessor.get_single_block(exp_data, '0190920', 'S1', 'RM14', format='exp',
                                      save_as=f'{folder_name}/exp_rm14_9_20_s1.pkl'),
        preprocessor.get_single_block(exp_data, '0190918', 'S2', 'RM14', format='exp',
                                      save_as=f'{folder_name}/exp_rm14_9_18_s2.pkl')
    ]

    kin_lst = [
        preprocessor.get_single_block(tkdf_16, '0190917', 'S1', '09172019', format='kin',
                                      save_as=f'{folder_name}/kin_rm16_9_17_s1.pkl'),
        preprocessor.get_single_block(tkdf_16, '0190918', 'S1', '09182019', format='kin',
                                      save_as=f'{folder_name}/kin_rm16_9_18_s1.pkl'),
        preprocessor.get_single_block(tkdf_16, '0190917', 'S2', '09172019', format='kin',
                                      save_as=f'{folder_name}/kin_rm16_9_17_s2.pkl'),
        preprocessor.get_single_block(tkdf_16, '0190920', 'S3', '09202019', format='kin',
                                      save_as=f'{folder_name}/kin_rm16_9_20_s3.pkl'),
        preprocessor.get_single_block(tkdf_16, '0190919', 'S3', '09192019', format='kin',
                                      save_as=f'{folder_name}/kin_rm16_9_19_s3.pkl'),
        preprocessor.get_single_block(tkdf_15, '0190925', 'S3', '09252019', format='kin',
                                      save_as=f'{folder_name}/kin_rm15_9_25_s3.pkl'),
        preprocessor.get_single_block(tkdf_15, '0190917', 'S4', '09172019', format='kin',
                                      save_as=f'{folder_name}/kin_rm15_9_17_s4.pkl'),
        preprocessor.get_single_block(tkdf_14, '0190920', 'S1', '09202019', format='kin',
                                      save_as=f'{folder_name}/kin_rm14_9_20_s1.pkl'),
        preprocessor.get_single_block(tkdf_14, '0190918', 'S2', '09182019', format='kin',
                                      save_as=f'{folder_name}/kin_rm14_9_18_s2.pkl')
    ]

    """# CREATE FEAT and LABEL DFS
    kin_dfs = []
    exp_dfs = []
    label_dfs = []
    for i in range(len(kin_lst)):
        kin_block = kin_lst[i]
        exp_block = exp_lst[i]
        label = labels[i]

        kin_feat_df, exp_feat_df = preprocessor.make_ml_feat_labels(kin_block, exp_block,
                                                                    label, et, el,
                                                                    window_length, pre,
                                                                    wv)
        # Check for NaNs and replace with zeros
        if kin_feat_df.isnull().values.any():
            print(f"{i}th Kin Block contains Nan!")
            for column in kin_feat_df:
                if kin_feat_df[column].isnull().values.any():
                    print(f"Kin '{kin_feat_df[column]}' contains NaN and replaced with 0!")
            kin_feat_df.fillna(0)

        if exp_feat_df.isnull().values.any():
            print(f"{i}th Exp Block contains Nan!")
            for column in kin_feat_df:
                if exp_feat_df[column].isnull().values.any():
                    print(f" Exp '{exp_feat_df[column]}' contains NaN and replaced with 0!")
            exp_feat_df.fillna(0)

        # append
        vec_labels, _ = CU.make_vectorized_labels(label)
        label_df = CU.make_vectorized_labels_to_df(vec_labels)
        label_dfs.append(label_df)
        kin_dfs.append(kin_feat_df)
        exp_dfs.append(exp_feat_df)

    # concat
    all_kin_features = Preprocessor.concat(kin_dfs, row=True)
    all_exp_features = Preprocessor.concat(exp_dfs, row=True)
    all_label_dfs = Preprocessor.concat(label_dfs, row=True)

    # save ML dfs
    Preprocessor.save_data(all_kin_features, f'{folder_name}/kin_feat.pkl', file_type='pkl')
    Preprocessor.save_data(all_exp_features, f'{folder_name}/exp_feat.pkl', file_type='pkl')
    Preprocessor.save_data(all_label_dfs, f'{folder_name}/label_dfs.pkl', file_type='pkl')
    """

def create_features():
    # GET SAVED BLOCKS
    # (df, date, session, rat, save_as=None, format='exp')
    exp_lst = [
        [f'{folder_name}/exp_rm16_9_17_s1.pkl',
         f'{folder_name}/exp_rm16_9_18_s1.pkl',
         f'{folder_name}/exp_rm16_9_17_s2.pkl',
         f'{folder_name}/exp_rm16_9_20_s3.pkl',
         f'{folder_name}/exp_rm16_9_19_s3.pkl'],

        [f'{folder_name}/exp_rm15_9_25_s3.pkl',
         f'{folder_name}/exp_rm15_9_17_s4.pkl'],

        [f'{folder_name}/exp_rm14_9_20_s1.pkl',
         f'{folder_name}/exp_rm14_9_18_s2.pkl']
    ]

    kin_lst = [
        [f'{folder_name}/kin_rm16_9_17_s1.pkl',
         f'{folder_name}/kin_rm16_9_18_s1.pkl',
         f'{folder_name}/kin_rm16_9_17_s2.pkl',
         f'{folder_name}/kin_rm16_9_20_s3.pkl',
         f'{folder_name}/kin_rm16_9_19_s3.pkl'],

        [f'{folder_name}/kin_rm15_9_25_s3.pkl',
         f'{folder_name}/kin_rm15_9_17_s4.pkl'],

        [f'{folder_name}/kin_rm14_9_20_s1.pkl',
         f'{folder_name}/kin_rm14_9_18_s2.pkl']
    ]

    # Define data paths
    tkdf_16 = 'DataFrames/tkdf16_f.pkl'
    tkdf_15 = 'DataFrames/3D_positions_RM15_f.pkl'
    tkdf_14 = 'DataFrames/3D_positions_RM14_f.pkl'
    RM16_expdf = 'DataFrames/RM16_expdf.pickle'
    RM15_expdf = 'DataFrames/RM15_expdf.pickle'
    RM14_expdf = 'DataFrames/RM14_expdf.pickle'

    # RM16 Blocks
    blocks_16 = [['17', 'S1', 'RM16'],  # issue
                 ['18', 'S1', 'RM16'],
                 ['17', 'S2', 'RM16'],
                 ['20', 'S3', 'RM16'],
                 ['19', 'S3', 'RM16']]
    # RM15 Blocks
    blocks_15 = [['25', 'S3', 'RM15'],
                 ['17', 'S4', 'RM15']] # issue

    # RM14 Blocks
    blocks_14 = [['20', 'S1', 'RM14'],
                 ['18', 'S2', 'RM14']]

    # Append paths
    kin_paths = [tkdf_16, tkdf_15, tkdf_14]
    exp_paths = [RM16_expdf, RM15_expdf, RM14_expdf]
    block_paths = [blocks_16, blocks_15, blocks_14]

    # Get labels
    # labels
    # RM16_9_17_s1
    # RM16, 9-18, S1
    # RM16, 9-17, S2
    # RM16, DATE 9-20, S3
    # RM16, 09-19-2019, S3
    # RM15, 25, S3
    # RM15, 17, S4
    # 2019-09-20-S1-RM14_cam2
    # 2019-09-18-S2-RM14-cam2
    labels = [[CU.rm16_9_17_s1_label,
               CU.rm16_9_18_s1_label,
               CU.rm16_9_17_s2_label,
               CU.rm16_9_20_s3_label,
               CU.rm16_9_19_s3_label],

              [CU.rm15_9_25_s3_label,
               CU.rm15_9_17_s4_label],

              [CU.rm14_9_20_s1_label,
               CU.rm14_9_18_s2_label]
              ]

    # CREATE FEAT and LABEL DFS
    feat_dfs = []
    label_dfs = []
    for i in range(len(kin_paths)):  # for each rat
        for j in range(len(block_paths[i])):
            kin_data = Preprocessor.load_data(kin_lst[i][j])
            exp_data = Preprocessor.load_data(exp_lst[i][j])
            date, session, rat = block_paths[i][j]
            label = labels[i][j]

            # Run ReachUtils
            R = CU.ReachUtils(rat, date, session, exp_data, kin_data, 's')  # init
            data = R.create_and_save_classification_features()
            print("SAVED block")

            # append
            vec_labels, _ = CU.make_vectorized_labels(label)
            label_df = CU.make_vectorized_labels_to_df(vec_labels)
            label_dfs.append(label_df)
            feat_dfs.append(data)

    # concat
    # all_feat_dfs = Preprocessor.concat(feat_dfs, row=True)
    all_label_dfs = Preprocessor.concat(label_dfs, row=True)

    # save ML dfs
    # Preprocessor.save_data(all_feat_dfs, f'{folder_name}/feat_dfs.pkl', file_type='h5')
    Preprocessor.save_data(all_label_dfs, f'{folder_name}/label_dfs.pkl', file_type='h5')


def main_run_ML():
    """
    Train models on sythetic data.
    Returns:

    """
    # LOAD DATA
    preprocessor = Preprocessor()
    all_kin_features = preprocessor.load_data(f'{folder_name}/kin_feat.pkl', file_type='pkl')
    all_exp_features = preprocessor.load_data(f'{folder_name}/exp_feat.pkl', file_type='pkl')
    y = preprocessor.load_data(f'{folder_name}/label_dfs.pkl', file_type='pkl')

    # take mean of exp features
    all_exp_features = ReachClassifier.mean_df(all_exp_features)
    # remove unused features
    all_exp_features = all_exp_features.drop(columns=['unused idx 4', 'unused idx 5', 'unused idx 6'])

    # concat kin and exp features
    all_kin_features.reset_index(drop=True, inplace=True)
    all_exp_features.reset_index(drop=True, inplace=True)
    X = Preprocessor.concat([all_kin_features, all_exp_features], row=False)
    # todo save x

    # TRAIN and SAVE MODELS
    t = ClassificationHierarchy()
    t.run_hierarchy(X, y, param_grid, models=None, save_models=True)


def predict_blocks():
    # LOAD DATA
    preprocessor = Preprocessor()
    all_kin_features = preprocessor.load_data(f'{folder_name}/kin_feat.pkl', file_type='pkl')
    all_exp_features = preprocessor.load_data(f'{folder_name}/exp_feat.pkl', file_type='pkl')
    y = preprocessor.load_data(f'{folder_name}/label_dfs.pkl', file_type='pkl')

    # take mean of exp features
    all_exp_features = ReachClassifier.mean_df(all_exp_features)
    # remove unused features
    all_exp_features = all_exp_features.drop(columns=['unused idx 4', 'unused idx 5', 'unused idx 6'])

    # concat kin and exp features
    all_kin_features.reset_index(drop=True, inplace=True)
    all_exp_features.reset_index(drop=True, inplace=True)
    X = Preprocessor.concat([all_kin_features, all_exp_features], row=False)
    # todo load x

    # load models
    models = [f'{folder_name}/TrialTypeModel.joblib', f'{folder_name}/NumReachesModel.joblib',
              f'{folder_name}/WhichHandModel.joblib']

    vals = ClassificationHierarchy().run_hierarchy_pretrained(X, y, models)
    print(vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    # define params for trializing blocks
    et = 0
    el = 0
    wv = 5
    window_length = 4  # TODO change to preferences, default = 250
    pre = 2  # TODO change to preferences, default = 10

    # labels
    # RM16_9_17_s1
    # RM16, 9-18, S1
    # RM16, 9-17, S2
    # RM16, DATE 9-20, S3
    # RM16, 09-19-2019, S3
    # RM15, 25, S3
    # RM15, 17, S4
    # 2019-09-20-S1-RM14_cam2
    # 2019-09-18-S2-RM14-cam2
    labels = [CU.rm16_9_17_s1_label,
              CU.rm16_9_18_s1_label,
              CU.rm16_9_17_s2_label,
              CU.rm16_9_20_s3_label,
              CU.rm16_9_19_s3_label,

              CU.rm15_9_25_s3_label,
              CU.rm15_9_17_s4_label,

              CU.rm14_9_20_s1_label,
              CU.rm14_9_18_s2_label
              ]

    param_grid = [
        {'classifier': [LogisticRegression()],
         'classifier__penalty': ['l1'],
         'classifier__C': [100, 80, 60, 40, 20, 15, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
         'classifier__solver': ['newton-cg', 'liblinear']}
        # {'classifier': [RandomForestClassifier()],
        # 'classifier__bootstrap': [True, False],
        # 'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        # 'classifier__max_features': ['auto', 'sqrt'],
        # 'classifier__min_samples_leaf': [1, 2, 4],
        # 'classifier__min_samples_split': [2, 5, 10],
        # 'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
        # {'classifier': [sklearn.svm.SVC()],
        # 'classifier__C': [50, 40, 30, 20, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
        # 'classifier__kernel': ['poly', 'rbf', 'sigmoid'],
        # 'classifier__gamma': ['scale']},
        # {'classifier': [RidgeClassifier()],
        # 'classifier__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    ]

    if args.function == 1:
        main_run_all()
    elif args.function == 2:
        main_run_ML()
    elif args.function == 3:
        predict_blocks()
    elif args.function == 4:
        # vis class imbalance (simple scatter)
        # load
        preprocessor = Preprocessor()
        all_kin_features = preprocessor.load_data(f'{folder_name}/kin_feat.pkl',
                                                  file_type='pkl')  # generate via TC.main
        all_exp_features = preprocessor.load_data(f'{folder_name}/exp_feat.pkl')
        all_label_dfs = preprocessor.load_data(f'{folder_name}/label_dfs.pkl')

        # todo change accordingly
        X = all_kin_features
        y = all_label_dfs['Which Hand'].values
        y = CU.hand_type_onehot(y)

        # Calc bias
        # X, y = ReachClassifier.adjust_class_imbalance(X, y)

        # Plot and summarize class distribution
        # Trial Type not null is 0.0, num reaches 0 if <1, which hand 0 as r/l
        counter = Counter(y)
        print(counter)
        X = X.values
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = np.where(y == label)[0]
            legend = f'r/l:{counter[0.0]}' if label == 0.0 else f'rla/lra/bi:{counter[1.0]}'
            plt.scatter(X[row_ix, 0], X[row_ix, 1], label=legend)  # uses first two features
        plt.legend()
        plt.title("Class Distribution: Hand Type")
        plt.savefig(f'{folder_name}/ClassBias_whichhand.png')

    elif args.function == 5:
        create_features()
