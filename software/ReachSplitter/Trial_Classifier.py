""" Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 5/18/2022
    Functions to classify trials, which are inputs from the ReachLoader class. Classes that are considered
    are the trial type (null, reach present), the number of reaches (1 vs multiple), and the handedness (L/R).
    If multiple reaches are present, additional classification is performed to determine the number of reaches.
    Ensemble learning methods are used to detect possible false positive/negatives. Results are returned as a vector
    with binary representations of the trial type, number of reaches present, handedness of reach (per reach), and
    an outlier flag for each class set through a majority vote (8/10) of unique supervised classification algorithms.
    For details on the methods used in training our supervised classifiers, please see /Classifier_Training."""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Trial_Classify:
    def __init__(self, null_model_path, reach_model_path, hand_model_path, input_dataframe, n_components=100):
        """ Method to perform classification on single-trial data using n principal components."""
        self.model_null = joblib.load(null_model_path)
        self.model_num = joblib.load(reach_model_path)
        self.model_hand = joblib.load(hand_model_path)
        self.null_result, self.num_result, self.hand_result, self.outlier_flag_null, self.outlier_flag_num, \
            self.outlier_flag_hand, self.preprocessed_data, self.preprocessed_data_PCS = \
            [], [], [], [], [], [], [], []
        self.PCA = PCA(whiten=True, n_components=n_components)  # Use 100 components of PCA for classification
        self.data = input_dataframe
        self.preprocess_data_into_discrete_PC()  # discretize incoming trial data
        self.null_class_result()
        self.num_class_result()
        self.hand_class_result()

    def null_class_result(self):
        """ Function to perform classification using pre-trained networks for trial type."""
        self.null_result = self.model_null.predict(self.preprocessed_data_PCS)

    def num_class_result(self):
        """ Function to perform classification using pre-trained networks for number of reaches."""
        self.num_result = self.model_num.predict(self.preprocessed_data_PCS)

    def hand_class_result(self):
        """ Function to perform classification using pre-trained networks for handedness of a reach. """
        self.hand_result = self.model_hand.predict(self.preprocessed_data_PCS)

    def preprocess_data_into_discrete_PC(self):
        """ Function to discretize incoming high-dimensional time-series data (using IQR and 95th percentile.). """
        data = {}
        for col_name in self.data.columns:
            data[f'{col_name}_p25'] = []
            data[f'{col_name}_p50'] = []
            data[f'{col_name}_p75'] = []
            data[f'{col_name}_p95'] = []
            col = self.data[col_name]  # get column values
            col = col.dropna()  # drop nans
            col = col.to_frame()
            len_col = len(col)
            if len_col > 1:  # is time series
                # Get summary stats
                p25 = np.percentile(col, 25)  # axis=0
                p50 = np.percentile(col, 50)  # median
                p75 = np.percentile(col, 75)
                p95 = np.percentile(col, 95)
                # add values
                data[f'{col_name}_p25'].append(p25)
                data[f'{col_name}_p50'].append(p50)
                data[f'{col_name}_p75'].append(p75)
                data[f'{col_name}_p95'].append(p95)
            else:
                print("Not a Time SERIES")
        discrete_df = pd.DataFrame(data)
        discrete_df2 = StandardScaler().fit_transform(discrete_df)
        self.preprocessed_data = pd.DataFrame(discrete_df2, columns=discrete_df.columns)

    def take_PCS_preprocessed_data(self):
        """ Function to perform PCA on pre-processed discretized trial data. """
        self.preprocessed_data_PCS = self.PCA.fit_transform(self.preprocessed_data)
