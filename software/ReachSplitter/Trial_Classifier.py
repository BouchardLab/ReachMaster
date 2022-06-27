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
import pdb


class Trial_Classify:
    def __init__(self, input_dataframe, n_components=100):
        """ Method to perform classification on single-trial data using n principal components."""
        #path_to_rm = 'Users/bassp/PycharmProjects/ReachMaster/software/ReachSplitter/models/'
        path_to_rm = ''
        self.null_pca_path = path_to_rm + 'null_PCA.sav'
        self.num_pca_path = path_to_rm + 'num_reach_PCA.sav'
        self.hand_pca_path = path_to_rm + 'hand_PCA.sav'
        self.null_model_path = 'null_model.joblib'
        self.null_model_path = path_to_rm + 'null_model.joblib'
        self.reach_model_path = path_to_rm + 'numReach_model.joblib'
        self.hand_model_path = path_to_rm + 'whichHand_model.joblib'
        try:
            self.model_null = joblib.load(self.null_model_path)
        except:
            pdb.set_trace()
        self.model_num = joblib.load(self.reach_model_path)
        self.model_hand = joblib.load(self.hand_model_path)
        self.PCA_null = joblib.load(self.null_pca_path)
        self.PCA_num = joblib.load(self.num_pca_path)
        self.PCA_hand = joblib.load(self.hand_pca_path)
        self.hand_PCS, self.null_PCS, self.num_PCS = 0, 0, 0
        self.null_result, self.num_result, self.hand_result, self.outlier_flag_null, self.outlier_flag_num, \
        self.outlier_flag_hand, self.preprocessed_data, self.preprocessed_data_PCS = \
            [], [], [], [], [], [], [], []
        self.num_reaches, self.three_or_more_reaches = None, False
        self.PCA = PCA(whiten=True, n_components=n_components)  # Use 100 components of PCA for classification
        self.timeseries_data = input_dataframe
        self.preprocess_data_into_discrete_PC()  # discretize incoming trial data
        self.null_class_result()
        if self.null_result == 0:
            self.num_reaches = 0
        else:
            self.num_class_result()
            if self.num_result == 1:
                self.hand_class_result()
            else:
                self.hand_result = 2  # no classification performed

    def null_class_result(self):
        """ Function to perform classification using pre-trained networks for trial type."""
        self.null_result = self.model_null.predict(self.null_PCS)

    def num_class_result(self):
        """ Function to perform classification using pre-trained networks for number of reaches."""
        self.num_result = self.model_num.predict(self.num_PCS)
        if self.num_result == 0:
            self.num_reaches = 1
        else:  # Predict 2 or more
            # self.double_or_more = self.model_double_num.predict(self.preprocessed_data_PCS) # Need to set up network for 2 vs more
            # if self.double_or_more == 0:
            self.num_reaches = 2
            # else:
            #    self.three_or_more_reaches = True

    def hand_class_result(self):
        """ Function to perform classification using pre-trained networks for handedness of a reach. """
        self.hand_result = self.model_hand.predict(self.hand_PCS)

    def preprocess_data_into_discrete_PC(self):
        """ Function to discretize incoming high-dimensional time-series data (using IQR and 95th percentile.). """
        data = {}
        for col_name in self.timeseries_data.columns:
            col = self.timeseries_data[col_name]  # get column values
            col = col.dropna()  # drop nans
            col = col.to_frame()
            col = col.values
            if col.shape[0] > 1:  # is time series
                # Get summary stats
                data[f'{col_name}_p25'] = []
                data[f'{col_name}_p50'] = []
                data[f'{col_name}_p75'] = []
                data[f'{col_name}_p95'] = []
                try:
                    p25 = np.percentile(col, 25)  # axis=0
                    p50 = np.percentile(col, 50)  # median
                    p75 = np.percentile(col, 75)
                    p95 = np.percentile(col, 95)
                    # add values
                    data[f'{col_name}_p25'].append(p25)
                    data[f'{col_name}_p50'].append(p50)
                    data[f'{col_name}_p75'].append(p75)
                    data[f'{col_name}_p95'].append(p95)
                except:  # Bad data type
                    print(col_name + 'bad')
            else:
                pass
        pdb.set_trace()
        discrete_df = pd.DataFrame(data)
        discrete_df2 = StandardScaler().fit_transform(discrete_df)
        self.preprocessed_data = pd.DataFrame(discrete_df2, columns=discrete_df.columns)
        self.take_PCS_preprocessed_data()
        pdb.set_trace()

    def take_PCS_preprocessed_data(self):
        """ Function to perform PCA on pre-processed discretized trial data. """
        self.hand_PCS = self.PCA_hand.transform(self.preprocessed_data)
        self.null_PCS = self.PCA_null.transform(self.preprocessed_data)
        self.num_PCS = self.PCA_num.transform(self.preprocessed_data)
