import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle5 as pickle
from sklearn.decomposition import PCA
import plotly.express as px
import pdb


def load_preprocessed_dataframe(dataframe_address):
    with open(dataframe_address, "rb") as fh:
        data = pickle.load(fh)
    data = data.reset_index()
    return data


def load_ground_truth_labels(labels_address):
    labels = pd.read_csv(labels_address)
    labels = labels.drop(["Unnamed: 0", "Start Frame", "Stop Frame", "Num Frames"], axis=1)
    return labels


def preprocess_df(df):
    df["Date"] = df["Date"].astype(int)
    for col_name in df.columns:
        if df.dtypes[col_name] == object:
            if (col_name != "Session") & (col_name != "endpoint_error") & (col_name != "response_sensor"):
                df = df.drop(col_name, axis=1)
    # function to binarize response sensor values, may be depreciated soon
    values = df["response_sensor"].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    df["response_sensor"] = integer_encoded
    return df


def filter_df(df, trial, rat, session, date):
    rr = df.loc[df['Date'] == date]
    rr1 = rr.loc[rr['Session'] == session]
    new_df = rr1.loc[rr1['Rat'] == rat]
    dff = new_df.loc[new_df['Trial'] == trial]
    return dff


def drop_labels_from_data(il_labels, rat, session, date):
    mod_labels = il_labels.loc[(il_labels['Rat'] != rat) | (il_labels["Session"] != session)
                               | (il_labels['Date'] != date)]
    return mod_labels


class Visualize:

    def __init__(self, dataframe_address, labels_address, drop_list):
        self.matched_kinematics_list, self.labels_list = [], []
        # Define variables we want to extract time series data from
        self.right_forearm_x, self.right_forearm_y, self.right_forearm_z, self.left_forearm_s, self.right_forearm_s, \
            self.left_forearm_x, self.left_forearm_z, self.left_wrist_z, self.right_wrist_z, \
            self.left_forearm_y, self.left_wrist_x, self.right_wrist_x, self.right_index_base_x, \
            self.right_index_tip_x, self.right_index_tip_z, self.left_index_tip_z, self.left_index_base_z, \
            self.right_index_base_z, self.left_index_base_x, \
            self.left_index_tip_x = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.left_palm_x, self.right_palm_x, self.left_palm_y, self.right_palm_y, self.right_palm_s, \
            self.left_palm_s, self.time_vector, self.handle_x, self.handle_y, self.right_palm_z, self.left_palm_z, \
            self.handle_speed, self.handle_z = [], [], [], [], [], [], [], [], [], [], [], [], []
        self.left_index_base_s, self.right_index_base_s, self.left_index_tip_s, self.right_index_tip_s, \
            self.right_wrist_s, self.left_wrist_s, self.left_wrist_y, self.right_wrist_y = [], [], [], [], [], [], [], []
        self.get_matched_labels_and_data(dataframe_address, labels_address, drop_list)
        self.PCA = PCA()

    def get_matched_labels_and_data(self, dataframe_address, labels_address, drop_session_list=None):
        """ Main function to generate list of time series variables for class. Input dataframe address, labels address,
            load list of dropped blocks [Rat, Session, Date]. """
        # Load data
        df = load_preprocessed_dataframe(dataframe_address)
        labels_df = load_ground_truth_labels(labels_address)
        # Preprocess (drop information, reformat dataframe)
        pp_df = preprocess_df(df)
        # Drop non-functioning experimental blocks from data (if any)
        if drop_session_list:  # If dropping any blocks
            for idx, item in enumerate(drop_session_list):
                if idx == 0:
                    labels_df_f = drop_labels_from_data(labels_df, item[0], item[1], item[2])  # Drop data
                else:
                    labels_df_f = drop_labels_from_data(labels_df_f, item[0], item[1], item[2])  # Drop data
        # Iterate over label dataframe, extract ordered labels and time series data from each entry
        self.match_input_data_with_labels_and_make_standardized_timeseries_dataframe_list(pp_df, labels_df_f)
        return

    def match_input_data_with_labels_and_make_standardized_timeseries_dataframe_list(self, df, i_labels):
        """ Function to iterate over appropriately filtered class labels within a given block, append time series
            data to named lists, and to append label information to ordered list."""
        trials = i_labels['Trial Num'].values
        rats = i_labels['Rat'].values
        sessions = i_labels['Session'].values
        dates = i_labels['Date'].values
        for i in np.arange(i_labels.shape[0]):
            trial = trials[i]
            rat = rats[i]
            # convert to int
            if rat == "RM16":
                rat = 16
            if rat == "RM15":
                rat = 15
            if rat == "RM14":
                rat = 14
            if rat == "RM13":
                rat = 13
            if rat == "RM12":
                rat = 12
            if rat == "RM11":
                rat = 11
            if rat == "RM10":
                rat = 10
            if rat == "RM9":
                rat = 9
            session = sessions[i]
            date = dates[i]
            trial_df = filter_df(df, trial, rat, session, date)  # find corresponding time series
            if trial_df.shape[0] != 0:  # if no error in finding df
                trial_df = trial_df.drop(["Trial", "Date", "Session", "Rat"], axis=1)  # drop trial ID columns
                self.add_kinematic_feature_list(trial_df)  # Append necessary time series to named list
                # Now we want to extract essential labels for visualization and quantification.
                new_labels = i_labels.drop(
                    i_labels.columns.difference(
                        ["Trial Type", "Num Reaches", "Which Hand", "Tug", "Hand Switch"]), axis=1).to_numpy()[i, :]
                self.labels_list.append(new_labels)
        return

    def add_kinematic_feature_list(self, new_df):
        """ Function to append kinematic variables to appropriate named lists. Used in iterative manner."""
        self.left_forearm_s.append(new_df['left_forearm_s'].to_numpy())
        self.right_forearm_s.append(new_df['right_forearm_s'].to_numpy())
        self.right_forearm_y.append(new_df['right_forearm_py'].to_numpy())
        self.left_forearm_y.append(new_df['left_forearm_py'].to_numpy())
        self.right_forearm_x.append(new_df['right_forearm_px'].to_numpy())
        self.left_forearm_x.append(new_df['left_forearm_px'].to_numpy())
        self.right_forearm_z.append(new_df['right_forearm_pz'].to_numpy())
        self.left_forearm_z.append(new_df['left_forearm_pz'].to_numpy())
        self.left_wrist_z.append(new_df['left_wrist_pz'].to_numpy())
        self.right_wrist_z.append(new_df['right_wrist_pz'].to_numpy())
        self.left_wrist_y.append(new_df['left_wrist_py'].to_numpy())
        self.right_wrist_y.append(new_df['right_wrist_py'].to_numpy())
        self.left_wrist_x.append(new_df['left_wrist_px'].to_numpy())
        self.right_wrist_x.append(new_df['right_wrist_px'].to_numpy())
        self.right_index_base_z.append(new_df['right_index_base_pz'].to_numpy())
        self.right_index_tip_z.append(new_df['right_index_tip_pz'].to_numpy())
        self.right_index_base_x.append(new_df['right_index_base_px'].to_numpy())
        self.right_index_tip_x.append(new_df['right_index_tip_px'].to_numpy())
        self.left_index_base_s.append(new_df['left_index_base_s'].to_numpy())
        self.right_index_base_s.append(new_df['right_index_base_s'].to_numpy())
        self.left_index_tip_s.append(new_df['left_index_tip_s'].to_numpy())
        self.right_index_tip_s.append(new_df['right_index_tip_s'].to_numpy())
        self.left_index_base_x.append(new_df['left_index_base_px'].to_numpy())
        self.left_index_tip_x.append(new_df['left_index_tip_px'].to_numpy())
        self.left_index_base_z.append(new_df['left_index_base_pz'].to_numpy())
        self.left_index_tip_z.append(new_df['left_index_tip_pz'].to_numpy())
        self.handle_x.append(new_df['handle_px'].to_numpy())
        self.handle_y.append(new_df['handle_py'].to_numpy())
        self.handle_z.append(new_df['handle_pz'].to_numpy())
        self.handle_speed.append(new_df['handle_s'].to_numpy())
        self.right_palm_z.append(new_df['right_palm_pz'].to_numpy())
        self.left_palm_z.append(new_df['left_palm_pz'].to_numpy())
        self.right_palm_x.append(new_df['right_palm_px'].to_numpy())
        self.left_palm_x.append(new_df['left_palm_px'].to_numpy())
        self.left_palm_y.append(new_df['left_palm_py'].to_numpy())
        self.right_palm_y.append(new_df['right_palm_py'].to_numpy())
        self.right_palm_s.append(new_df['right_palm_s'].to_numpy())
        self.left_palm_s.append(new_df['left_palm_s'].to_numpy())
        self.left_wrist_s.append(new_df['left_wrist_s'].to_numpy())
        self.right_wrist_s.append(new_df['right_wrist_s'].to_numpy())
        self.time_vector.append(new_df['time_vector'].to_numpy())
