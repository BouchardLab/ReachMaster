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
    values = df["response_sensor"].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    df["response_sensor"] = integer_encoded
    return df


def select_colnames(df, colname_list):
    selected_col_names = []
    selected_col_names.extend(["Rat", "Session", "Trial", "Date"])
    for col in df.columns:
        for r in colname_list:
            if r in col:
                selected_col_names.append(col)
    return_df = df.drop(df.columns.difference(selected_col_names), axis=1)
    return return_df


def get_sensor_data(i_df):
    sensor_list = ['handle_moving_sensor', 'lick_beam', 'reward_zone', 'response_sensor', 'x_rob', 'y_rob', 'z_rob']
    sensor_df = i_df.drop(i_df.columns.differences(sensor_list), axis=1)
    return sensor_df

def filterdf(df, trial, rat, session, date):
    rr = df.loc[df['Date'] == date]
    rr1 = rr.loc[rr['Session'] == session]
    new_df = rr1.loc[rr1['Rat'] == rat]
    dff = new_df.loc[new_df['Trial'] == trial]
    if dff.shape[0] == 0:
        print(f"NO matching Trial was found for {trial, rat, session, date}")
    return dff


def get_sep_dataframe(df, selected_col_names):
    filteredData = df.drop(df.columns.difference(selected_col_names), axis=1)
    return filteredData


def match_input_data_with_labels_and_make_standardized_discrete_dataframe(df, labels):
    trials = labels['Trial Num'].values
    rats = labels['Rat'].values
    sessions = labels['Session'].values
    dates = labels['Date'].values
    all_timeSeries = []
    # for each trial
    for i in np.arange(labels.shape[0]):
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
        # find corresponding time series
        trial_df = filterdf(df, trial, rat, session, date)
        if trial_df.shape[0] != 0:  # if no error in finding df
            # drop trial ID columns
            trial_df = trial_df.drop(["Trial", "Date", "Session", "Rat"], axis=1)
            all_timeSeries.append(trial_df)
    data = {}
    trial_df = all_timeSeries[0]
    # for each column
    for col_name in trial_df.columns:
        data[f'{col_name}_p25'] = []
        data[f'{col_name}_p50'] = []
        data[f'{col_name}_p75'] = []
        data[f'{col_name}_p95'] = []
    pd.DataFrame(data)
    for i in np.arange(len(all_timeSeries)):
        # add feature data
        trial_df = all_timeSeries[i]
        # for each column
        for col_name in trial_df.columns:
            col = trial_df[col_name]  # get column values
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
    discrete_df = pd.DataFrame(discrete_df2, columns=discrete_df.columns)
    return discrete_df


def drop_labels_from_data(labels, rat, date, session):
    mod_labels = labels.loc[(labels['Rat'] != rat) | (labels["Session"] != session) | (labels['Date'] != date)]
    return mod_labels


def concat_labels_to_discrete_dataframe(new_labels, discrete_df):
    new_labels = new_labels.drop(
        new_labels.columns.difference(["Rat", "Date", "Session", "Trial", "Trial Type", "Num Reaches", "Which Hand"]),
        axis=1)
    new_labels.reset_index(inplace=True)
    new_df = pd.concat([discrete_df, new_labels], axis=1)
    return new_df


def get_matched_labels_and_data(dataframe_address, labels_address, var_col_names, drop_session_list=None):
    df = load_preprocessed_dataframe(dataframe_address)
    labels_df = load_ground_truth_labels(labels_address)
    pp_df = preprocess_df(df)
    pp_df_kinematics = select_colnames(pp_df, var_col_names)
    pp_df_sensor_data = get_sensor_data(pp_df)
    if drop_session_list:
        for item in drop_session_list:
            labels_df = drop_labels_from_data(labels_df, item[0], item[1], item[2])
    matched_data = match_input_data_with_labels_and_make_standardized_discrete_dataframe(pp_df_kinematics, labels_df)
    concat_data = concat_labels_to_discrete_dataframe(labels_df, matched_data)
    return concat_data, matched_data, pp_df_sensor_data


class Visualize:

    def __init__(self, dataframe_address, labels_address, col_names, drop_list):
        self.matched_data, self.discrete_data, self.sensor_data = get_matched_labels_and_data(dataframe_address,
                                                                                              labels_address,
                                                                                               col_names, drop_list)
        self.PCA = PCA()
        self.PCA_kinematic_data = self.PCA.fit_transform(self.discrete_data)
        self.PCA_sensor_data = self.PCA.fit_transform(self.sensor_data)

    def visualize_PCA_cumulative_sum(self):
        exp_var_cumul = np.cumsum(self.PCA_data.explained_variance_ratio_)
        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        fig.show()
