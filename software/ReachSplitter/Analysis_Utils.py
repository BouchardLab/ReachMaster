import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb


def preprocessing_df(df_in):
    robot_df = date_wipe(df_in)
    return robot_df



def date_wipe(df):
    d=[]
    for index, row_value in df['Date'].iteritems():
        if "_" in row_value:
            d.append((row_value)[4:6])
        else:
            d.append((row_value)[5:7])
    df['Date']=d
    return df


def return_plotting_arrays(df):
    trial_percent_total=[]
    index_total=[]
    rat_total=[]
    dim_total=[]
    datelist = ['17','18','19','20','25','26','27','28']
    for index, row in df.iterrows():
        dim_total.append(row['dim'])
        calc_row = row['SF']
        #pdb.set_trace()
        rat = row['rat']
        amt_trials = len(row['m_start'])
        s_trials = len(calc_row)
        try:
            trial_percent = s_trials / amt_trials
        except ZeroDivisionError:
            trial_percent = 0
        if trial_percent > 1.0:
            continue
        else:
            trial_percent_total.append(trial_percent)
            if '_' in row['Date']:
                index_total.append((row['Date'])[-3:-1])
            else:
                index_total.append((row['Date'])[-2:])
            rat_total.append(rat)
    df = pd.DataFrame({'Total':np.asarray(trial_percent_total), 'Date': index_total, 'Rat': rat_total,'dim':dim_total})
    return df


def query_df(df,idx_name, query_subject, set_index=False):
    return_df = df.loc[df[idx_name] == idx_name]
    if set_index:
        return_df.set_index(set_index)
    return return_df




