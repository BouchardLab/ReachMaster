import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pdb


def preprocessing_df(df):
    list_of_names=['x_p','y_p','z_p','r_x','r_y','r_z', 'time', 'm_start', 'm_stop','SF','Date','r_start','r_stop']
    #shaking = xform_variables_to_numpy(df,list_of_names)
    shaking = dim_change(df)
    shaking=date_wipe(shaking)
    return shaking
    


def xform_variables_to_numpy(df, list_of_names):
    for i in list_of_names:
        for k, row_value in df[i].iteritems():
            x=str(df[i][k])
            x=x.strip('[,]')
            x=x.replace('...',"").replace('\n',"")
            x=x.replace('..',"")
            x=x.replace(',',"").replace('inRewardWin',"")
            x=x.replace('Name:',"").replace('_',"").replace('robmoving',"")
            x=x.replace('rob_moving',"").replace("Length","").replace(':',"").replace("dtype","").replace("int64","")
            x=x.replace('inRewardWin',"")
            x=str(x).split()
            df[i][k]=np.asarray(x, dtype=np.float32)
    return df


def date_wipe(df):
    index_total=[]
    for index, row_value in df['Date'].iteritems():
        if '_' in str(row_value):
            index_total.append((row_value)[-3:-1])
        else:
            index_total.append((row_value)[-2:])
    df['Date'] = np.asarray(index_total)
    return df


def dim_change(df):
    for k,x in enumerate(df['dim']):
        try:
            if "cone" in x:
                df['dim'][k] = 3
            elif "20mm" in x:
                df['dim'][k] = 0
            elif "10mm" in x:
                df['dim'][k] = 0
            else:
                df['dim'][k] = 0
        except:
            continue
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
    return_df = df.loc[df[idx_name] == name]
    if set_index:
        return_df.set_index(set_index)
    return return_df

def time_diff_dataframe(df):
    d=[]
    for i in range(len(shaking['m_stop'])):
        try:
            d.append( shaking['m_stop'][i]-shaking['m_start'][i])
        except:
            if len(shaking['m_stop'][i]) > len(shaking['m_start'][i]):
                sierra = shaking['m_stop'][i][:-1]
                d.append( sierra-shaking['m_start'][i])
            else:
                sierra = shaking['m_start'][i][:-1]
                d.append(shaking['m_stop'][i]-sierra)

    p_df = pd.DataFrame({'Rat': shaking['rat'], 'Date': shaking['Date'], 'time_vector': np.asarray(d), 'SF':shaking['SF']})
    list_of_names=['time_vector','SF']
    shaking_time = xform_variables_to_numpy(p_df,list_of_names)
    return shaking_time


def find_times(df):
    mean_time_vector_s = []
    mean_time_vector_f=[]
    for idx, row in p_df.iterrows():
    # match up S/F times
        s_f = np.asarray(row['SF']).astype(int)
    #pdb.set_trace()
        trial_time = row['time_vector']
    #print(trial_time)
    #lick_times=row['lick']
        x=np.array(trial_time)
        mask=np.full(len(trial_time),True,dtype=bool)
        try:
            mask[s_f]=False
            y=x[mask]
            z=x[~mask]
        except:
            y=x[mask]
            z=x[~mask]
    # 0 for correct
        mean_time_vector_s.append(np.mean(z))
        mean_time_vector_f.append(np.mean(y))
    return mean_time_vector_s, mean_time_vector_f