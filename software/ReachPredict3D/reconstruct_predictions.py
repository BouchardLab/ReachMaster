"""Module intended to reconstruct DLC inferred positional predictions into 3-D Euclidean Space using
DLT (Direct Linear Transformation) method.
"""
import os
import glob
import tqdm
import pickle
from software.preprocessing.Scripts.data_extraction import name_scrape, get_name
import pandas as pd
from software.preprocessing.video_data.DLC.Reconstruction import get_kinematic_data


def return_block_kinematic_df(f):
    """function to extract a single experimental session of positional predictions in 3-D using DLT
    Attributes
    --------------
    f: list containing file path and dlt_path used to generate PNS directory for kinematic data scrape

    Returns
    -----------
    kd : 3-D predictions for a given session, pandas dataframe

    """
    file_ = f[0]
    dlt_path = f[1]
    controller_path, config_path, exp_name, name, ix, trodes_name, video_path = name_scrape(file_)
    # dim, reward_dur, x_pos, y_pos, z_pos, x0, y0, z0, r, t1, t2 = get_config_data(str(config_path))
    date = get_name(file_)
    print(exp_name + ' is being added..')
    kd = get_kinematic_data(video_path, dlt_path, 'resnet', name, date, ix, 0)
    return kd


def save_kinematics(unpickled_list):
    """function intended to save list of dataframes from a given rat as a single dataframe
    Attributes
    ------------
    unpickled_list: list containing all N dataframes from a given rat's session

    Returns
    -------------
    df1: dataframe containing all N dataframes for a given rat

    """
    encountered_df = False
    for i in range(len(unpickled_list)):
        rat_df1 = unpickled_list[i]
        if ((rat_df1 is not 0) and (type(rat_df1) is not list)):
            if (not encountered_df):
                encountered_df = True
            # create a new dataframe (copy of original), then removes levels corresponding to (rat, date, session, dim)
            df1 = rat_df1.droplevel([0, 1, 2, 3], axis=1)
            # inserts columns and data for (rat, date, session, dim)
            pos_arr = [0, 1, 1, 3]  # order to insert columns (rat, date, session, dim)
            for i in range(4):
                col_name = rat_df1.columns.names[i]
                val = rat_df1.columns.levels[i][0]
                df1.insert(pos_arr[i], col_name, val)
            else:
                df2 = rat_df1.droplevel([0, 1, 2, 3], axis=1)
                pos_arr = [0, 1, 1, 3]  # order to insert columns (rat, date, session, dim)
                for i in range(4):
                    col_name = rat_df1.columns.names[i]
                    val = rat_df1.columns.levels[i][0]
                    df2.insert(pos_arr[i], col_name, val)
                df1 = pd.concat([df1, df2], axis=0, sort=False)  # concat new df to existing df
    df1 = df1.set_index(['rat', 'date', 'session', 'dim'])
    return df1


class Reconstruct3D:
    """Class to iterate over PNS directory sessions and return/save 3-D positional predictions from individual session files. At the
    end of iterating over a PNS directory, the entire dataframe is saved.
    """
    def __init__(self):
        self.pickling = False
        self.rat_name = '16'
        self.cns_path = '/clusterfs/bebb/users/bnelson/CNS/' + self.rat_name
        self.cns_pattern = self.cns_path + '/**/*.rec'
        self.pns_path = '/clusterfs/bebb/users/bnelson/PNS_data/'
        self.cns = '/clusterfs/bebb/users/bnelson/CNS'
        self.dlt_path = ''
        self.save_path='new_file.csv'
        self.cl = glob.glob(self.cns_pattern, recursive=True)
        print(str(len(self.cl)) + ' of experimental blocks for this rat.')
        d = []
        for file in tqdm(glob.glob(self.cns_pattern, recursive=True)):
            dc = return_block_kinematic_df([file, self.dlt_path])
            d.append(dc)
        print('Saving DataFrame')
        os.chdir(self.cns)
        if self.pickling:
            with open(self.save_path, 'wb') as output:
                pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)
        self.new_df = save_kinematics(d)
        self.new_df.to_hdf(self.new_df, self.save_path) # saves total list


