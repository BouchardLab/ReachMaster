import pdb

import pandas as pd
import pickle
from software.ReachSplitter.ReachLoader import *

""" This method is intended to load pre-processed, pandas dataframes and numpy vector stacks for the ReachAnalysis library.
    User inputs list of strings of: addresses of experimental and 3-D variable dataframes
    Calling this method may increase memory load.
"""


class DataLoader:
    def __init__(self, kinematic_addresses, experimental_addresses):
        """ Initializes dataloader method using selected addresses of kinematic (3-D kinematics) and experimental data
            from reconstructed data objects obtained through the ReachPredict3D module.
            :param list kinematic_addresses: list of strings (file addresses) for each desired rat to perform analysis on
            :param list experimental_addresses: list of strings (file addresses) for each desired rat to perform analysis on
        """
        self.kin_16, self.kin_15, self.kin_14, self.kin_13, self.kin_12, self.kin_11, self.kin_10, \
        self.kin_9 = [], [], [], [], [], [], [], []
        self.exp_16, self.exp_15, self.exp_14, self.exp_13, self.exp_12, self.exp_11, self.exp_10, \
        self.exp_9 = [], [], [], [], [], [], [], []
        self.set_block_file_addresses(kinematic_addresses, experimental_addresses)
        self.block_video_file = ''  # Necessary to initiate ReachLoader method
        return

    def set_block_file_addresses(self, k_a, exp_a):
        """Function to select kinematic and experimental datafile addresses
            :param list k_a: list containing string addresses of kinematic data
            :param list exp_a: list containing string addresses of experimental data
        """
        for i in k_a:
            if 'RM9' in i:
                self.kin_9 = i
            if 'RM10' in i:
                self.kin_10 = i
            if 'RM11' in i:
                self.kin_11 = i
            if 'RM12' in i:
                self.kin_12 = i
            if 'RM13' in i:
                self.kin_13 = i
            if 'RM14' in i:
                self.kin_14 = i
            if 'RM15' in i:
                self.kin_15 = i
            if 'RM16' in i:
                self.kin_16 = i

        for j in exp_a:
            if 'RM9' in j:
                self.exp_9 = j
            if 'RM10' in j:
                self.exp_10 = j
            if 'RM11' in j:
                self.exp_11 = j
            if 'RM12' in j:
                self.exp_12 = j
            if 'RM13' in j:
                self.exp_13 = j
            if 'RM14' in j:
                self.exp_14 = j
            if 'RM15' in j:
                self.exp_15 = j
            if 'RM16' in j:
                self.exp_16 = j
        return

    def select_blocks_from_list(self, rats, sessions, dates, save_dataframe_address=None):
        """ Creates dataframe and numpy stack for a given set of experimental blocks. Blocks must be defined through
            three parameters given as a list: the rat, the date, and session. An example would be 16, '25' 'S1'.
            :param list rats: rats, defined as list of ints (16, 15)
            :param list sessions: experimental sessions, defined as list of strings
            :param list dates: dates experiments performed on, truncated to the day as string ('17', '25' etc)
            :param str save_dataframe_address: optional address string for saving the created dataframe
            :return: pandas multi-index dataframe object
        """
        for ix, date in enumerate(dates):
            # set kin, exp_data file based on rat
            sesh = sessions[ix]
            ratties = rats[ix]
            print('Extracting ' + str(ratties) + str(date) + str(sesh))
            if ratties == 16:
                assert len(self.kin_16) > 0, print('Empty list')
                assert len(self.exp_16) > 0, print('Empty list')
                kin_file = self.kin_16
                exp_datafile = self.exp_16
            if ratties == 15:
                assert len(self.kin_15) > 0, print('Empty list')
                assert len(self.exp_15) > 0, print('Empty list')
                kin_file = self.kin_15
                exp_datafile = self.exp_15
            if ratties == 14:
                assert len(self.kin_14) > 0, print('Empty list')
                assert len(self.exp_14) > 0, print('Empty list')
                kin_file = self.kin_14
                exp_datafile = self.exp_14
            if ratties == 13:
                assert len(self.kin_13) > 0, print('Empty list')
                assert len(self.exp_13) > 0, print('Empty list')
                kin_file = self.kin_13
                exp_datafile = self.exp_13
            if ratties == 12:
                assert len(self.kin_12) > 0, print('Empty list')
                assert len(self.exp_12) > 0, print('Empty list')
                kin_file = self.kin_12
                exp_datafile = self.exp_12
            if ratties == 11:
                assert len(self.kin_11) > 0, print('Empty list')
                assert len(self.exp_11) > 0, print('Empty list')
                kin_file = self.kin_11
                exp_datafile = self.exp_11
            if ratties == 10:
                assert len(self.kin_10) > 0, print('Empty list')
                assert len(self.exp_10) > 0, print('Empty list')
                kin_file = self.kin_10
                exp_datafile = self.exp_10
            if ratties == 9:
                assert len(self.kin_9) > 0, print('Empty list')
                assert len(self.exp_9) > 0, print('Empty list')
                kin_file = self.kin_9
                exp_datafile = self.exp_9
            dd = self.get_data_from_block(date, sesh, exp_datafile, kin_file, ratties)
            if ix == 0:
                sd = dd
            else:
                sd = pd.concat([sd, dd])
        pdb.set_trace()
        if save_dataframe_address:
            sd.to_pickle(save_dataframe_address)
        return sd

    def get_data_from_block(self, date, sesh, experiment_data, kin_data, ratt):
        try:
            R = ReachViz(date, sesh, experiment_data, self.block_video_file, kin_data, ratt)
        except:
            pdb.set_trace()
        #dd = R.get_reach_dataframe_from_block()
        dd = R.get_preprocessed_trial_blocks()
        return dd


dates = ['17', '18', '17', '20', '19', '25', '17', '20', '18', '19', '18', '17', '19']
rats = [16, 16, 16, 16, 16, 15, 15, 14, 14, 12, 11, 10, 9]
sessions = ['S1', 'S1', 'S2', 'S3', 'S3', 'S3', 'S4', 'S1', 'S2', 'S1', 'S4', 'S2', 'S3']
root = "C:\\Users\\bassp\\Desktop"
os.chdir(root)
block_video_file = 'Classification Project\\2019-09-20-S1-RM14_cam2DLC_FinalColors.mp4'
save_df_address = 'Total_Reaches.pkl'

kinematics_addresses = ['DataFrames\\3D_positions_RM9.pkl',
                        'DataFrames\\3D_positions_RM10.pkl',
                        'DataFrames\\3D_positions_RM11.pkl',
                        'DataFrames\\3D_positions_RM12.pkl',
                        'DataFrames\\3D_positions_RM13.pkl',
                        'DataFrames\\3D_positions_RM14.pkl',
                        'DataFrames\\3D_positions_RM15.pkl',
                        'DataFrames\\3D_positions_RM16.pkl']

exp_addresses = ['DataFrames\\RM9_expdf.pickle',
                 'DataFrames\\RM10_expdf.pickle',
                 'DataFrames\\RM11_expdf.pickle',
                 'DataFrames\\RM12_expdf.pickle',
                 'DataFrames\\RM13_expdf.pickle',
                 'DataFrames\\RM14_expdf.pickle',
                 'DataFrames\\RM15_expdf.pickle',
                 'DataFrames\\RM16_expdf.pickle']

DL = DataLoader(kinematics_addresses, exp_addresses)
DL.select_blocks_from_list(rats, sessions, dates, save_dataframe_address=save_df_address)
