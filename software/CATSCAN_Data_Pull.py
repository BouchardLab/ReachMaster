from software.ReachSplitter.ReachLoader import ReachViz
import pandas as pd
import os

root = "C:\\Users\\bassp\\Desktop"
os.chdir(root)
block_video_file = 'Classification Project\\2019-09-20-S1-RM14_cam2DLC_FinalColors.mp4'
save_df_address = 'Full_Pilot_Reaches.pkl'

kinematics_addresses = [#'DataFrames\\3D_positions_RM9.pkl',
                        #'DataFrames\\3D_positions_RM10.pkl',
                        #'DataFrames\\3D_positions_RM11.pkl',
                        'DataFrames\\3D_positions_RM12.pkl',
                        'DataFrames\\3D_positions_RM13.pkl',
                        'DataFrames\\3D_positions_RM14.pkl',
                        'DataFrames\\3D_positions_RM15.pkl',
                        'DataFrames\\3D_positions_RM16.pkl']

exp_addresses = [#'DataFrames\\RM9_expdf.pickle',
                 #'DataFrames\\RM10_expdf.pickle',
                 #'DataFrames\\RM11_expdf.pickle',
                 'DataFrames\\RM12_expdf.pickle',
                 'DataFrames\\RM13_expdf.pickle',
                 'DataFrames\\RM14_expdf.pickle',
                 'DataFrames\\RM15_expdf.pickle',
                 'DataFrames\\RM16_expdf.pickle']
# For each rat, we have a kinematics (positions) file. Additionally, we have a sensor data (experiment datafile)

rats = ['RM9', 'RM10', 'RM11', 'RM12', 'RM13', 'RM14', 'RM15', 'RM16']


def loop_over_rat_and_extract_reaches(prediction_dataframe, e_dataframe, dummy_video_path, rat):
    # Get rat, date, session for each block we need to process.
    k_dataframe = pd.read_pickle(prediction_dataframe)

    for ilx, kk in enumerate(k_dataframe):
        session = kk.columns[2][1]
        date = kk.columns[2][0][2:4]
        print(session, date)
        R = ReachViz(date, session, e_dataframe, dummy_video_path, prediction_dataframe, rat)
        # Using ReachViz object, pull dataframe
        reaching_df = R.get_reach_dataframe_from_block()
        if ilx == 0:
            final_df = reaching_df
        final_df = pd.concat([final_df, reaching_df])
    return final_df


def extract_reaching_data_from_unprocessed_data(block_video_file_id, kin_file_base_array, exp_datafile_base_array):
    for num, single_file in enumerate(kin_file_base_array):
        exp_file = exp_datafile_base_array[num]
        ratt = single_file[-8:-4]
        print(ratt)
        if '_' in ratt:
            ratt = ratt[1:]
            print(ratt)
        complete_rat_df = loop_over_rat_and_extract_reaches(single_file, exp_file, block_video_file_id, ratt)
        if num == 0:
            rat_final_df = complete_rat_df
        else:
            rat_final_df = pd.concat([complete_rat_df, rat_final_df])
    return rat_final_df

save_path = '/Users/bassp/OneDrive/Desktop/Classification Project/reach_data/Pilot_Data.h5'
final_df = extract_reaching_data_from_unprocessed_data(block_video_file, kinematics_addresses, exp_addresses)
final_df.to_hdf(save_path)