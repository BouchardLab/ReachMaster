from software.ReachSplitter.ReachLoader import ReachViz

block_video_file = '/Users/bassp/OneDrive/Desktop/Classification Project/2019-09-20-S1-RM14_cam2DLC_FinalColors.mp4'
kin_file = '/Users/bassp/OneDrive/Desktop/DataFrames/3D_positions_RM14.pkl'
exp_datafile = '/Users/bassp/OneDrive/Desktop/DataFrames/RM14_expdf.pickle'
date = '20'
session = 'S1'
rat = 'RM14'
R = ReachViz(date, session, exp_datafile, block_video_file, kin_file, rat)
reach_array = R.vid_splitter_and_grapher(plot=True, plot_reach=True)
