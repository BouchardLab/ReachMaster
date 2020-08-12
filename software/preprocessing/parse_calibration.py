from trodes_data import calibration_data_parser as cdp
#To do: callable from CLI

data_dir = '/home/kallanved/Desktop'
trodes_name = 'calibration_BN_07212020'

cal_frame = cdp.get_calibration_frame(data_dir, trodes_name) 

cal_frame.to_csv(path_or_buf='/home/kallanved/Desktop/calibration.csv')
