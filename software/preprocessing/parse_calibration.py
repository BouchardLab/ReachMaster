from trodes_data import calibration_data_parser as cdp
#To do: callable from CLI

data_dir = '/home/cns/Desktop/Trodes/linux/experiments'
trodes_name = 'calibration20200318_124649'

cal_frame = cdp.get_calibration_frame(data_dir, trodes_name) 

cal_frame.to_csv(path_or_buf='/home/cns/Desktop/Trodes/linux/calibrationDF.csv')
