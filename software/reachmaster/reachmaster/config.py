import os 
import numpy as np
import json
import datetime

def default_cfg():
    cfg = {
             'ReachMaster':       {
                                   'data_dir':os.getcwd(),
                                   'cfg_file':'Default',
                                   'exp_control_path':'/dev/ttyACM0',
                                   'rob_control_path':'/dev/ttyACM1',
                                   'serial_baud':2000000,
                                   'control_timeout':5
                                   },
             'CameraSettings':    {
                                   'num_cams':3,
                                   'imgdataformat':'XI_RAW8',
                                   'fps':200,
                                   'exposure':2000,
                                   'gain':15.0,
                                   'sensor_feature_value':1,
                                   'gpi_selector':'XI_GPI_PORT1',
                                   'gpi_mode':'XI_GPI_TRIGGER',
                                   'trigger_source':'XI_TRG_EDGE_RISING',
                                   'gpo_selector':'XI_GPO_PORT1',
                                   'gpo_mode':'XI_GPO_EXPOSURE_ACTIVE',
                                   'img_width':688,
                                   'img_height':688,
                                   'offset_x':304,
                                   'offset_y':168,
                                   'downsampling':'XI_DWN_1x1',
                                   'saved_pois':[],
                                   'poi_threshold':15                                  
                                   },
             'ExperimentSettings':{
                                   'baseline_dur':5.0,
                                   'buffer_dur':0.5,
                                   'reach_timeout':4000,
                                   'flush_dur':10000,
                                   'solenoid_open_dur':75,
                                   'solenoid_bounce_dur':500,
                                   'reward_win_dur':3000,
                                   'max_rewards':3,
                                   'lights_off_dur':3000,
                                   'lights_on_dur':5000,
                                   'reach_delay':100
                                   },
             'RobotSettings':     {
                                   'calibration_file':'None',
                                   'dis':'None',
                                   'pos':'None',
                                   'xpush_dur':'None',
                                   'xpull_dur':'None',
                                   'ypush_dur':'None',
                                   'ypull_dur':'None',
                                   'zpush_dur':'None',
                                   'zpull_dur':'None',
                                   'command_type':'None',
                                   'command_set':'None',
                                   'command_file':'None',
                                   'xcommand_pos':'None',
                                   'ycommand_pos':'None',
                                   'zcommand_pos':'None',
                                   'rcommand_pos':'None',
                                   'thetay_command_pos':'None',
                                   'thetaz_command_pos':'None',
                                   'r_low':10,
                                   'r_high':40,
                                   'theta_mag':np.pi/3,
                                   'Ly':64,
                                   'Lz':47,
                                   'Axx':168,
                                   'Ayy':100,
                                   'Azz':117,
                                   'x0':1024,
                                   'y0':608,
                                   'z0':531,
                                   'alpha':0.1,
                                   'tol':np.round(1023.0/50.0/3.0,decimals=1),
                                   'period':125.0*1000.0,
                                   'off_dur':1000,
                                   'numTol':5,
                                   'xpush_wt':1.0,
                                   'xpull_wt':1.0,
                                   'ypush_wt':1.0,
                                   'ypull_wt':1.0,
                                   'zpush_wt':1.0,
                                   'zpull_wt':1.0,
                                   'RZx':1000,
                                   'RZy_low':558,
                                   'RZy_high':658,
                                   'RZz_low':481,
                                   'RZz_high':581
                                   },
              'Protocol':'TRIALS'
             }

    cfg['CameraSettings']['output_params'] = {
                                                    "-vcodec":"libx264", 
                                                    "-crf": 28,
                                                    "-preset":"ultrafast", 
                                                    "-tune":"zerolatency",
                                                    "-output_dimensions": (
                                                                           cfg['CameraSettings']['num_cams']*
                                                                           cfg['CameraSettings']['img_width'],
                                                                           cfg['CameraSettings']['img_height']
                                                                           )
                                                   }

    return cfg

def _byteify(data, ignore_dicts = False):
    if isinstance(data, unicode):
        return data.encode('utf-8')
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    return data

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def save_cfg(cfg):
    cfgPath = cfg['ReachMaster']['data_dir']+"/config/"
    if not os.path.isdir(cfgPath):
            os.makedirs(cfgPath)
    fn = cfgPath + 'cfg: ' + str(datetime.datetime.now()) + '.txt'
    with open(fn, 'w') as outfile:
        json.dump(cfg, outfile, indent=4)

def save_tmp(cfg):
    cfgPath = "./temp/"
    if not os.path.isdir(cfgPath):
            os.makedirs(cfgPath)
    fn = cfgPath + 'tmp_config.txt'
    with open(fn, 'w') as outfile:
        json.dump(cfg, outfile, indent=4)