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
                                   'x_push_dur':'None',
                                   'x_pull_dur':'None',
                                   'y_push_dur':'None',
                                   'y_pull_dur':'None',
                                   'z_push_dur':'None',
                                   'z_pull_dur':'None',
                                   'command_source':'None',
                                   'command_set':'None',
                                   'command_file':'None',
                                   'x_command_pos':'None',
                                   'y_command_pos':'None',
                                   'z_command_pos':'None',
                                   'r_command_pos':'None',
                                   'theta_y_command_pos':'None',
                                   'theta_z_command_pos':'None',
                                   'reach_dist_min':10,
                                   'reach_dist_max':40,
                                   'reach_angle_max':np.pi/3,
                                   'ygimbal_to_joint':64,
                                   'zgimbal_to_joint':47,
                                   'xgimbal_xoffset':168,
                                   'ygimbal_yoffset':100,
                                   'zgimbal_zoffset':117,
                                   'x_origin':1024,
                                   'y_origin':608,
                                   'z_origin':531,
                                   'alpha':0.1,
                                   'tol':np.round(1023.0/50.0/3.0,decimals=1),
                                   'period':125.0*1000.0,
                                   'off_dur':1000,
                                   'num_tol':5,
                                   'x_push_wt':1.0,
                                   'x_pull_wt':1.0,
                                   'y_push_wt':1.0,
                                   'y_pull_wt':1.0,
                                   'z_push_wt':1.0,
                                   'z_pull_wt':1.0,
                                   'rew_zone_x':1000,
                                   'rew_zone_y_min':558,
                                   'rew_zone_y_max':658,
                                   'rew_zone_z_min':481,
                                   'rew_zone_z_max':581
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