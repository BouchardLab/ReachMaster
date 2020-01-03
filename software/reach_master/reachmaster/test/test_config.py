import unittest2
import sys
import os
from os import path
if __name__ == '__main__' and __package__ is None:
    #make sure script can be called from home folder
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import config
else:
    from .. import config

class TestConfigModule(unittest2.TestCase):

    def setUp(self):
        self.test_config = config.default_config()
        self.maxDiff = None

    def test_default_config_is_dict(self):        
        self.assertIsInstance(self.test_config,dict)

    def test_default_config_keys_correct(self):
        with self.subTest(key_level = 'root'):
            must_have_keys = set([
                'ReachMaster', 
                'CameraSettings', 
                'ExperimentSettings', 
                'Protocol', 
                'RobotSettings'
            ])
            test_keys = set(self.test_config.keys())
            self.assertSetEqual(test_keys, must_have_keys)
        with self.subTest(key_level = 'ReachMaster'):
            must_have_keys = set([
                'config_file', 
                'rob_control_port', 
                'serial_baud', 
                'control_timeout', 
                'exp_control_port', 
                'data_dir'
                ])
            test_keys = set(self.test_config['ReachMaster'].keys())
            self.assertSetEqual(test_keys, must_have_keys)
        with self.subTest(key_level = 'CameraSettings'):
            must_have_keys = set([
                'saved_pois', 
                'gpo_mode', 
                'gpi_selector', 
                'sensor_feature_value', 
                'gpi_mode', 
                'output_params', 
                'downsampling', 
                'imgdataformat', 
                'img_width', 
                'gpo_selector', 
                'trigger_source', 
                'img_height', 
                'offset_x', 
                'gain', 
                'fps', 
                'poi_threshold', 
                'offset_y', 
                'exposure', 
                'num_cams'
                ])
            test_keys = set(self.test_config['CameraSettings'].keys())
            self.assertSetEqual(test_keys, must_have_keys)
        with self.subTest(key_level = 'ExperimentSettings'):
            must_have_keys = set([
                'reach_timeout', 
                'reach_delay', 
                'flush_dur', 
                'max_rewards', 
                'solenoid_bounce_dur', 
                'lights_off_dur', 
                'solenoid_open_dur', 
                'lights_on_dur', 
                'baseline_dur', 
                'buffer_dur', 
                'reward_win_dur'
                ])
            test_keys = set(self.test_config['ExperimentSettings'].keys())
            self.assertSetEqual(test_keys, must_have_keys)
        with self.subTest(key_level = 'RobotSettings'):
            must_have_keys = set([
                'x_command_pos', 
                'off_dur', 
                'y_pull_dur', 
                'x_pull_dur', 
                'rew_zone_z_max', 
                'y_push_dur', 
                'pos', 
                'calibration_file', 
                'ygimbal_yoffset', 
                'zgimbal_to_joint', 
                'z_pull_dur', 
                'y_pull_wt', 
                'rew_zone_y_max', 
                'z_origin', 
                'command_set', 
                'rew_zone_y_min', 
                'tol', 'y_origin', 
                'x_origin', 
                'x_push_wt', 
                'xgimbal_xoffset', 
                'z_command_pos', 
                'period', 
                'theta_z_command_pos', 
                'rew_zone_x', 
                'zgimbal_zoffset', 
                'reach_angle_max', 
                'x_push_dur', 
                'z_push_dur', 
                'dis', 
                'reach_dist_max', 
                'z_pull_wt', 
                'command_type', 
                'y_push_wt', 
                'num_tol', 
                'reach_dist_min', 
                'z_push_wt', 
                'theta_y_command_pos', 
                'ygimbal_to_joint', 
                'r_command_pos', 
                'y_command_pos', 
                'pos_smoothing', 
                'x_pull_wt', 
                'command_file', 
                'rew_zone_z_min'
                ])
            test_keys = set(self.test_config['RobotSettings'].keys())
            self.assertSetEqual(test_keys, must_have_keys)
        with self.subTest(key_level = 'Protocol'):
            must_have_keys = set(['type'])
            test_keys = set(self.test_config['Protocol'].keys())
            self.assertSetEqual(test_keys, must_have_keys)
        
    def test_loaded_saved_configs_equal(self):
        with self.subTest(test_fun = 'save_config'):
            fn = config.save_config(self.test_config)
            loaded_config = config.load_config(fn)
            os.remove(fn)
            self.assertEqual(loaded_config, self.test_config)
        with self.subTest(test_fun = 'save_tmp'):
            fn = config.save_tmp(self.test_config)            
            loaded_config = config.load_config(fn)
            os.remove(fn)
            self.assertEqual(loaded_config, self.test_config)

if __name__ == '__main__':
    unittest2.main()