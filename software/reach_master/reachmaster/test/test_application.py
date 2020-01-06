import unittest2
import mock
import Tkinter as tk
import sys
import os
import time
from os import path
if __name__ == '__main__' and __package__ is None:
    #make sure script can be called from home folder
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import application as app
    import config as cfg
else:
    from .. import application as app
    from .. import config as cfg

class TestApplication(unittest2.TestCase):
    """Simple unit tests for the application module.

    """

    def setUp(self):
            self.test_config = cfg.default_config()
            self.rm = app.ReachMaster()

    def tearDown(self):
        self.rm.window.update()
        self.rm.window.destroy()

    def test_attributes_exist(self):
        must_have_attributes = set([
            'protocol_list', 
            'data_dir', 
            'rob_control_port', 
            'protocol_running', 
            'rob_connected', 
            'running', 
            'protocol', 
            'exp_controller_menu', 
            'exp_connected', 
            'window', 
            'exp_control_port', 
            'port_list', 
            'child', 
            'config_file', 
            'protocol_menu', 
            'config', 
            'rob_controller_menu'
            ])
        rm_dict = self.rm.__dict__
        rm_attributes = set(rm_dict.keys())
        #consider testing only a subset of most crucial attributes
        self.assertSetEqual(must_have_attributes, rm_attributes)

    @mock.patch(
        'reachmaster.application.tkFileDialog.askdirectory', 
        mock.MagicMock(return_value = 'mockdir')
        )
    def test_data_dir_browse_callback(self):
        self.rm.data_dir_browse_callback()
        self.rm.window.update()
        config  = cfg.load_config('./temp/tmp_config.json')
        self.assertEqual(config['ReachMaster']['data_dir'], 'mockdir')

    @mock.patch(
        'reachmaster.application.tkMessageBox.showinfo', 
        mock.MagicMock(return_value = None)
        )    
    def test_config_file_browse_callback(self):
        self.rm.config_file_browse_callback()
        #may have to use patch as a context manager...
    
if __name__ == '__main__':
    unittest2.main()

