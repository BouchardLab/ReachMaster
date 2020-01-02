"""The robot settings window is opened as a child of the 
ReachMaster root application. It allows the user to set any
parameters located on the robot microcontroller (e.g., 
position smoothing, valve periods etc.), or having to do 
with the robot kinematics. It provides options for the user 
to select previously generated calibration and position
command files. Soon it will allow users to run a robot 
calibration from within the ReachMaster application.

Todo:
    * Add labels to specify the microcontroller/kinematics parameters
    * Run calibration callback
    * Re-add load calibration callback?
    * Automate unit tests
    * Python 3 compatibility
    * PEP 8

"""

from .. import config
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import numpy as np
import os

class RobotSettings(tk.Toplevel):
    """The primary class for the robot settings window.

    Configures and provides window callback methods. Also provides 
    methods to . The 
    `on_quit()` method is called prior to application destruction.   

    Attributes
    ----------
    config : dict
        The current configuration settings for the application
    pos_smoothing : instance
        Tkinter StringVar that captures the user-selected position
        smoothing parameter (0-1, 0 less smoothing).
    tol : instance
        Tkinter StringVar that captures the user-selected tolerance
        (bits) for accepting robot convergence to the command 
        position. (i.e., the L2-norm of the potentiometer readings 
        must be less than tol).
    period : instance
        Tkinter StringVar that captures the user-selected valve 
        period (us). Equivalent to pulse width modulation (PWM) 
        period.
    off_dur : instance
        Tkinter StringVar that captures the user-selected minimum
        off duration (ms) that is enforced between convergence 
        times. Once the robot converges to a command position, it
        will not move to a new command position for this duration.
    num_tol : instance
        Tkinter StringVar that captures the user-selected number of 
        consecutive periods the robot must be within convergence in
        order to be considered "in position".
    (x/y/z)_(push/pull)_wt : instance
        Tkinter StringVar that captures the user-selected weighting 
        coefficient placed on the `x/y/z` actuator in the 
        `push/pull` direction. Acts as a gain on the calibration 
        coefficients. Typically defaults to 1 for a good 
        calibration, but gives the user an option to make quick 
        manual adjustments without runnning an entirely new 
        calibration. Not used very often in practice.
    rew_zone_x : instance
        Tkinter StringVar that captures the user-selected minimum
        `x` position (bits) the handle must cross in order to enter
        the reward zone.
    rew_zone_(y/z)_(min/max) : instance
        Tkinter StringVar that captures the user-selected min/max
        `x/z` positions (bits) the handle must be within in order to 
        be located in the reward zone. Conditions must be satisfied 
        on all actuators simultaneously.
    calibration_file : instance
        Tkinter StringVar that captures the user-selected robot 
        calibration file.
    command_file : instance
        Tkinter StringVar that captures the user-selected position
        command file.
    command_type : instance
        Tkinter StringVar that captures the user-selected option 
        determining how commands are generated. Can be 
        'read_from_file', 'sample_from_file', or 'parametric_sample'.
        'read_from_file' takes the commands directly from the command
        file. 'sample_from_file' generates a sequence of commands by
        sampling from the command file with replacement. 
        'parametric_sample' does not use the command file. Rather, it
        samples commands uniformly from the `reach volume` determined
        by user-selected inverse kinematics parameters. 
    (y/z)gimbal_to_joint : instance
        Tkinter StringVar that captures the user-measured 
        center-to-center distance (mm) from an y/z gimbal to its
        respective spherical joint (link to hardware page).
    (x/y/z)gimbal_(x/y/z)offset : instance
        Tkinter StringVar that captures the user-measured offset (mm)
        of the (x/y/z) gimbal from the origin in the (x/y/z) 
        direction.
    (x/y/z)_origin : instance
        Tkinter StringVar that captures the user-measured (x/y/z) 
        position (bits) of the origin.
    reach_dist_(min/max) : instance
        Tkinter StringVar that captures the user-specified (min/max)
        reach distance of the `reach volume`.  
    reach_angle_max : instance
        Tkinter StringVar that captures the user-specified max reach 
        angle (rad) of the `reach volume`.  

    """

    def __init__(self, parent):
        #create window
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()
        self.title('Robot Settings')   
        self.configure(bg='white')
        self.protocol('WM_DELETE_WINDOW', self.on_quit)
        #initialize tk variables from config
        self.config = config.load_config(open('./temp/tmp_config.txt'))
        self.pos_smoothing = tk.StringVar()
        self.pos_smoothing.set(str(self.config['RobotSettings']['pos_smoothing']))
        self.tol = tk.StringVar()
        self.tol.set(str(self.config['RobotSettings']['tol']))
        self.period = tk.StringVar()
        self.period.set(str(self.config['RobotSettings']['period']))
        self.off_dur = tk.StringVar()
        self.off_dur.set(str(self.config['RobotSettings']['off_dur']))
        self.num_tol = tk.StringVar()
        self.num_tol.set(str(self.config['RobotSettings']['num_tol']))
        self.x_push_wt = tk.StringVar()
        self.x_push_wt.set(str(self.config['RobotSettings']['x_push_wt']))
        self.x_pull_wt = tk.StringVar()
        self.x_pull_wt.set(str(self.config['RobotSettings']['x_pull_wt']))
        self.y_push_wt = tk.StringVar()
        self.y_push_wt.set(str(self.config['RobotSettings']['y_push_wt']))
        self.y_pull_wt = tk.StringVar()
        self.y_pull_wt.set(str(self.config['RobotSettings']['y_pull_wt']))
        self.z_push_wt = tk.StringVar()
        self.z_push_wt.set(str(self.config['RobotSettings']['z_push_wt']))
        self.z_pull_wt = tk.StringVar()
        self.z_pull_wt.set(str(self.config['RobotSettings']['z_pull_wt']))
        self.rew_zone_x = tk.StringVar()
        self.rew_zone_x.set(str(self.config['RobotSettings']['rew_zone_x']))
        self.rew_zone_y_min = tk.StringVar()
        self.rew_zone_y_min.set(str(self.config['RobotSettings']['rew_zone_y_min']))
        self.rew_zone_y_max = tk.StringVar()
        self.rew_zone_y_max.set(str(self.config['RobotSettings']['rew_zone_y_max']))
        self.rew_zone_z_min = tk.StringVar()
        self.rew_zone_z_min.set(str(self.config['RobotSettings']['rew_zone_z_min']))
        self.rew_zone_z_max = tk.StringVar()
        self.rew_zone_z_max.set(str(self.config['RobotSettings']['rew_zone_z_max']))
        self.calibration_file = tk.StringVar()
        self.calibration_file.set(str(self.config['RobotSettings']['calibration_file']))
        self.command_file = tk.StringVar()
        self.command_file.set(str(self.config['RobotSettings']['command_file']))
        self.command_type = tk.StringVar()
        self.command_type.set(self.config['RobotSettings']['command_type'])
        self.ygimbal_to_joint = tk.StringVar()
        self.ygimbal_to_joint.set(str(self.config['RobotSettings']['ygimbal_to_joint']))
        self.zgimbal_to_joint = tk.StringVar()
        self.zgimbal_to_joint.set(str(self.config['RobotSettings']['zgimbal_to_joint']))
        self.xgimbal_xoffset = tk.StringVar()
        self.xgimbal_xoffset.set(str(self.config['RobotSettings']['xgimbal_xoffset']))
        self.ygimbal_yoffset = tk.StringVar()
        self.ygimbal_yoffset.set(str(self.config['RobotSettings']['ygimbal_yoffset']))
        self.zgimbal_zoffset = tk.StringVar()
        self.zgimbal_zoffset.set(str(self.config['RobotSettings']['zgimbal_zoffset']))
        self.x_origin = tk.StringVar()
        self.x_origin.set(str(self.config['RobotSettings']['x_origin']))
        self.y_origin = tk.StringVar()
        self.y_origin.set(str(self.config['RobotSettings']['y_origin']))
        self.z_origin = tk.StringVar()
        self.z_origin.set(str(self.config['RobotSettings']['z_origin']))
        self.reach_dist_min = tk.StringVar()
        self.reach_dist_min.set(str(self.config['RobotSettings']['reach_dist_min']))
        self.reach_dist_max = tk.StringVar()
        self.reach_dist_max.set(str(self.config['RobotSettings']['reach_dist_max']))
        self.reach_angle_max = tk.StringVar()
        self.reach_angle_max.set(str(self.config['RobotSettings']['reach_angle_max']))
        #configure window
        self._configure_window()

    def on_quit(self):      
        """Called prior to destruction of the robot settings window.

        Prior to destruction, the configuration file must be updated
        to reflect the change in settings. 

        """    
        self.config['RobotSettings']['pos_smoothing'] = float(self.pos_smoothing.get())
        self.config['RobotSettings']['tol'] = float(self.tol.get())
        self.config['RobotSettings']['period'] = float(self.period.get())
        self.config['RobotSettings']['off_dur'] = int(self.off_dur.get()) 
        self.config['RobotSettings']['num_tol'] = int(self.num_tol.get())
        self.config['RobotSettings']['x_push_wt'] = float(self.x_push_wt.get())
        self.config['RobotSettings']['x_pull_wt'] = float(self.x_pull_wt.get())
        self.config['RobotSettings']['y_push_wt'] = float(self.y_push_wt.get())
        self.config['RobotSettings']['y_pull_wt'] = float(self.y_pull_wt.get())
        self.config['RobotSettings']['z_push_wt'] = float(self.z_push_wt.get())
        self.config['RobotSettings']['z_pull_wt'] = float(self.z_pull_wt.get())
        self.config['RobotSettings']['rew_zone_x'] = int(self.rew_zone_x.get())
        self.config['RobotSettings']['rew_zone_y_min'] = int(self.rew_zone_y_min.get())
        self.config['RobotSettings']['rew_zone_y_max'] = int(self.rew_zone_y_max.get())
        self.config['RobotSettings']['rew_zone_z_min'] = int(self.rew_zone_z_min.get())
        self.config['RobotSettings']['rew_zone_z_max'] = int(self.rew_zone_z_max.get())
        self.config['RobotSettings']['calibration_file'] = self.calibration_file.get()
        self.config['RobotSettings']['command_file'] = self.command_file.get()
        self.config['RobotSettings']['command_type'] = self.command_type.get()        
        self.config['RobotSettings']['ygimbal_to_joint'] = int(self.ygimbal_to_joint.get())
        self.config['RobotSettings']['zgimbal_to_joint'] = int(self.zgimbal_to_joint.get())
        self.config['RobotSettings']['xgimbal_xoffset'] = int(self.xgimbal_xoffset.get())
        self.config['RobotSettings']['ygimbal_yoffset'] = int(self.ygimbal_yoffset.get())
        self.config['RobotSettings']['zgimbal_zoffset'] = int(self.zgimbal_zoffset.get())
        self.config['RobotSettings']['x_origin'] = int(self.x_origin.get())
        self.config['RobotSettings']['y_origin'] = int(self.y_origin.get())
        self.config['RobotSettings']['z_origin'] = int(self.z_origin.get())
        self.config['RobotSettings']['reach_dist_min'] = int(self.reach_dist_min.get())
        self.config['RobotSettings']['reach_dist_max'] = int(self.reach_dist_max.get())
        self.config['RobotSettings']['reach_angle_max'] = float(self.reach_angle_max.get())        
        config.save_tmp(self.config)
        self.destroy()

    def _configure_window(self):
        tk.Label(
            self,
            text = 'Position Smoothing:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row=1, column=0)   
        tk.Entry(self, textvariable = self.pos_smoothing, width = 17).grid(row = 1, column = 1)
        tk.Label(
            self,
            text = 'Valve Period (usec):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row=2, column=0)   
        tk.Entry(self, textvariable = self.period, width = 17).grid(row = 2, column = 1)
        tk.Label(
            self,
            text = 'Off Duration (msec):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 3, column = 0)   
        tk.Entry(self, textvariable = self.off_dur, width = 17).grid(row = 3, column = 1)
        tk.Label(
            self,
            text = 'Converge Tolerance (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 4, column = 0)   
        tk.Entry(self, textvariable = self.tol, width = 17).grid(row = 4, column = 1)
        tk.Label(
            self,
            text = '# w/in Tolerance:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 5, column = 0)   
        tk.Entry(self, textvariable = self.num_tol, width = 17).grid(row = 5, column = 1)
        tk.Label(
            self,
            text = 'X Push Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 6, column = 0)   
        tk.Entry(self, textvariable = self.x_push_wt, width = 17).grid(row = 6, column = 1)
        tk.Label(
            self,
            text = 'X Pull Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 7, column = 0)   
        tk.Entry(self, textvariable = self.x_pull_wt, width = 17).grid(row = 7, column = 1)
        tk.Label(
            self,
            text = 'Y Push Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 8, column = 0)   
        tk.Entry(self, textvariable = self.y_push_wt, width = 17).grid(row = 8, column = 1)
        tk.Label(self,
            text = 'Y Pull Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 9, column = 0)   
        tk.Entry(self, textvariable = self.y_pull_wt, width = 17).grid(row = 9, column = 1)
        tk.Label(
            self,
            text = 'Z Push Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 10, column = 0)   
        tk.Entry(self, textvariable = self.z_push_wt, width = 17).grid(row = 10, column = 1)
        tk.Label(
            self,
            text = 'Z Pull Weight:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,anchor='e'
            ).grid(row = 11, column = 0)   
        tk.Entry(self, textvariable = self.z_pull_wt, width = 17).grid(row = 11, column = 1)
        tk.Label(
            self,
            text = 'Reward Zone X Min (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 12, column = 0)   
        tk.Entry(self, textvariable = self.rew_zone_x, width = 17).grid(row = 12, column = 1)
        tk.Label(
            self,
            text = 'Reward Zone Y Min (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 13, column = 0)   
        tk.Entry(self, textvariable = self.rew_zone_y_min, width = 17).grid(row = 13, column = 1)
        tk.Label(
            self,
            text = 'Reward Zone Y Max (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 14, column = 0)   
        tk.Entry(self, textvariable = self.rew_zone_y_max, width = 17).grid(row = 14, column = 1)
        tk.Label(
            self,
            text = 'Reward Zone Z Min (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 15, column = 0)   
        tk.Entry(self, textvariable = self.rew_zone_z_min, width = 17).grid(row = 15, column = 1)
        tk.Label(
            self,
            text = 'Reward Zone Z Max (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 16, column = 0)   
        tk.Entry(self, textvariable = self.rew_zone_z_max, width = 17).grid(row = 16, column = 1)
        tk.Button(
            self,
            text = 'Run Calibration',
            font = 'Arial 10 bold',
            width = 14,
            command = self.run_calibration_callback
            ).grid(row = 1, column = 5)
        tk.Label(
            self,
            text = 'Calibration File:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 2, column = 4)
        tk.Label(self, textvariable = self.calibration_file, bg = 'white').grid(row = 2, column = 5)
        tk.Button(
            self,
            text = 'Browse', 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.calibration_browse_callback
            ).grid(row = 2, column = 6)
        tk.Label(
            self,
            text = 'Command File:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 3, column = 4)
        tk.Label(self, textvariable = self.command_file, bg = 'white').grid(row = 3, column = 5)
        tk.Button(
            self,
            text = 'Browse', 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.command_browse_callback
            ).grid(row = 3, column = 6)
        tk.Label(
            self,
            text = 'Command Type:', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 4, column = 4)
        self.command_type_menu = tk.OptionMenu(
            self,
            self.command_type,
            'read_from_file',
            'sample_from_file',
            'parametric_sample'            
            )
        self.command_type_menu.configure(width = 26)
        self.command_type_menu.grid(row = 4, column = 5)
        tk.Label(
            self,
            text = 'Min Reach Distance (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 5, column = 4)   
        tk.Entry(self, textvariable = self.reach_dist_min, width = 17).grid(row = 5, column = 5)
        tk.Label(
            self,
            text = 'Max Reach Distance (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 6, column = 4)   
        tk.Entry(self, textvariable = self.reach_dist_max, width = 17).grid(row = 6, column = 5)
        tk.Label(
            self,
            text = 'Max Reach Angle (rad):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 7, column = 4)   
        tk.Entry(self, textvariable = self.reach_angle_max, width = 17).grid(row = 7, column = 5)
        tk.Label(
            self,
            text = 'Y Gimbal to Joint (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 8, column = 4)   
        tk.Entry(self, textvariable = self.ygimbal_to_joint, width = 17).grid(row = 8, column = 5)
        tk.Label(
            self,
            text = 'Z Gimbal to Joint (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 9, column = 4)   
        tk.Entry(self, textvariable = self.zgimbal_to_joint, width = 17).grid(row = 9, column = 5)
        tk.Label(
            self,
            text = 'X Gimbal X Offset (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 10, column = 4)   
        tk.Entry(self, textvariable = self.xgimbal_xoffset, width = 17).grid(row = 10, column = 5)
        tk.Label(
            self,
            text = 'Y Gimbal Y Offset (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 11, column = 4)   
        tk.Entry(self, textvariable = self.ygimbal_yoffset, width = 17).grid(row = 11, column = 5)
        tk.Label(
            self,
            text = 'Z Gimbal Z Offset (mm):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 12, column = 4)   
        tk.Entry(self, textvariable = self.zgimbal_zoffset, width = 17).grid(row = 12, column = 5)
        tk.Label(
            self,
            text = 'X Origin (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 13, column = 4)   
        tk.Entry(self, textvariable = self.x_origin, width = 17).grid(row = 13, column = 5)
        tk.Label(
            self,
            text = 'Y Origin (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 14, column = 4)   
        tk.Entry(self, textvariable = self.y_origin, width = 17).grid(row = 14, column = 5)
        tk.Label(
            self,
            text = 'Z Origin (bits):', 
            font = 'Arial 10 bold', 
            bg = 'white',
            width = 26,
            anchor = 'e'
            ).grid(row = 15, column = 4)   
        tk.Entry(self, textvariable = self.z_origin, width = 17).grid(row = 15, column = 5)

    # Callbacks -----------------------------------------------------------------------------------------

    def run_calibration_callback(self):
        """Not yet implented. Should load the robot calibration script to
        the robot microcontroller along with any relevant arguments, and 
        run the calibration."""
        print('not implemented')

    def calibration_browse_callback(self):
        """Allows the user to select a calibration file to be loaded to the
        robot microcontroller."""
        self.calibration_file.set(tkFileDialog.askopenfilename())
        self.config['RobotSettings']['calibration_file'] = self.calibration_file.get()                  

    def command_browse_callback(self):
        """Allows the user to select a position command file to be loaded 
        to the robot microcontroller."""
        self.command_file.set(tkFileDialog.askopenfilename())
        self.config['RobotSettings']['command_file'] = self.command_file.get()