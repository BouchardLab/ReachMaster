"""The protocol window is opened as a child of the 
ReachMaster root application whenever a protocol is run. It 
provides basic functionality for interacting with the rig 
while an experiment is running (e.g., for manual reward 
delivery, toggling lights, etc.).

Todo:
    * Keep adding new protocol types
    * GPU-accelerated video encoding
    * Automate unit tests

"""

from . import config
from .interfaces import camera_interface as camint
from .interfaces import robot_interface as robint
from .interfaces import experiment_interface as expint
import tkinter as tk 
import tkinter.filedialog
import tkinter.messagebox
import time
import datetime
import os 
from collections import deque
from vidgear.gears import WriteGear
import numpy as np
import serial
from ximea import xiapi

def list_protocols():
    """Generate a list of the available protocol types. Currently 
    limited to 'TRIALS' and 'CONTINUOUS'."""
    protocol_list = list(["TRIALS","CONTINUOUS"])
    return protocol_list

class Protocols(tk.Toplevel):
    """The primary class for the protocol window.

    Configures and provides window callback methods. Also provides 
    special `run` methods that implement each of the protocol types. 
    Protocol types also has special `on_quit()` methods that are 
    called prior to application destruction.   

    Attributes
    ----------
    config : dict
        The current configuration settings for the application.
    output_params : dict
        Video encoding parameters for WriteGear.
    exp_connected : bool
        True when experiment controller interface is activated.
    rob_connected : bool
        True when robot controller interface is activated.
    cams_connected : bool
        True when camera interface is activated.
    video_open : bool
        True when a file is open for video encoding.
    poi_means : list
        A list of the mean value of each pixel-of-interest for 
        each camera as measured from the baseline acquisition
        period. Used for reach detection.
    poi_stds : list
        A list of standard deviations for each pixel-of-interest 
        for each camera as measured from the baseline acquisition
        period. Used for reach detection.
    poi_obs : list
        A list of all pixel-of-interest values from the most 
        recently captured group of images. Used for reach detection.
    poi_zscores : list
        A list of all pixel-of-interest zscores relative to baseline
        for the most recently captured group of images. Used for 
        reach detection.
    lights_on : bool
        True if neopixel lights are on.
    buffer_full : bool
        True if video buffer is full when `protocol type` is 'TRIALS'
    reach_detected : bool
        True when poi_zscores from all cameras are above the 
        user-specified threshold.  
    control_message : char
        Character message that is sent to the experiment controller
        according to the communication protocol.
    exp_response : char
        Character received in response to the control_message from 
        the experiment controller interpreted according to the 
        communication protocol. 

    """

    def __init__(self, parent):
        #create window
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()        
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)         
        self.config = config.load_config('./temp/tmp_config.json')           
        self.output_params = self.config['CameraSettings']['output_params']
        self.title("Protocol: " + self.config['Protocol']['type'])                    
        #initialize protocol variables 
        self.baseline_acquired = False
        self.exp_connected = False
        self.rob_connected = False
        self.cams_connected = False     
        self.video_open = False       
        self.poi_means = []
        self.poi_stds = []
        self.poi_obs = []
        self.poi_zscores = []  
        self.lights_on = False          
        self.buffer_full = False 
        self.reach_detected = False
        #check config for errors
        if len(self.config['CameraSettings']['saved_pois']) == 0:
           tkinter.messagebox.showinfo("Warning", "No saved POIs")
           self.on_quit()
           return
        #start interfaces, load settings and aquire baseline for reach detection
        try:
            print("starting interfaces...")
            self.exp_controller = expint.start_interface(self.config) 
            self.exp_connected = True 
            print("loading experiment settings...")        
            expint.set_exp_controller(self.exp_controller, self.config)                      
            self.rob_controller = robint.start_interface(self.config) 
            self.rob_connected = True     
            print("loading robot settings...")
            self.config = robint.set_rob_controller(self.rob_controller, self.config)      
            self.cams = camint.start_interface(self.config)         
            self.cams_connected = True                       
            self.img = camint.init_image() 
            self._acquire_baseline() 
            self.baseline_acquired = True          
        except Exception as err:
            print(err)
            self.on_quit()  
            return                  
        self._init_data_output()      
        self._init_special_protocol() 
        self._configure_window() 
        self.control_message = 'b'
        self.exp_response = expint.start_experiment(self.exp_controller)                             

    def on_quit(self):
        """Called prior to destruction protocol window.

        Prior to destruction, all interfaces must be stopped and 
        protocol-specific cleanup needs to take place. 

        """
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
        if self.cams_connected:
            camint.stop_interface(self.cams)
        self._special_protocol_quit()
        self.destroy()

    def _special_protocol_quit(self): 
        if self.config['Protocol']['type'] == 'CONTINUOUS' and self.video_open:
            self.video.close()
        elif self.config['Protocol']['type'] == 'TRIALS':
            self.img_buffer = deque()

    def _acquire_baseline(self):
        print("Acquiring baseline...")
        #make sure lights are on
        if not self.lights_on:
            self.toggle_lights_callback()
        num_imgs = (
            int(
                np.round(
                    float(
                        self.config['ExperimentSettings']['baseline_dur']
                        ) * 
                    float(
                        self.config['CameraSettings']['fps']
                        ), 
                    decimals = 0
                    )
                )
            )
        baseline_pois = []
        for i in range(self.config['CameraSettings']['num_cams']):
            baseline_pois.append(
                np.zeros(shape = (len(self.config['CameraSettings']['saved_pois'][i]), num_imgs))
                )
        #get baseline images and extract sample pois for each camera
        for cnt in range(num_imgs):
            expint.trigger_image(self.exp_controller)
            for i in range(self.config['CameraSettings']['num_cams']):
                # npimg = camint.get_npimage(self.cams[i],self.img)
                self.cams[i].get_image(self.img, timeout = 2000)                  
                npimg = self.img.get_image_data_numpy()
                for j in range(len(self.config['CameraSettings']['saved_pois'][i])): 
                    baseline_pois[i][j,cnt] = npimg[
                    self.config['CameraSettings']['saved_pois'][i][j][1],
                    self.config['CameraSettings']['saved_pois'][i][j][0]
                    ]
        #compute poi stats for each camera
        for i in range(self.config['CameraSettings']['num_cams']):   
            self.poi_means.append(np.mean(baseline_pois[i], axis = 1))             
            self.poi_stds.append(
                np.std(
                    np.sum(
                        np.square(
                    baseline_pois[i] - 
                    self.poi_means[i].reshape(
                        len(
                            self.config['CameraSettings']['saved_pois'][i]), 1
                        )
                    )
                        ,axis=0
                        )
                    )
                )
            self.poi_obs.append(np.zeros(len(self.config['CameraSettings']['saved_pois'][i])))
            self.poi_zscores.append(0)
        print("Baseline acquired!")

    def _init_special_protocol(self):        
        if self.config['Protocol']['type'] == 'CONTINUOUS':
            self.vid_fn = self.video_data_path + str(datetime.datetime.now()) + '.mp4' 
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **self.output_params)
            self.video_open = True
        elif self.config['Protocol']['type'] == 'TRIALS':
            self.img_buffer = deque()

    def _init_data_output(self):
        self.controller_data_path = self.config['ReachMaster']['data_dir'] + "/controller_data/"
        self.video_data_path = self.config['ReachMaster']['data_dir'] + "/videos/" 
        if not os.path.isdir(self.controller_data_path):
            os.makedirs(self.controller_data_path)
        if not os.path.isdir(self.video_data_path):
            os.makedirs(self.video_data_path)
        controller_data_file = self.controller_data_path + str(datetime.datetime.now())
        self.outputfile = open(controller_data_file, "w+")
        header = "time trial exp_response rob_moving image_triggered in_reward_window z_poi"
        self.outputfile.write(header + "\n")

    def _configure_window(self):
        tk.Button(
            self, 
            text = "Move Robot", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.move_robot_callback
            ).grid(row = 0, sticky = 'W')
        tk.Button(
            self, 
            text = "Toggle LED", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.toggle_led_callback
            ).grid(row = 1, sticky = 'W')
        tk.Button(
            self, 
            text = "Toggle Lights", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.toggle_lights_callback
            ).grid(row = 2, sticky = 'W')
        tk.Button(
            self, 
            text = "Deliver Water", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.deliver_water_callback
            ).grid(row = 3, sticky = 'W')

    #Callbacks ------------------------------------------------------------------

    def move_robot_callback(self):
        """Commands the robot to move to the next position in
        its currently loaded command sequence."""
        expint.move_robot(self.exp_controller)
        if self.config['Protocol']['type'] == 'TRIALS':
            self.lights_on = 0

    def toggle_led_callback(self):
        """Toggles the LED located on the robot handle."""
        expint.toggle_led(self.exp_controller)

    def toggle_lights_callback(self):
        """Toggles the neopixel lights."""
        expint.toggle_lights(self.exp_controller)
        self.lights_on = not self.lights_on

    def deliver_water_callback(self):
        """Opens the reward solenoid for the reward duration set 
        in experiment settings."""
        expint.deliver_water(self.exp_controller)

    #Protocol types ---------------------------------------------------------------

    def run_continuous(self):
        """Operations performed for a single iteration of protocol type CONTINOUS.

        If an image is triggered by the experiment controller, reach detection is
        performed on the images, the images are concatenated, and encoded to mp4
        continuously, even during intertrial intervals, in real-time. Consequently,
        for this protocol type, the lights remain on during the intertrial interval. 
        Messages are sent to the experiment controller to signal important events 
        such as reach detections, beginnings/ends of trials, and to initiate robot 
        movement. Responses are read from the experiment controller and saved to 
        the data output file.   

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement. 
            * Absorb communication codes into experiment interface module

        """
        now = str(int(round(time.time()*1000)))  
        if self.exp_response[3]=='1': 
            self.lights_on = 1 
            for i in range(self.config['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.config['CameraSettings']['saved_pois'][i])): 
                    self.poi_obs[i][j] = npimg[
                    self.config['CameraSettings']['saved_pois'][i][j][1],
                    self.config['CameraSettings']['saved_pois'][i][j][0]
                    ]
                self.poi_zscores[i] = np.round(
                    np.sum(
                        np.square(
                            self.poi_obs[i] - self.poi_means[i]
                            )
                        ) / (self.poi_stds[i] + np.finfo(float).eps), decimals = 1)
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))
        else:
            self.lights_on = 0
            for i in range(self.config['CameraSettings']['num_cams']):
                self.poi_zscores[i] = 0
        expint.write_message(self.exp_controller, self.control_message) 
        if self.exp_response[3] == '1':
            self.video.write(frame)
        self.exp_response = expint.read_response(self.exp_controller) 
        self.outputfile.write(
            now + " " + self.exp_response[0:-1:1] + " " + str(min(self.poi_zscores)) + "\n"
            )
        self.exp_response = self.exp_response.split() 
        if (
            self.exp_response[1] == 's' and 
            self.exp_response[2] == '0' and 
            min(self.poi_zscores) > self.config['CameraSettings']['poi_threshold']
            ): 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            self.reach_detected = False
            self.control_message = 's'   
            print((self.exp_response[0]))
        elif (
            self.reach_detected and 
            (int(now) - int(self.reach_init)) > 
            self.config['ExperimentSettings']['reach_timeout'] and 
            self.exp_response[4] == '0'
            ):
            self.move_robot_callback()

    def run_trials(self): 
        """Operations performed for a single iteration of protocol type TRIALS.

        If an image is triggered by the experiment controller, reach detection is
        performed on the images, the images are added/removed to an online buffer, 
        and encoded to mp4 during the intertrial intertval. For this protocol 
        type, the lights are turned off during the intertrial interval. Messages 
        are sent to the experiment controller to signal important events such as 
        reach detections, beginnings/ends of trials, and to initiate robot 
        movement. Responses are read from the experiment controller and saved to 
        the data output file.    

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement.
            * Absorb communication codes into experiment interface module

        """
        now = str(int(round(time.time()*1000)))   
        if self.exp_response[3] == '1': 
            self.lights_on = 1
            for i in range(self.config['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.config['CameraSettings']['saved_pois'][i])): 
                    self.poi_obs[i][j] = npimg[self.config['CameraSettings']['saved_pois'][i][j][1],
                        self.config['CameraSettings']['saved_pois'][i][j][0]]
                self.poi_zscores[i] = np.round(
                    np.sum(
                        np.square(
                            self.poi_obs[i] - self.poi_means[i]
                            )
                        ) / (self.poi_stds[i] + np.finfo(float).eps), decimals=1)
                self.img_buffer.append(npimg)
                if (
                    len(self.img_buffer) > 
                    self.config['CameraSettings']['num_cams'] * 
                    self.config['ExperimentSettings']['buffer_dur'] * 
                    self.config['CameraSettings']['fps'] and not 
                    self.reach_detected
                    ):
                    self.img_buffer.popleft()
        else:
            self.lights_on = 0
            for i in range(self.config['CameraSettings']['num_cams']):
                self.poi_zscores[i] = 0     
        expint.write_message(self.exp_controller, self.control_message)  
        self.exp_response = expint.read_response(self.exp_controller) 
        self.outputfile.write(
            now + " " + self.exp_response[0:-2:1] + " " + str(min(self.poi_zscores)) + "\n"
            )
        self.exp_response = self.exp_response.split() 
        if (
            self.exp_response[1] == 's' and 
            self.exp_response[2] == '0' and 
            min(self.poi_zscores) > self.config['CameraSettings']['poi_threshold']
            ): 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            if not os.path.isdir(self.video_data_path):
                os.makedirs(self.video_data_path)
            trial_fn = self.video_data_path + 'trial: ' + str(self.exp_response[0]) + '.mp4' 
            self.video = WriteGear(
                output_filename = trial_fn,
                compression_mode = True,
                logging = False, 
                **self.output_params
                )
            for i in range(
                len(self.img_buffer) / self.config['CameraSettings']['num_cams']
                ):
                frame = (
                    self.img_buffer[
                    (i + 1) * self.config['CameraSettings']['num_cams'] - 
                    self.config['CameraSettings']['num_cams']
                    ]
                    )
                for f in range(self.config['CameraSettings']['num_cams'] - 1):
                    frame = (
                        np.hstack((
                            frame, self.img_buffer[(i + 1) * 
                            self.config['CameraSettings']['num_cams'] - 
                            self.config['CameraSettings']['num_cams'] + f + 1]
                            )
                        )
                        )
                self.video.write(frame)   
            self.video.close()
            self.reach_detected = False
            self.control_message = 's' 
            self.img_buffer = deque() 
            print((self.exp_response[0]))
        elif (
            self.reach_detected and 
            (int(now) - int(self.reach_init)) > 
            self.config['ExperimentSettings']['reach_timeout'] and 
            self.exp_response[4] == '0'
            ):
            self.move_robot_callback()

    def run(self):
        """Execute a single iteration of the selected protocol type.
        
        The currently supported protocol types are TRIALS and 
        CONTINUOUS. In order to add a custom type, developers must 
        write a `run_newtype()` method and add it to the call 
        search here. The run methods for each protocol type are 
        where all the vital real-time operations are executed.

        """
        #As number of types increase, consider converting to switch
        if self.config['Protocol']['type'] == 'CONTINUOUS':
            self.run_continuous()
        elif self.config['Protocol']['type'] == 'TRIALS':
            self.run_trials()
        else:
            print("Invalid protocol!")
            self.on_quit()
        