"""The protocol window is opened as a child of the 
ReachMaster root application whenever a protocol is run. It 
provides basic functionality for interacting with the rig 
while an experiment is running (e.g., for manual reward 
delivery, toggling lights, etc.).

Todo:
    * Keep adding new protocol types
    * Automate unit tests

"""

from . import config
from .interfaces import camera_interface as camint
from .interfaces import robot_interface as robint
from .interfaces import experiment_interface as expint
import tkinter as tk 
import tkinter.filedialog
import tkinter.messagebox
from time import time, sleep
import datetime
import os 
import numpy as np
import threading

def list_protocols():
    """Generate a list of the available protocol types. Currently 
    limited to 'TRIALS' and 'CONTINUOUS'."""
    protocol_list = list(['TRIALS','CONTINUOUS'])
    return protocol_list

class Protocols(tk.Toplevel):
    """The primary class for the protocol window.

    Configures and provides window callback methods. Also provides 
    special `run` methods that implement each of the protocol types.
    The run methods work in concert with the interface modules and
    microntroller scripts to implement the event logic required by
    a specific experiment. If an experiment requires a new protocol
    type that is not currently available, implementing a new run 
    method will often be a good place to start.      

    Attributes
    ----------
    ready : bool
        True when it is okay to call the run method.
    config : dict
        The current configuration settings for the application.
    exp_connected : bool
        True when experiment controller interface is activated.
    rob_connected : bool
        True when robot controller interface is activated.
    cams_connected : bool
        True when camera interface is activated.
    lights_on : bool
        True if neopixel lights are on.
    baseline_acquired : bool
        True is camera processes have collected all baseline images.
    reach_detected : bool
        True if the minimum pixel of interest deviaton across all
        cameras in greater than the selected threshold.
    reach_init : int
        Time at which most recent reach was detected.
    control_message : char
        Character message that is sent to the experiment controller
        according to the communication protocol.
    exp_response : char
        Character received in response to the control_message from 
        the experiment controller interpreted according to the 
        communication protocol. 

    """

    def __init__(self, parent):
        self.ready = False
        #create window
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()        
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)         
        self.config = config.load_config('./temp/tmp_config.json')           
        self.title("Protocol: " + self.config['Protocol']['type'])                    
        #initialize protocol variables         
        self.exp_connected = False
        self.rob_connected = False
        self.cams_connected = False         
        self.lights_on = False
        self.baseline_acquired = False  
        self.reach_detected = False
        self.lick_window = False
        self.reach_init = 0      
        #check config for errors
        if len(self.config['CameraSettings']['saved_pois']) == 0:
           tkinter.messagebox.showinfo("Warning", "No saved POIs")
           self.on_quit()
           return
        #start interfaces, load settings and acquire baseline for reach detection

        print("starting interfaces...")
        self.exp_controller = expint.start_interface(self.config)
        self.exp_connected = True
        print("loading experiment settings...")
        expint.set_exp_controller(self.exp_controller, self.config)
        self.rob_controller = robint.start_interface(self.config)
        self.rob_connected = True
        print("loading robot settings...")
        self.config = robint.set_rob_controller(self.rob_controller, self.config)
        self.cams = camint.CameraInterface(self.config)
        self.cams_connected = True
        self.cams.start_protocol_interface() # start the camera interface
        #self.cam_thread = threading.Thread(target=self.cam_init())
        #self.cam_thread.start()
        sleep(30) # give the cameras time to start
        self._acquire_baseline()

        self._init_data_output()      
        self._configure_window() 
        self.control_message = 'b'
        while not self.cams.all_triggerable():
            pass
        self.exp_response = expint.start_experiment(self.exp_controller)    
        self.ready = True                         


    def cam_init(self):
        self.cams.start_protocol_interface()


    def on_quit(self):
        """Called prior to destruction protocol window.

        Prior to destruction, all interfaces must be stopped.

        """
        self.ready = False
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
        if self.cams_connected:
            self.cam_thread.join()
            self.cams.stop_interface()
        self.destroy()

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
        for cnt in range(num_imgs):
            while not self.cams.all_triggerable():
                pass
            expint.trigger_image(self.exp_controller)
            self.cams.triggered()
        self.baseline_acquired = True
        print("Baseline acquired!")
        return 1
        
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
        the controller data output file.   

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement. 
            * Absorb communication codes into experiment interface module and document

        """
        now = str(int(round(time()*1000)))  
        if self.exp_response[3]=='1':
            self.cams.triggered() 
            self.lights_on = 1             
            self.poi_deviation =  self.cams.get_poi_deviation()
            while not self.cams.all_triggerable():
                pass
        else:
            self.lights_on = 0
            self.poi_deviation = 0
        expint.write_message(self.exp_controller, self.control_message) 
        self.exp_response = expint.read_response(self.exp_controller)
        self.outputfile.write(
            now + " " + self.exp_response[0:-1:1] + " " + str(self.poi_deviation) + "\n"
            )
        self.exp_response = self.exp_response.split() 
        if (
            self.exp_response[1] == 's' and 
            self.exp_response[2] == '0' and 
            self.poi_deviation > self.config['CameraSettings']['poi_threshold']
            ):  
            self.reach_init = now  
            self.reach_detected = True   
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            self.poi_deviation = 0
            self.control_message = 's'   
            self.reach_detected = False
            print((self.exp_response[0]))
        elif (
            self.reach_detected and 
            (int(now) - int(self.reach_init)) > 
            self.config['ExperimentSettings']['reach_timeout'] and 
            self.exp_response[4] == '0'
            ):
            self.move_robot_callback()
            self.reach_detected = False

    def run_trials(self): 
        """Operations performed for a single iteration of protocol type TRIALS.

        If an image is triggered by the experiment controller, reach detection is
        performed on the images. For this protocol type, the lights are turned off 
        during the intertrial interval and a separate video is encoded for each 
        trial. Messages are sent to the experiment controller to signal important 
        events such as reach detections, beginnings/ends of trials, and to initiate 
        robot movement. Responses are read from the experiment controller and saved 
        to the controller data output file.    

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement.
            * Absorb communication codes into experiment interface module and document

        """
        now = str(int(round(time()*1000)))   
        if self.exp_response[3] == '1': 
            self.cams.triggered() 
            self.lights_on = 1             
            self.poi_deviation =  self.cams.get_poi_deviation()
            while not self.cams.all_triggerable():
                pass
        else:
            self.lights_on = 0
            self.poi_deviation = 0   
        expint.write_message(self.exp_controller, self.control_message)  
        self.exp_response = expint.read_response(self.exp_controller) 
        self.outputfile.write(
            now + " " + self.exp_response[0:-1:1] + " " + str(self.poi_deviation) + "\n"
            )
        self.exp_response = self.exp_response.split() 
        if (
            self.exp_response[1] == 's' and 
            self.exp_response[2] == '0' and 
            self.poi_deviation > self.config['CameraSettings']['poi_threshold']
            ): 
            self.reach_init = now
            self.reach_detected = True
            self.reach_detected = True     
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            self.cams.trial_ended()
            self.poi_deviation = 0
            self.control_message = 's' 
            self.reach_detected = False
            print((self.exp_response[0]))
        elif (
            self.reach_detected and 
            (int(now) - int(self.reach_init)) > 
            self.config['ExperimentSettings']['reach_timeout'] and 
            self.exp_response[4] == '0'
            ):
            self.move_robot_callback()
            self.reach_detected = False

    def run(self):
        """Execute a single iteration of the selected protocol type.
        
        The currently supported protocol types are TRIALS and 
        CONTINUOUS. In order to add a custom type, developers must 
        write a `run_newtype()` method and add it to the call 
        search here. The run methods for each protocol type are 
        where all the vital real-time operations are executed via 
        communication with the interface modules.

        """
        if self.config['Protocol']['type'] == 'CONTINUOUS':
            self.run_continuous()
        elif self.config['Protocol']['type'] == 'TRIALS':
            self.run_trials()
        else:
            print("Invalid protocol!")
            self.on_quit()
        