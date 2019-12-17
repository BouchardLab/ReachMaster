import config
import interfaces.camera_interface as camint
import interfaces.robot_interface as robint
import interfaces.experiment_interface as expint
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import time
import datetime
import os 
from collections import deque
from vidgear.gears import WriteGear
import numpy as np
import serial
from ximea import xiapi

def list_protocols():
    protocol_list = list(["TRIALS","CONTINUOUS"])
    return protocol_list

class Protocols(tk.Toplevel):

    def __init__(self, parent):
        #create window
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()        
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)         
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))           
        self.output_params = self.cfg['CameraSettings']['output_params']
        self.title("Protocol: " + self.cfg['Protocol']['type'])                    
        #initialize protocol variables 
        self.exp_connected = False
        self.rob_connected = False
        self.cams_connected = False     
        self.video_open = False       
        self.poi_means = []
        self.poi_stds = []
        self.obs_pois = []
        self.zscored_pois = []  
        self.lights_on = False          
        self.buffer_full = False 
        self.reach_detected = False
        #check config for errors
        if len(self.cfg['CameraSettings']['saved_pois']) == 0:
           print("No saved _pois.") 
           self.on_quit()
        #start interfaces, load settings and aquire baseline for reach detection
        try:
            print("starting interfaces...")
            self.exp_controller = expint.start_interface(self.cfg) 
            self.exp_connected = True           
            self.rob_controller = robint.start_interface(self.cfg) 
            self.rob_connected = True           
            self.cams = camint.start_interface(self.cfg)         
            self.cams_connected = True    
            print("loading experiment settings...")        
            expint.set_exp_controller(self.exp_controller, self.cfg)
            print("loading robot settings...")
            self.cfg = robint.set_rob_controller(self.rob_controller, self.cfg)            
            self.img = camint.init_image() 
            self.acquire_baseline()           
        except Exception as err:
            print(err)
            self.on_quit()                    
        self.init_data_output()        
        self.special_protocol_init() 
        self.configure_window() 
        self.control_message = 'b'
        self.exp_response = expint.start_experiment(self.exp_controller)                             

    def on_quit(self):
        #stop interfaces
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
        if self.cams_connected:
            camint.stop_interface(self.cams)
        #stop protocol-specific stuff
        self.special_protocol_quit()
        self.destroy()

    def special_protocol_init(self):        
        if self.cfg['Protocol']['type'] == "CONTINUOUS":
            self.vid_fn = self.video_data_path + str(datetime.datetime.now()) + '.mp4' 
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **self.output_params)
            self.video_open = True

    def special_protocol_quit(self):        
        if self.video_open:
            self.video.close()

    def init_data_output(self):
        self.controller_data_path = self.cfg['ReachMaster']['data_dir'] + "/controller_data/"
        self.video_data_path = self.cfg['ReachMaster']['data_dir'] + "/videos/" 
        if not os.path.isdir(self.controller_data_path):
            os.makedirs(self.controller_data_path)
        if not os.path.isdir(self.video_data_path):
            os.makedirs(self.video_data_path)
        controller_data_file = self.controller_data_path + str(datetime.datetime.now())
        self.outputfile = open(controller_data_file, "w+")
        header = "time trial exp_response rob_moving image_triggered in_reward_window z_poi"
        self.outputfile.write(header + "\n")

    def acquire_baseline(self):
        print("Acquiring baseline...")
        #make sure lights are on
        if not self.lights_on:
            self.toggle_lights_callback()
        num_imgs = (int(np.round(float(self.cfg['ExperimentSettings']['baseline_dur'])*
            float(self.cfg['CameraSettings']['fps']),decimals=0)))
        baseline_pois = []
        for i in range(self.cfg['CameraSettings']['num_cams']):
            baseline_pois.append(np.zeros(shape = (len(self.cfg['CameraSettings']['saved_pois'][i]), num_imgs)))
        #get baseline images and extract sample pois for each camera
        for cnt in range(num_imgs):
            self.exp_controller.write("t")
            for i in range(self.cfg['CameraSettings']['num_cams']):
                # npimg = camint.get_npimage(self.cams[i],self.img)
                self.cams[i].get_image(self.img, timeout = 2000)                  
                npimg = self.img.get_image_data_numpy()
                for j in range(len(self.cfg['CameraSettings']['saved_pois'][i])): 
                    baseline_pois[i][j,cnt] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
        #compute poi stats for each camera
        for i in range(self.cfg['CameraSettings']['num_cams']):   
            self.poi_means.append(np.mean(baseline_pois[i], axis = 1))             
            self.poi_stds.append(np.std(np.sum(np.square(baseline_pois[i]-
                self.poi_means[i].reshape(len(self.cfg['CameraSettings']['saved_pois'][i]),1)),axis=0)))
            self.obs_pois.append(np.zeros(len(self.cfg['CameraSettings']['saved_pois'][i])))
            self.zscored_pois.append(0)
        print("Baseline acquired!")

    def configure_window(self):
        tk.Button(self, text="Move Robot", font='Arial 10 bold',width=16, command=self.move_robot_callback).grid(row=0, sticky='W')
        tk.Button(self, text="Toggle LED", font='Arial 10 bold',width=14, command=self.toggle_led_callback).grid(row=1, sticky='W')
        tk.Button(self, text="Toggle Lights", font='Arial 10 bold',width=14, command=self.toggle_lights_callback).grid(row=2, sticky='W')
        tk.Button(self, text="Deliver Water", font='Arial 10 bold',width=14, command=self.deliver_water_callback).grid(row=3, sticky='W')

    def move_robot_callback(self):
        expint.move_robot(self.exp_controller)
        if self.protocol == "TRIALS":
            self.lights_on = 0

    def toggle_led_callback(self):
        expint.toggle_led(self.exp_controller)

    def toggle_lights_callback(self):
        expint.toggle_lights(self.exp_controller)
        self.lights_on = not self.lights_on

    def deliver_water_callback(self):
        expint.deliver_water(self.exp_controller)

    def run_continuous(self):
        now = str(int(round(time.time()*1000)))  
        if self.exp_response[3]=='1': 
            self.lights_on = 1 
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.cfg['CameraSettings']['saved_pois'][i])): 
                    self.obs_pois[i][j] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
                self.zscored_pois[i] = np.round(np.sum(np.square(self.obs_pois[i]-self.poi_means[i]))/(self.poi_stds[i]+np.finfo(float).eps),decimals=1)
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))
        else:
            self.lights_on = 0
            for i in range(self.cfg['CameraSettings']['num_cams']):
                self.zscored_pois[i] = 0
        expint.write_message(self.exp_controller,self.control_message) 
        if self.exp_response[3]=='1':
            self.video.write(frame)
        self.exp_response = expint.read_message(self.exp_controller) 
        self.outputfile.write(now+" "+self.exp_response[0:-2:1]+" "+str(min(self.zscored_pois))+"\n")
        self.exp_response = self.exp_response.split() 
        if self.exp_response[1] == 's' and self.exp_response[2] == '0' and min(self.zscored_pois)>self.cfg['CameraSettings']['poi_threshold']: 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            self.reach_detected = False
            self.control_message = 's'   
            print(self.exp_response[0])
        elif self.reach_detected and\
         (int(now)-int(self.reach_init))>self.cfg['ExperimentSettings']['reach_timeout'] and self.exp_response[4]=='0':
            self.move_robot_callback()

    def run_trials(self): 
        now = str(int(round(time.time()*1000)))   
        if self.exp_response[3]=='1': 
            self.lights_on = 1
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.cfg['CameraSettings']['saved_pois'][i])): 
                    self.obs_pois[i][j] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
                self.zscored_pois[i] = np.round(np.sum(np.square(self.obs_pois[i]-self.poi_means[i]))/(self.poi_stds[i]+np.finfo(float).eps),decimals=1)
                self.img_buffer.append(npimg)
                if len(self.img_buffer)>self.cfg['CameraSettings']['num_cams']*\
                self.cfg['ExperimentSettings']['buffer_dur']*self.cfg['CameraSettings']['fps'] and not self.reach_detected:
                    self.img_buffer.popleft()
        else:
            self.lights_on = 0
            for i in range(self.cfg['CameraSettings']['num_cams']):
                self.zscored_pois[i] = 0     
        expint.write_message(self.exp_controller,self.control_message)  
        self.exp_response = expint.read_message(self.exp_controller) 
        self.outputfile.write(now+" "+self.exp_response[0:-2:1]+" "+str(min(self.zscored_pois))+"\n")
        self.exp_response = self.exp_response.split() 
        if self.exp_response[1] == 's' and self.exp_response[2] == '0' and min(self.zscored_pois)>self.cfg['CameraSettings']['poi_threshold']: 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.exp_response[1] == 'e': 
            if not os.path.isdir(self.video_data_path):
                os.makedirs(self.video_data_path)
            trial_fn = self.video_data_path + 'trial: ' + str(self.exp_response[0]) + '.mp4' 
            self.video = WriteGear(output_filename = trial_fn,compression_mode = True,logging=False,**self.output_params)
            for i in range(len(self.img_buffer)/self.cfg['CameraSettings']['num_cams']):
                frame = self.img_buffer[(i+1)*self.cfg['CameraSettings']['num_cams']-self.cfg['CameraSettings']['num_cams']]
                for f in range(self.cfg['CameraSettings']['num_cams']-1):
                    frame = np.hstack((frame,self.img_buffer[(i+1)*\
                        self.cfg['CameraSettings']['num_cams']-self.cfg['CameraSettings']['num_cams']+f+1])) 
                self.video.write(frame)   
            self.video.close()
            self.reach_detected = False
            self.control_message = 's' 
            self.img_buffer = deque() 
            print(self.exp_response[0])
        elif self.reach_detected and (int(now)-int(self.reach_init))>\
        self.cfg['ExperimentSettings']['reach_timeout'] and self.exp_response[4]=='0':
            self.move_robot_callback()

    def run(self):
        if self.cfg['Protocol']['type'] == 'CONTINUOUS':
            self.run_continuous()
        elif self.cfg['Protocol']['type'] == 'TRIALS':
            self.run_trials()
        else:
            print("Invalid protocol!")
            self.on_quit()
        