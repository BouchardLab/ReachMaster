import config
import interfaces.camera_interface as camint
import interfaces.robot_interface as robint
import interfaces.experiment_interface as expint
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import time
import datetime
import serial
import os 
from collections import deque
from vidgear.gears import WriteGear

class Protocols(tk.Toplevel):

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()        
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)  
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))        
        if len(self.cfg['CameraSettings']['saved_pois']) == 0:
           print("No saved POIs.") 
           self.on_quit()   
        self.output_params = self.cfg['CameraSettings']['output_params']
        self.title("Protocol: " + self.cfg['Protocol'])   
        #interface variables
        self.exp_controller = []
        self.rob_controller = []
        self.cams = []
        self.exp_connected = False
        self.rob_connected = False
        self.cams_connected = False
        self.rob_calibration_loaded = False
        self.rob_commands_loaded = False
        #protocol variables             
        self.baseline_acquired = False
        self.poi_means = []
        self.poi_stds = []
        self.obs_pois = []
        self.zscored_pois = []  
        self.lights_on = False          
        self.buffer_full = False 
        try:
            self.exp_controller = expint.start_interface(self.cfg)
            self.exp_connected = True
            self.rob_controller = robint.start_interface(self.cfg)
            self.rob_connected = True
            if self.cfg['RobotSettings']['calibration_file'] != 'None':
                robint.load_calibration_all(self.rob_controller, self.cfg)
            if self.cfg['RobotSettings']['command_type'] != 'None':
                self.cfg = robint.load_commands_all(self.rob_controller, self.cfg)
            self.cams = camint.start_interface(self.cfg)
            self.cams_connected = True
            self.img = camint.init_image()
            self.acquire_baseline()
        except Exception as err:
            print(err)
            self.on_quit()             
        #setup data output files
        self.controller_data_path = self.cfg['ReachMaster']['data_dir'] + "/controller_data/"
        self.video_data_path = self.cfg['ReachMaster']['data_dir'] + "/videos/" 
        if not os.path.isdir(self.controller_data_path):
            os.makedirs(self.controller_data_path)
        if not os.path.isdir(self.video_data_path):
            os.makedirs(self.video_data_path)
        controller_data_file = self.controller_data_path + str(datetime.datetime.now())    
        self.outputfile = open(controller_data_file, "w+")
        header = "time trial serPNS triggered inRewardWin zPOI"
        self.outputfile.write(header + "\n")
        self.control_message = "b"         
        self.exp_controller.write(self.control_message)
        while not self.exp_controller.in_waiting:
            pass
        self.newline = self.exp_controller.readline().split()
        self.exp_controller.flushInput()
        self.init_protocol()
        self.configure_window()        
        print('trials completed:')
        print(self.newline[0])
        self.reach_detected = False
        self.protocol_running = True                   

    def on_quit(self):
        expint.stop_interface(self.exp_controller)
        robint.stop_interface(self.rob_controller)
        self.destroy()

    def init_protocol(self):
        if self.cfg['Protocol'] == "CONTINUOUS":
            self.vid_fn = self.video_data_path + str(datetime.datetime.now()) + '.mp4' 
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **self.output_params)

    def configure_window(self):
        tk.Button(text="Move Robot", font='Arial 10 bold',width=16, command=self.move_rob_callback).grid(row=0, sticky='W')
        tk.Button(text="Toggle LED", font='Arial 10 bold',width=14, command=expint.toggle_led(self.exp_controller)).grid(row=1, sticky='W')
        tk.Button(text="Toggle Lights", font='Arial 10 bold',width=14, command=self.lights_callback).grid(row=2, sticky='W')
        tk.Button(text="Deliver Water", font='Arial 10 bold',width=14, command=expint.deliver_water(self.exp_controller)).grid(row=3, sticky='W')

    def move_rob_callback(self):
        expint.move_robot(self.exp_controller)
        if self.protocol == "TRIALS":
            self.lights_on = 0

    def lights_callback(self):
        expint.toggle_lights(self.exp_controller)
        self.lights_on = not self.lights_on

    def acquire_baseline(self):
        if not self.lights_on:
            self.lights_callback()        
        num_imgs = int(np.round(float(self.cfg['ExperimentSettings']['baseline_dur'])*\
            float(self.cfg['CameraSettings']['fps']),decimals=0)) 
        for i in range(self.cfg['CameraSettings']['num_cams']):
            self.baseline_pois.append(np.zeros(shape = (len(self.cfg['CameraSettings']['saved_pois'][i]), num_imgs)))
        self.control_message = "s"
        print("Acquiring baseline...")
        for cnt in range(num_imgs):
            time.sleep(1.0/float(self.cfg['ExperimentSettings']['baseline_dur']))
            self.exp_controller.write(self.control_message)
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.cfg['CameraSettings']['saved_pois'][i])): 
                    self.baseline_pois[i][j,cnt] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
        for i in range(self.cfg['CameraSettings']['num_cams']):   
            self.poi_means.append(np.mean(baseline_pois[i], axis = 1))             
            self.poi_stds.append(np.std(np.sum(np.square(baseline_pois[i]-
                poi_means[i].reshape(len(self.cfg['CameraSettings']['saved_pois'][i]),1)),axis=0)))
            self.obs_pois.append(np.zeros(len(self.cfg['CameraSettings']['saved_pois'][i])))
            self.zscored_pois.append(0)
        self.baseline_acquired = True
        print("Baseline acquired!")

    def run_continuous(self):
        now = str(int(round(time.time()*1000)))  
        if self.newline[3]=='1': 
            self.lights_on = 1 
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.cfg['CameraSettings']['savedPOIs'][i])): 
                    obsPOIs[i][j] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
                self.zscored_pois[i] = np.round(np.sum(np.square(obs_pois[i]-poi_means[i]))/(poi_stds[i]+np.finfo(float).eps),decimals=1)
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))
        else:
            self.lights_on = 0
            for i in range(self.cfg['CameraSettings']['num_cams']):
                self.zscored_pois[i] = 0
        self.exp_controller.write(self.control_message) 
        if self.newline[3]=='1':
            self.video.write(frame)
        while not self.exp_controller.in_waiting:
            pass 
        self.newline = self.exp_controller.readline() 
        self.exp_controller.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(self.zscored_pois))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and self.newline[2] == '0' and min(self.zscored_pois)>self.cfg['CameraSettings']['poi_threshold']: 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.newline[1] == 'e': 
            self.reach_detected = False
            self.control_message = 's'   
            print(self.newline[0])
        elif self.reach_detected and\
         (int(now)-int(self.reach_init))>self.cfg['ExperimentSettings']['reach_timeout'] and self.newline[4]=='0':
            self.move_rob_callback()

    def run_trials(self): 
        now = str(int(round(time.time()*1000)))   
        if self.newline[3]=='1': 
            self.lights_on = 1
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                for j in range(len(self.cfg['CameraSettings']['saved_pois'][i])): 
                    obs_pois[i][j] = npimg[self.cfg['CameraSettings']['saved_pois'][i][j][1],
                        self.cfg['CameraSettings']['saved_pois'][i][j][0]]
                self.zscored_pois[i] = np.round(np.sum(np.square(obs_pois[i]-poi_means[i]))/(poi_stds[i]+np.finfo(float).eps),decimals=1)
                self.img_buffer.append(npimg)
                if len(self.img_buffer)>self.cfg['CameraSettings']['num_cams']*\
                self.cfg['ExperimentSettings']['buffer_dur']*self.cfg['CameraSettings']['fps'] and not self.reach_detected:
                    self.img_buffer.popleft()
        else:
            self.lights_on = 0
            for i in range(self.cfg['CameraSettings']['num_cams']):
                self.zscored_pois[i] = 0     
        self.exp_controller.write(self.control_message) 
        while not self.exp_controller.in_waiting:
            pass 
        self.newline = self.exp_controller.readline() 
        self.exp_controller.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(self.zscored_pois))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and self.newline[2] == '0' and min(self.zscored_pois)>self.cfg['CameraSettings']['poi_threshold']: 
            self.reach_detected = True  
            self.reach_init = now     
            self.control_message = 'r'
        elif self.newline[1] == 'e': 
            if not os.path.isdir(self.video_data_path):
                os.makedirs(self.video_data_path)
            trial_fn = self.video_data_path + 'trial: ' + str(self.newline[0]) + '.mp4' 
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
            print(self.newline[0])
        elif self.reach_detected and (int(now)-int(self.reach_init))>\
        self.cfg['ExperimentSettings']['reach_timeout'] and self.newline[4]=='0':
            self.movRobCallback()

    def run(self):
        if self.protocol is 'TRIALS':
            self.run_trials()
        elif self.protocol is 'CONTINUOUS':
            self.run_continuous()