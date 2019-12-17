from .. import config
from ..interfaces import experiment_interface as expint
from ..interfaces import camera_interface as camint
import Tkinter as tk 
import tkMessageBox
import cv2
import PIL.Image, PIL.ImageTk
from ximea import xiapi
import time
import datetime
import numpy as np
import serial
import os 
from collections import deque
from vidgear.gears import WriteGear

class CameraSettings(tk.Toplevel):

    def __init__(self, parent):
        #create window
    	tk.Toplevel.__init__(self, parent)
    	self.transient(parent) 
    	self.grab_set()
        self.title("Camera Settings")
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit) 
        #initialize tk variables from config
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
        self.output_params = self.cfg['CameraSettings']['output_params'] 
        self.exp_controller = expint.start_interface(self.cfg)
        self.exp_connected = True       
        self.num_cams = tk.StringVar()
        self.num_cams.set(str(self.cfg['CameraSettings']['num_cams']))
        self.fps = tk.StringVar()
        self.fps.set(str(self.cfg['CameraSettings']['fps']))
        self.exposure = tk.StringVar()
        self.exposure.set(str(self.cfg['CameraSettings']['exposure']))
        self.gain = tk.StringVar()
        self.gain.set(str(self.cfg['CameraSettings']['gain']))   
        self.gpi_mode = tk.StringVar()
        self.gpi_mode.set(self.cfg['CameraSettings']['gpi_mode'])
        self.trigger_source = tk.StringVar()
        self.trigger_source.set(self.cfg['CameraSettings']['trigger_source'])
        self.gpo_mode = tk.StringVar()
        self.gpo_mode.set(self.cfg['CameraSettings']['gpo_mode'])
        self.baseline_dur = tk.StringVar()
        self.baseline_dur.set(str(self.cfg['ExperimentSettings']['baseline_dur']))
        self.buffer_dur = tk.StringVar()
        self.buffer_dur.set(str(self.cfg['ExperimentSettings']['buffer_dur']))
        self.img_width = tk.StringVar()
        self.img_width.set(str(self.cfg['CameraSettings']['img_width']))
        self.img_height = tk.StringVar()
        self.img_height.set(str(self.cfg['CameraSettings']['img_height']))
        self.offset_x = tk.StringVar()
        self.offset_x.set(str(self.cfg['CameraSettings']['offset_x']))
        self.offset_y = tk.StringVar()
        self.offset_y.set(str(self.cfg['CameraSettings']['offset_y']))
        self.downsampling = tk.StringVar()
        self.downsampling.set(str(self.cfg['CameraSettings']['downsampling']))
        self.poi_threshold = tk.StringVar()
        self.poi_threshold.set(str(self.cfg['CameraSettings']['poi_threshold']))   
        #initialize housekeeping variables     
        self.cams_loaded = False
        self.streaming = False
        self.cams_connected = False
        self.draw_saved = False
        self.add_pois = False
        self.remove_pois = False
        self.added_pois = [[] for _ in range(self.cfg['CameraSettings']['num_cams'])]
        self.saved_pois = [[] for _ in range(self.cfg['CameraSettings']['num_cams'])] 
        self.capture = False
        self.record = False
        self.img_num = [1]
        self.exp_control_on = False  
        #configure window      
        self.configure_window()

    def on_quit(self):
        self.cfg['CameraSettings']['num_cams'] = int(self.num_cams.get())
        self.cfg['CameraSettings']['fps'] = int(self.fps.get())
        self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
        self.cfg['CameraSettings']['gain'] = float(self.gain.get()) 
        self.cfg['ExperimentSettings']['baseline_dur'] = float(self.baseline_dur.get())
        self.cfg['ExperimentSettings']['buffer_dur'] = float(self.buffer_dur.get())
        self.cfg['CameraSettings']['img_width'] = int(self.img_width.get())
        self.cfg['CameraSettings']['img_height'] = int(self.img_height.get())
        self.cfg['CameraSettings']['offset_x'] = int(self.offset_x.get())
        self.cfg['CameraSettings']['offset_y'] = int(self.offset_y.get())
        self.cfg['CameraSettings']['downsampling'] = self.downsampling.get()
        self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
        self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
        self.cfg['CameraSettings']['poi_threshold'] = float(self.poi_threshold.get())
        self.cfg['CameraSettings']['self.output_params'] = (self.cfg['CameraSettings']['num_cams']*
            self.cfg['CameraSettings']['img_width'],self.cfg['CameraSettings']['img_height'])
        config.save_tmp(self.cfg)
        if self.streaming:
            self.on_stream_quit()
        expint.stop_interface(self.exp_controller)
        self.destroy()

    def configure_window(self):        
        tk.Label(self,text="# Cameras:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=0, sticky='W')   
        self.num_cams_menu = tk.OptionMenu(self,self.num_cams,"1","2","3")
        self.num_cams_menu.configure(width=12,anchor="w")
        self.num_cams_menu.grid(row=0, column=1)
        tk.Label(self,text="FPS:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, sticky='W')   
        tk.Entry(self,textvariable=self.fps,width=17).grid(row=1, column=1)
        tk.Label(self,text="Exposure (usec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, sticky='W')   
        tk.Entry(self,textvariable=self.exposure,width=17).grid(row=2, column=1)
        tk.Label(self,text="Gain:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, sticky='W')   
        tk.Entry(self,textvariable=self.gain,width=17).grid(row=3, column=1)
        tk.Label(self,text="Trigger Source:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, sticky='W')   
        self.gpi_trig_menu = tk.OptionMenu(self,self.trigger_source,
            "XI_TRG_OFF",
            "XI_TRG_EDGE_RISING",
            "XI_TRG_EDGE_FALLING",
            "XI_TRG_SOFTWARE",
            "XI_TRG_LEVEL_HIGH",
            "XI_TRG_LEVEL_LOW")
        self.gpi_trig_menu.configure(width=12,anchor="w")
        self.gpi_trig_menu.grid(row=4, column=1)
        tk.Label(self,text="Sync Mode:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, sticky='W')   
        self.gpo_mode_menu = tk.OptionMenu(self,self.gpo_mode,
            "XI_GPO_OFF",
            "XI_GPO_ON",
            "XI_GPO_FRAME_ACTIVE",
            "XI_GPO_FRAME_ACTIVE_NEG",
            "XI_GPO_EXPOSURE_ACTIVE",
            "XI_GPO_EXPOSURE_ACTIVE_NEG",
            "XI_GPO_FRAME_TRIGGER_WAIT",
            "XI_GPO_FRAME_TRIGGER_WAIT_NEG",
            "XI_GPO_EXPOSURE_PULSE",
            "XI_GPO_EXPOSURE_PULSE_NEG",
            "XI_GPO_BUSY",
            "XI_GPO_BUSY_NEG",
            "XI_GPO_HIGH_IMPEDANCE",
            "XI_GPO_FRAME_BUFFER_OVERFLOW")
        self.gpo_mode_menu.configure(width=12,anchor="w")
        self.gpo_mode_menu.grid(row=5, column=1)
        tk.Label(self,text="Image Buffer (sec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, sticky='W')   
        tk.Entry(self,textvariable=self.buffer_dur,width=17).grid(row=6, column=1)
        tk.Label(self,text="Image Width (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, sticky='W')   
        tk.Entry(self,textvariable=self.img_width,width=17).grid(row=7, column=1)
        tk.Label(self,text="Image Height (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, sticky='W')   
        tk.Entry(self,textvariable=self.img_height,width=17).grid(row=8, column=1)
        tk.Label(self,text="Image X Offest (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, sticky='W')   
        tk.Entry(self,textvariable=self.offset_x,width=17).grid(row=9, column=1)
        tk.Label(self,text="Image Y Offset (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, sticky='W')   
        tk.Entry(self,textvariable=self.offset_y,width=17).grid(row=10, column=1)
        tk.Label(self,text="Downsampling:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, sticky='W') 
        self.downsampling_menu = tk.OptionMenu(self,self.downsampling,
            "XI_DWN_1x1",
            "XI_DWN_2x2")
        self.downsampling_menu.configure(width=12,anchor="w")
        self.downsampling_menu.grid(row=11, column=1)
        tk.Button(self,text="Start Streaming",font='Arial 10 bold',width=14,command=self.start_stream_callback).grid(row=12, column=0,sticky="e")
        tk.Button(self,text="Stop Streaming",font='Arial 10 bold',width=14,command=self.stop_stream_callback).grid(row=13, column=0,sticky="e")
        tk.Button(self,text="Load POIs",font='Arial 10 bold',width=14,command=self.load_pois_callback).grid(row=12, column=1)
        tk.Button(self,text="Save POIs",font='Arial 10 bold',width=14,command=self.save_pois_callback).grid(row=13, column=1)
        tk.Button(self,text="Add POIs",font='Arial 10 bold',width=14,command=self.add_pois_callback).grid(row=12, column=2)
        tk.Button(self,text="Remove POIs",font='Arial 10 bold',width=14,command=self.remove_pois_callback).grid(row=13, column=2)
        tk.Button(self,text="Capture Image",font='Arial 10 bold',width=14,command=self.capture_image_callback).grid(row=14, column=0,sticky="e")
        tk.Button(self,text="Start Record",font='Arial 10 bold',width=14,command=self.start_rec_callback).grid(row=14, column=1)
        tk.Button(self,text="Stop Record",font='Arial 10 bold',width=14,command=self.stop_rec_callback).grid(row=14, column=2)        
        tk.Label(self,text="POI Threshold (stdev):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, sticky='W')   
        tk.Entry(self,textvariable=self.poi_threshold,width=17).grid(row=15, column=1)
        tk.Button(self,text="Toggle Lights", font='Arial 10 bold',width=14, command=self.toggle_lights_callback).grid(row=16, column=1)

    def start_stream_callback(self):
        if not self.cams_connected:
            self.cfg['CameraSettings']['num_cams'] = int(self.num_cams.get())
            self.cfg['CameraSettings']['fps'] = int(self.fps.get())
            self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
            self.cfg['CameraSettings']['gain'] = float(self.gain.get())   
            self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.cfg['CameraSettings']['img_width'] = int(self.img_width.get())
            self.cfg['CameraSettings']['img_height'] = int(self.img_height.get())
            self.cfg['CameraSettings']['offset_x'] = int(self.offset_x.get())
            self.cfg['CameraSettings']['offset_y'] = int(self.offset_y.get())  
            self.cfg['CameraSettings']['downsampling'] = self.downsampling.get()
            self.cams = camint.start_interface(self.cfg)
            self.cams_connected = True 
            self.img = camint.init_image()                                     
        if not self.streaming:
            self.start_stream()
        else: 
            tkMessageBox.showinfo("Warning", "Already streaming.") 

    def stop_stream_callback(self):
        self.streaming = False       

    def start_stream(self):
        self.cam_windows = [0 for _ in range(self.cfg['CameraSettings']['num_cams'])]
        for i in range(self.cfg['CameraSettings']['num_cams']):
            self.cam_windows[i] = tk.Toplevel(self)
            self.cam_windows[i].title("Camera"+str(i))
            self.cam_windows[i].protocol("WM_DELETE_WINDOW", self.on_stream_quit)
            self.cam_windows[i].canvas = tk.Canvas(self.cam_windows[i], 
                width = self.cfg['CameraSettings']['img_width'], 
                height = self.cfg['CameraSettings']['img_height'])
            self.cam_windows[i].canvas.grid(row=0,column= 0)            
        self.delay = int(np.round(1.0/float(self.cfg['CameraSettings']['fps'])*1000.0))
        self.photo_img = [0 for _ in range(self.cfg['CameraSettings']['num_cams'])]
        self.streaming = True
        self.refresh()

    def on_stream_quit(self):
        self.streaming = False          
        self.poi_active = False  
        self.draw_saved = False    
        for i in range(self.cfg['CameraSettings']['num_cams']):
            self.cam_windows[i].destroy()
        camint.stop_interface(self.cams)
        self.cams_connected = False

    def refresh(self):
        if self.streaming:
            expint.trigger_image(self.exp_controller)
            now = str(int(round(time.time()*1000)))            
            for i in range(self.cfg['CameraSettings']['num_cams']):
                #display image
                npimg = camint.get_npimage(self.cams[i],self.img)
                npimg = cv2.cvtColor(npimg,cv2.COLOR_BAYER_BG2RGB)
                self.photo_img[i] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(npimg))
                self.cam_windows[i].canvas.create_image(0,0, image = self.photo_img[i], anchor = tk.NW)                
                #draw saved pixels (green)
                if self.draw_saved:
                    for poi in self.saved_pois[i]:                         
                        self.cam_windows[i].canvas.create_line(poi[0],poi[1],poi[0]+1,poi[1],width=1,fill='green')
                #draw currently addded pixels (red)
                for poi in self.added_pois[i]:                        
                    self.cam_windows[i].canvas.create_line(poi[0],poi[1],poi[0]+1,poi[1],width=1,fill='red')
                #draw cursor for adding/removing pois
                if self.add_pois or self.remove_pois:
                    self.draw_cursor(i)
                    self.cam_windows[i].bind('<Button-1>',lambda event,camid=i:self.draw_poi(event,camid))
                #prepare frame for possible capture
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))
            if self.capture:
                self.calibration_path = self.cfg['ReachMaster']['data_dir'] + "/calibration_images/"
                if not os.path.isdir(self.calibration_path):
                    os.makedirs(self.calibration_path)
                fn = "image" + str(self.img_num[0])
                cv2.imwrite('%s/%s.png' % (self.calibration_path, fn), frame)
                self.capture = False
                self.img_num[0] += 1
            self.after(self.delay,self.refresh)

    def load_pois_callback(self):
        if self.streaming:
            if len(self.cfg['CameraSettings']['saved_pois'])>0:
                self.saved_pois = self.cfg['CameraSettings']['saved_pois']
                self.draw_saved = True
            else:
                tkMessageBox.showinfo("Warning", "No saved POIs.")
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to load POIs.")

    def add_pois_callback(self):
        if self.streaming:
            self.add_pois = True
            self.remove_pois = False
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to add POIs.") 

    def remove_pois_callback(self):
        if self.streaming:
            if (len(self.added_pois)+len(self.saved_pois))>0:
                self.add_pois = False
                self.remove_pois = True
            else:
                tkMessageBox.showinfo("Warning", "No POIs to remove.")
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to remove POIs.")

    def save_pois_callback(self):
        for i in range(self.cfg['CameraSettings']['num_cams']):
            self.saved_pois[i] += self.added_pois[i] 
        self.cfg['CameraSettings']['saved_pois'] = self.saved_pois 
        self.added_pois = [[] for _ in range(self.cfg['CameraSettings']['num_cams'])]

    def draw_cursor(self,i):
        self.cam_windows[i].bind('<Motion>', self.cam_windows[i].config(cursor = "cross"))        

    def draw_poi(self, event, camid):
        if self.add_pois:
            self.added_pois[camid].append([event.x,event.y])  
        elif self.remove_pois:
            if len(self.saved_pois[camid])>0:
                tmp = []
                for poi in self.saved_pois[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.saved_pois[camid] = tmp
            if len(self.added_pois[camid])>0:
                tmp = []
                for poi in self.added_pois[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.added_pois[camid] = tmp

    def capture_image_callback(self):
        if self.streaming:
            self.capture = True
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to capture images.")

    def start_rec_callback(self):
        if not self.streaming:
            self.cfg['CameraSettings']['num_cams'] = int(self.num_cams.get())
            self.cfg['CameraSettings']['fps'] = int(self.fps.get())
            self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
            self.cfg['CameraSettings']['gain'] = float(self.gain.get())   
            self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.cfg['CameraSettings']['img_width'] = int(self.img_width.get())
            self.cfg['CameraSettings']['img_height'] = int(self.img_height.get())
            self.cfg['CameraSettings']['offset_x'] = int(self.offset_x.get())
            self.cfg['CameraSettings']['offset_y'] = int(self.offset_y.get())
            self.output_params = (self.cfg['CameraSettings']['num_cams']*
            self.cfg['CameraSettings']['img_width'],self.cfg['CameraSettings']['img_height']) 
            self.cfg['CameraSettings']['output_params'] = self.output_params            
            self.cams = camint.start_interface(self.cfg)
            self.cams_connected = True
            self.img = camint.init_image()
            self.calibration_path = self.cfg['ReachMaster']['data_dir'] + "/calibration_videos/"
            if not os.path.isdir(self.calibration_path):
                os.makedirs(self.calibration_path)
            self.vid_fn = self.calibration_path + str(datetime.datetime.now()) + '.mp4'             
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **self.output_params)
            self.delay = int(np.round(1.0/float(self.cfg['CameraSettings']['fps'])*1000.0))
            self.record = True
            self.rec()
        else: 
            tkMessageBox.showinfo("Warning", "Shouldn't record while streaming. Bad framerates!")

    def stop_rec_callback(self):
        self.record = False
        self.video.close()
        camint.stop_interface(self.cams)
        self.cams_connected = False

    def rec(self):
        if self.record:
            expint.trigger_image(self.exp_controller)
            now = str(int(round(time.time()*1000)))          
            for i in range(self.cfg['CameraSettings']['num_cams']):
                npimg = camint.get_npimage(self.cams[i],self.img)
                # npimg = cv2.cvtColor(npimg,cv2.COLOR_BAYER_BG2RGB)
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))               
            self.video.write(frame)
            self.after(self.delay,self.rec)

    def toggle_lights_callback(self):
        if self.exp_connected:
            expint.toggle_lights(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")