from .. import config
import Tkinter as tk 
import tkFileDialog
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
import json

class CameraSettings(tk.Toplevel):

    def __init__(self, parent):
    	tk.Toplevel.__init__(self, parent)
    	self.transient(parent) 
    	self.grab_set()
        self.title("Camera Settings")
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.onQuit)  
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
        self.output_params = self.cfg['CameraSettings']['output_params']
        self.numCams = tk.StringVar()
        self.numCams.set(str(self.cfg['CameraSettings']['numCams']))
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
        self.baselineDur = tk.StringVar()
        self.baselineDur.set(str(self.cfg['ExperimentSettings']['baselineDur']))
        self.bufferDur = tk.StringVar()
        self.bufferDur.set(str(self.cfg['ExperimentSettings']['bufferDur']))
        self.imgWidth = tk.StringVar()
        self.imgWidth.set(str(self.cfg['CameraSettings']['imgWidth']))
        self.imgHeight = tk.StringVar()
        self.imgHeight.set(str(self.cfg['CameraSettings']['imgHeight']))
        self.offsetX = tk.StringVar()
        self.offsetX.set(str(self.cfg['CameraSettings']['offsetX']))
        self.offsetY = tk.StringVar()
        self.offsetY.set(str(self.cfg['CameraSettings']['offsetY']))
        self.downsampling = tk.StringVar()
        self.downsampling.set(str(self.cfg['CameraSettings']['downsampling']))
        self.poiThreshold = tk.StringVar()
        self.poiThreshold.set(str(self.cfg['CameraSettings']['poiThreshold']))
        self.camsLoaded = False
        self.streaming = False
        self.streamStarted = False
        self.drawSaved = False
        self.addPOIs = False
        self.removePOIs = False
        self.addedPOIs = [[] for _ in range(self.cfg['CameraSettings']['numCams'])]
        self.savedPOIs = [[] for _ in range(self.cfg['CameraSettings']['numCams'])] 
        self.capture = False
        self.record = False
        self.imgNum = [1]
        self.vidMode = tk.StringVar()
        self.vidMode.set(self.cfg['CameraSettings']['vidMode'])
        self.expControlOn = False
        self.expConnect()
        self.setup_UI()

    def onQuit(self):
        self.cfg['CameraSettings']['numCams'] = int(self.numCams.get())
        self.cfg['CameraSettings']['fps'] = int(self.fps.get())
        self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
        self.cfg['CameraSettings']['gain'] = float(self.gain.get()) 
        self.cfg['ExperimentSettings']['baselineDur'] = float(self.baselineDur.get())
        self.cfg['ExperimentSettings']['bufferDur'] = float(self.bufferDur.get())
        self.cfg['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
        self.cfg['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
        self.cfg['CameraSettings']['offsetX'] = int(self.offsetX.get())
        self.cfg['CameraSettings']['offsetY'] = int(self.offsetY.get())
        self.cfg['CameraSettings']['downsampling'] = self.downsampling.get()
        self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
        self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
        self.cfg['CameraSettings']['poiThreshold'] = float(self.poiThreshold.get())
        self.output_params["-output_dimensions"] = (self.cfg['CameraSettings']['numCams']*
            self.cfg['CameraSettings']['imgWidth'],self.cfg['CameraSettings']['imgHeight'])
        self.cfg['CameraSettings']['vidMode'] = self.vidMode.get()
        if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS":
            self.output_params["-crf"] = 28
        elif self.cfg['CameraSettings']['vidMode'] == "TRIALS":
            self.output_params["-crf"] = 23
        self.cfg['CameraSettings']['self.output_params'] = self.output_params
        config.save_tmp(self.cfg)
        if self.streaming:
            self.stopStream()
        self.expDisconnect()
        self.destroy()

    def expConnect(self):
        global expController        
        expController = serial.Serial(self.cfg['ReachMaster']['expControlPath'],
            self.cfg['ReachMaster']['serialBaud'],
            timeout=self.cfg['ReachMaster']['controlTimeout'])
        # time.sleep(2) #wait for controller to wake up
        expController.flushInput()
        expController.write("h")
        response = expController.read()
        if response=="h":
            self.expControlOn = True
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def expDisconnect(self):
        if self.expControlOn:
            expController.write("e")
            expController.close()
        else:
            pass

    def setup_UI(self):        
        tk.Label(self,text="# Cameras:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=0, sticky='W')   
        self.numCamsMenu = tk.OptionMenu(self,self.numCams,"1","2","3")
        self.numCamsMenu.configure(width=12,anchor="w")
        self.numCamsMenu.grid(row=0, column=1)
        tk.Label(self,text="FPS:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, sticky='W')   
        tk.Entry(self,textvariable=self.fps,width=17).grid(row=1, column=1)
        tk.Label(self,text="Exposure (usec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, sticky='W')   
        tk.Entry(self,textvariable=self.exposure,width=17).grid(row=2, column=1)
        tk.Label(self,text="Gain:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, sticky='W')   
        tk.Entry(self,textvariable=self.gain,width=17).grid(row=3, column=1)
        tk.Label(self,text="Trigger Source:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, sticky='W')   
        self.gpiTrigMenu = tk.OptionMenu(self,self.trigger_source,
            "XI_TRG_OFF",
            "XI_TRG_EDGE_RISING",
            "XI_TRG_EDGE_FALLING",
            "XI_TRG_SOFTWARE",
            "XI_TRG_LEVEL_HIGH",
            "XI_TRG_LEVEL_LOW")
        self.gpiTrigMenu.configure(width=12,anchor="w")
        self.gpiTrigMenu.grid(row=4, column=1)
        tk.Label(self,text="Sync Mode:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, sticky='W')   
        self.gpoModeMenu = tk.OptionMenu(self,self.gpo_mode,
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
        self.gpoModeMenu.configure(width=12,anchor="w")
        self.gpoModeMenu.grid(row=5, column=1)
        tk.Label(self,text="Image Buffer (sec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, sticky='W')   
        tk.Entry(self,textvariable=self.bufferDur,width=17).grid(row=6, column=1)
        tk.Label(self,text="Image Width (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, sticky='W')   
        tk.Entry(self,textvariable=self.imgWidth,width=17).grid(row=7, column=1)
        tk.Label(self,text="Image Height (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, sticky='W')   
        tk.Entry(self,textvariable=self.imgHeight,width=17).grid(row=8, column=1)
        tk.Label(self,text="Image X Offest (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, sticky='W')   
        tk.Entry(self,textvariable=self.offsetX,width=17).grid(row=9, column=1)
        tk.Label(self,text="Image Y Offset (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, sticky='W')   
        tk.Entry(self,textvariable=self.offsetY,width=17).grid(row=10, column=1)
        tk.Label(self,text="Downsampling:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, sticky='W') 
        self.downsamplingMenu = tk.OptionMenu(self,self.downsampling,
            "XI_DWN_1x1",
            "XI_DWN_2x2")
        self.downsamplingMenu.configure(width=12,anchor="w")
        self.downsamplingMenu.grid(row=11, column=1)
        tk.Button(self,text="Start Streaming",font='Arial 10 bold',width=14,command=self.startStreamCallback).grid(row=12, column=0,sticky="e")
        tk.Button(self,text="Stop Streaming",font='Arial 10 bold',width=14,command=self.stopStreamCallback).grid(row=13, column=0,sticky="e")
        tk.Button(self,text="Load POIs",font='Arial 10 bold',width=14,command=self.loadPOIsCallback).grid(row=12, column=1)
        tk.Button(self,text="Save POIs",font='Arial 10 bold',width=14,command=self.savePOIsCallback).grid(row=13, column=1)
        tk.Button(self,text="Add POIs",font='Arial 10 bold',width=14,command=self.addPOIsCallback).grid(row=12, column=2)
        tk.Button(self,text="Remove POIs",font='Arial 10 bold',width=14,command=self.removePOIsCallback).grid(row=13, column=2)
        tk.Button(self,text="Capture Image",font='Arial 10 bold',width=14,command=self.captureImgCallback).grid(row=14, column=0,sticky="e")
        tk.Button(self,text="Start Record",font='Arial 10 bold',width=14,command=self.startRecCallback).grid(row=14, column=1)
        tk.Button(self,text="Stop Record",font='Arial 10 bold',width=14,command=self.stopRecCallback).grid(row=14, column=2)        
        tk.Label(self,text="POI Threshold (stdev):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, sticky='W')   
        tk.Entry(self,textvariable=self.poiThreshold,width=17).grid(row=15, column=1)
        tk.Label(self,text="Video Mode:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=16, sticky='W')   
        self.vidModeMenu = tk.OptionMenu(self,self.vidMode,
            "CONTINUOUS",
            "TRIALS")
        self.vidModeMenu.configure(width=12,anchor="w")
        self.vidModeMenu.grid(row=16, column=1)

    def startStreamCallback(self):
        if not self.streamStarted:
            self.cfg['CameraSettings']['numCams'] = int(self.numCams.get())
            self.cfg['CameraSettings']['fps'] = int(self.fps.get())
            self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
            self.cfg['CameraSettings']['gain'] = float(self.gain.get())   
            self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.cfg['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
            self.cfg['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
            self.cfg['CameraSettings']['offsetX'] = int(self.offsetX.get())
            self.cfg['CameraSettings']['offsetY'] = int(self.offsetY.get())  
            self.cfg['CameraSettings']['downsampling'] = self.downsampling.get()
            self.loadCameras() 
            self.startStream()
        elif not self.streaming:
            self.startStream()
        else: 
            tkMessageBox.showinfo("Warning", "Already streaming.") 

    def stopStreamCallback(self):
        self.streaming = False 

    def loadCameras(self):        
        self.camList = []
        for i in range(self.cfg['CameraSettings']['numCams']):
            print('opening camera %s ...' %(i))
            cam = xiapi.Camera(dev_id = i)
            cam.open_device()
            cam.set_imgdataformat(self.cfg['CameraSettings']['imgdataformat'])
            cam.set_exposure(self.cfg['CameraSettings']['exposure'])
            cam.set_gain(self.cfg['CameraSettings']['gain'])
            cam.set_sensor_feature_value(self.cfg['CameraSettings']['sensor_feature_value'])
            cam.set_gpi_selector(self.cfg['CameraSettings']['gpi_selector'])
            # cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            # cam.set_framerate(fps)
            cam.set_gpi_mode("XI_GPI_TRIGGER")
            cam.set_trigger_source(self.cfg['CameraSettings']['trigger_source'])
            cam.set_gpo_selector(self.cfg['CameraSettings']['gpo_selector'])
            cam.set_gpo_mode(self.cfg['CameraSettings']['gpo_mode'])
            if self.cfg['CameraSettings']['downsampling'] == "XI_DWN_2x2":
                cam.set_downsampling(self.cfg['CameraSettings']['downsampling'])
            else:
                widthIncrement = cam.get_width_increment()
                heightIncrement = cam.get_height_increment()
                if (self.cfg['CameraSettings']['imgWidth']%widthIncrement)!=0:
                    tkMessageBox.showinfo("Warning", "Image width not divisible by "+str(widthIncrement))
                    break
                elif (self.cfg['CameraSettings']['imgHeight']%heightIncrement)!=0:
                    tkMessageBox.showinfo("Warning", "Image height not divisible by "+str(heightIncrement))
                    break
                elif (self.cfg['CameraSettings']['imgWidth']+self.cfg['CameraSettings']['offsetX'])>1280:
                    tkMessageBox.showinfo("Warning", "Image width + x offset > 1280") 
                    break
                elif (self.cfg['CameraSettings']['imgHeight']+self.cfg['CameraSettings']['offsetY'])>1024:
                    tkMessageBox.showinfo("Warning", "Image height + y offset > 1024") 
                    break
                else:
                    cam.set_height(self.cfg['CameraSettings']['imgHeight'])
                    cam.set_width(self.cfg['CameraSettings']['imgWidth'])
                    cam.set_offsetX(self.cfg['CameraSettings']['offsetX'])
                    cam.set_offsetY(self.cfg['CameraSettings']['offsetY'])                
            self.camList.append(cam)
            self.camList[i].start_acquisition()   
        self.imgBuffer = deque()
        self.camsLoaded = True        

    def unloadCameras(self):
        for i in range(self.cfg['CameraSettings']['numCams']):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camsLoaded = False

    def startStream(self):
        if not self.streamStarted:
            self.buffer_full = False 
            self.camWindows = [0 for _ in range(self.cfg['CameraSettings']['numCams'])]
            for i in range(self.cfg['CameraSettings']['numCams']):
                self.camWindows[i] = tk.Toplevel(self)
                self.camWindows[i].title("Camera"+str(i))
                self.camWindows[i].protocol("WM_DELETE_WINDOW", self.stopStream)
                self.camWindows[i].canvas = tk.Canvas(self.camWindows[i], 
                    width = self.cfg['CameraSettings']['imgWidth'], 
                    height = self.cfg['CameraSettings']['imgHeight'])
                self.camWindows[i].canvas.grid(row=0,column= 0)            
            self.streamStarted = True
        self.delay = int(np.round(1.0/float(self.cfg['CameraSettings']['fps'])*1000.0))
        self.streaming = True
        self.refresh()

    def stopStream(self):
        self.streaming = False 
        self.streamStarted = False 
        self.poiActive = False  
        self.drawSaved = False    
        for i in range(self.cfg['CameraSettings']['numCams']):
            self.camWindows[i].destroy()
        self.unloadCameras()

    def refresh(self):
        if self.streaming:
            expController.write("t")
            now = str(int(round(time.time()*1000)))
            npImg = np.zeros(shape = (self.cfg['CameraSettings']['imgHeight'], self.cfg['CameraSettings']['imgWidth'])) 
            img = xiapi.Image()
            self.photoImg = [0 for _ in range(self.cfg['CameraSettings']['numCams'])]
            for i in range(self.cfg['CameraSettings']['numCams']):
                # print(i)
                self.camList[i].get_image(img,timeout = 2000)
                npImg = img.get_image_data_numpy()
                npImg = cv2.cvtColor(npImg,cv2.COLOR_BAYER_BG2RGB)
                if i == 0:
                    frame = npImg
                else:
                    frame = np.hstack((frame,npImg))
                self.photoImg[i] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(npImg))
                self.camWindows[i].canvas.create_image(0,0, image = self.photoImg[i], anchor = tk.NW)
                if self.drawSaved:
                    for poi in self.savedPOIs[i]:                         
                        self.camWindows[i].canvas.create_line(poi[0],poi[1],poi[0]+1,poi[1],width=1,fill='green')
                if self.addPOIs or self.removePOIs:
                    self.drawCursor(i)
                    self.camWindows[i].bind('<Button-1>',lambda event,camid=i:self.drawPOI(event,camid))
                    for poi in self.addedPOIs[i]:                        
                        self.camWindows[i].canvas.create_line(poi[0],poi[1],poi[0]+1,poi[1],width=1,fill='red')
            if self.capture:
                self.calipath = self.cfg['ReachMaster']['dataDir'] + "/calibration_images/"
                if not os.path.isdir(self.calipath):
                    os.makedirs(self.calipath)
                fn = "image" + str(self.imgNum[0])
                cv2.imwrite('%s/%s.png' % (self.calipath, fn), frame)
                self.imgBuffer = deque()
                self.capture = False
                self.imgNum[0] += 1
            self.window.after(self.delay,self.refresh)

    def loadPOIsCallback(self):
        if self.streaming:
            if len(self.cfg['CameraSettings']['savedPOIs'])>0:
                self.savedPOIs = self.cfg['CameraSettings']['savedPOIs']
                self.drawSaved = True
            else:
                tkMessageBox.showinfo("Warning", "No saved POIs.")
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to load POIs.")

    def addPOIsCallback(self):
        if self.streaming:
            self.addPOIs = True
            self.removePOIs = False
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to add POIs.") 

    def removePOIsCallback(self):
        if self.streaming:
            if (len(self.addedPOIs)+len(self.savedPOIs))>0:
                self.addPOIs = False
                self.removePOIs = True
            else:
                tkMessageBox.showinfo("Warning", "No POIs to remove.")
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to remove POIs.")

    def savePOIsCallback(self):
        global baselineAcquired
        baselineAcquired = False
        for i in range(self.cfg['CameraSettings']['numCams']):
            self.savedPOIs[i] += self.addedPOIs[i] 
        self.cfg['CameraSettings']['savedPOIs'] = self.savedPOIs 
        self.addedPOIs = [[] for _ in range(self.cfg['CameraSettings']['numCams'])]

    def drawCursor(self,i):
        self.camWindows[i].bind('<Motion>', self.camWindows[i].config(cursor = "cross"))        

    def drawPOI(self, event, camid):
        if self.addPOIs:
            self.addedPOIs[camid].append([event.x,event.y])  
        elif self.removePOIs:
            if len(self.savedPOIs[camid])>0:
                tmp = []
                for poi in self.savedPOIs[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.savedPOIs[camid] = tmp
            if len(self.addedPOIs[camid])>0:
                tmp = []
                for poi in self.addedPOIs[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.addedPOIs[camid] = tmp

    def captureImgCallback(self):
        if self.streaming:
            self.capture = True
        else: 
            tkMessageBox.showinfo("Warning", "Must be streaming to capture images.")

    def startRecCallback(self):
        if not self.streaming:
            self.cfg['CameraSettings']['numCams'] = int(self.numCams.get())
            self.cfg['CameraSettings']['fps'] = int(self.fps.get())
            self.cfg['CameraSettings']['exposure'] = int(self.exposure.get())
            self.cfg['CameraSettings']['gain'] = float(self.gain.get())   
            self.cfg['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.cfg['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.cfg['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
            self.cfg['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
            self.cfg['CameraSettings']['offsetX'] = int(self.offsetX.get())
            self.cfg['CameraSettings']['offsetY'] = int(self.offsetY.get())  
            self.record = True
            self.loadCameras() 
            self.calipath = self.cfg['ReachMaster']['dataDir'] + "/calibration_videos/"
            if not os.path.isdir(self.calipath):
                os.makedirs(self.calipath)
            self.vid_fn = self.calipath + str(datetime.datetime.now()) + '.mp4' 
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **self.output_params)
            self.delay = int(np.round(1.0/float(self.cfg['CameraSettings']['fps'])*1000.0))
            self.rec()
        else: 
            tkMessageBox.showinfo("Warning", "Shouldn't record while streaming. Bad framerates!")

    def stopRecCallback(self):
        self.record = False
        self.video.close()
        self.unloadCameras()

    def rec(self):
        if self.record:
            expController.write("t")
            now = str(int(round(time.time()*1000)))
            npImg = np.zeros(shape = (self.cfg['CameraSettings']['imgHeight'], self.cfg['CameraSettings']['imgWidth'])) 
            img = xiapi.Image()            
            for i in range(self.cfg['CameraSettings']['numCams']):
                self.camList[i].get_image(img,timeout = 2000)
                npImg = img.get_image_data_numpy()
                # npImg = cv2.cvtColor(npImg,cv2.COLOR_BAYER_BG2RGB)
                if i == 0:
                    frame = npImg
                else:
                    frame = np.hstack((frame,npImg))               
            self.video.write(frame)
            self.window.after(self.delay,self.rec)
        
    def contModeWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("contMode")
            if expController.readline() == "v":
                expController.write("1")