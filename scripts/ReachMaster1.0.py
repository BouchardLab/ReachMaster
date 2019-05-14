import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import threading 
import cv2
import PIL.Image, PIL.ImageTk
# import cam_func.camera_start_acq as csa
from ximea import xiapi
import reach_master.camera_utilities.ImageTuple as imgTup
import reach_master.camera_utilities.serializeBuffer as serBuf
import time
import datetime
import numpy as np
import serial
from serial.tools import list_ports
import binascii
import struct 
import os 
from collections import deque

#declare global variables and set to defaults
dataDir = os.getcwd()                       #directory to save data
paramFile = 'None'                          #file with experiment parameters
#camera settings 
numCams = 2
imgdataformat = "XI_RAW8"                   #raw camera output format (note: use XI_RAW8 for full fps) 
fps = 200
exposure = 3000                             #exposure time (microseconds) (note: determines minimum trigger period) 
gain = 0.0                                  #gain: sensitivity of camera 
sensor_feature_value = 1
gpi_selector = "XI_GPI_PORT1" 
gpi_mode =  "XI_GPI_TRIGGER"                
trigger_source = "XI_TRG_EDGE_RISING"       
gpo_selector = "XI_GPO_PORT1"
gpo_mode = "XI_GPO_EXPOSURE_ACTIVE"
baselineDur = 5.0
bufferDur = 0.5                            #duration (sec) to buffer images
reachTimeout = 4000
savedPOIs = []
poiMeans = []
poiStds = []
obsPOIs = []
zPOIs = []
poiThreshold = 10
baselineAcquired = False
#controller settings
expControlPath = '/dev/ttyACM0'
robControlPath = '/dev/ttyACM1'
serialBaud = 2000000
controlTimeout = 5
#experiment settings
flushDur = 10000              #time to keep solenoid open during initial water flush
solenoidOpenDur = 75          #time to keep solenoid open during single reward delivery
solenoidBounceDur = 500       #time between reward deliveries
rewardWinDur = 3000           #duration (ms) for which rewards could be delivered
maxRewards = 3                #maximum number of rewards that should be delivered in a trial
lightsOffDur = 3000           #minimum time to keep lights off in between trials
lightsOnDur = 5000            #maximum time to keep tights on during a trial
#robot settings
lightsOn = 0

class ReachMaster:

    def __init__(self, window):
        #setup root UI
        self.window = window
        self.window.title("ReachMaster 1.0")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.mainActive = False
        self.dataDir = tk.StringVar()
        self.dataDir.set(dataDir) 
        self.paramFile = tk.StringVar()
        self.paramFile.set(paramFile)
        self.portList = list(list_ports.comports())
        for i in range(len(self.portList)):
            self.portList[i] = self.portList[i].device
        self.expControlPath = tk.StringVar()
        self.robControlPath = tk.StringVar()
        if expControlPath in self.portList:
            self.expControlPath.set(expControlPath) 
        else:
            self.expControlPath.set(self.portList[0])
        if robControlPath in self.portList:
            self.robControlPath.set(robControlPath) 
        else:
            self.robControlPath.set(self.portList[0])
        self.expControlOn = False
        self.robControlOn = False
        self.camerasLoaded = False
        self.expBegan = False
        self.expActive = False      
        self.expEnded = False    
        self.setup_UI()
        #run main program
        self.main()

    def onQuit(self):
        answer = tkMessageBox.askyesnocancel("Question", "Save parameters?")
        if answer == True:
            self.saveParameters()
            self.mainActive = False
        elif answer == False:
            self.mainActive = False
        else:
            pass

    def saveParameters(self):
        print("Parameters saved!")

    def setup_UI(self):
        tk.Label(text="Data Directory:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=0, column=0)
        tk.Label(textvariable=self.dataDir, bg="white").grid(row=0, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.ddBrowseCallback).grid(row=0, column=2)
        tk.Label(text="Parameter File:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=1, column=0)
        tk.Label(textvariable=self.paramFile, bg="white").grid(row=1, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.pfBrowseCallback).grid(row=1, column=2)
        tk.Label(text="Experiment Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=2, column=0)
        self.expControllerMenu = apply(tk.OptionMenu, (self.window, self.expControlPath) + tuple(self.portList))
        self.expControllerMenu.grid(row=2, column=1)
        tk.Button(text="Connect", font='Arial 10 bold',width=14, command=self.expConnectCallback).grid(row=2, column=2)
        tk.Button(text="Disconnect", font='Arial 10 bold',width=14, command=self.expDisconnectCallback).grid(row=2, column=3)
        tk.Label(text="Robot Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=3, column=0)
        self.robControllerMenu = apply(tk.OptionMenu, (self.window, self.robControlPath) + tuple(self.portList))
        self.robControllerMenu.grid(row=3, column=1)
        tk.Button(text="Connect", font='Arial 10 bold',width=14, command=self.robConnectCallback).grid(row=3, column=2)
        tk.Button(text="Disconnect", font='Arial 10 bold',width=14, command=self.robDisconnectCallback).grid(row=3, column=3)
        tk.Button(text="Camera Settings", font='Arial 10 bold',width=16, command=self.camSetCallback).grid(row=4, sticky='W')
        tk.Button(text="Experiment Settings", font='Arial 10 bold',width=16, command=self.expSetCallback).grid(row=5, sticky='W')
        tk.Button(text="Robot Settings", font='Arial 10 bold',width=16, command=self.robSetCallback).grid(row=6, sticky='W')
        tk.Button(text="Move Robot", font='Arial 10 bold',width=16, command=self.movRobCallback).grid(row=7, sticky='W')
        tk.Button(text="Toggle LED", font='Arial 10 bold',width=14, command=self.ledCallback).grid(row=4, column=1)
        tk.Button(text="Toggle Lights", font='Arial 10 bold',width=14, command=self.lightsCallback).grid(row=5, column=1)
        tk.Button(text="Deliver Water", font='Arial 10 bold',width=14, command=self.waterCallback).grid(row=6, column=1)
        tk.Button(text="Flush Water", font='Arial 10 bold',width=14, command=self.flushCallback).grid(row=7, column=1)
        tk.Button(text="Begin Experiment", font='Arial 10 bold',width=14, command=self.beginExpCallback).grid(row=4, column=2)
        tk.Button(text="Pause Experiment", font='Arial 10 bold',width=14, command=self.pauseExpCallback).grid(row=5, column=2)
        tk.Button(text="End Experiment", font='Arial 10 bold',width=14, command=self.endExpCallback).grid(row=6, column=2)

    def ddBrowseCallback(self):
        self.dataDir.set(tkFileDialog.askdirectory())

    def pfBrowseCallback(self):
        self.paramFile.set(tkFileDialog.askopenfilename())
        #open file and read in parameters

    def expConnectCallback(self):
        global expControlPath
        expControlPath = self.expControlPath.get()
        self.expController = serial.Serial(expControlPath,serialBaud,timeout=controlTimeout)
        time.sleep(2) #wait for controller to wake up
        self.expController.flushInput()
        self.expController.write("h")
        response = self.expController.readline()
        if response:
            self.expControlOn = True
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def expDisconnectCallback(self):
        if self.expControlOn:
            if not self.expActive:
                self.expController.write("e")
                self.expController.close()
            else:
                tkMessageBox.showinfo("Warning", "Experiment is active! Not safe to disconnect.")
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def robConnectCallback(self):
        global robControlPath
        robControlPath = self.robControlPath.get()
        self.robController = serial.Serial(robControlPath,serialBaud,timeout=controlTimeout)
        time.sleep(2) #wait for controller to wake up
        self.robController.flushInput()
        self.robController.write("h")
        response = self.robController.readline()
        if response:
            self.robControlOn = True
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def robDisconnectCallback(self):
        if self.robControlOn:
            self.robController.write("e")
            self.robController.close()
        else:
            tkMessageBox.showinfo("Warning", "Robot controller not connected.")

    def camSetCallback(self):  
        if not self.expActive:
            camSetRoot = tk.Toplevel(self.window)     
            CameraSettings(camSetRoot)
        else:
            pass

    def expSetCallback(self):
        if self.expControlOn:
            expSetRoot = tk.Toplevel(self.window)     
            ExperimentSettings(expSetRoot, self.expController)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def robSetCallback(self):
        if self.robControlOn:
            robSetRoot = tk.Toplevel(self.window)     
            RobotSettings(robSetRoot, robController)
        else:
            tkMessageBox.showinfo("Warning", "Robot controller not connected.")

    def movRobCallback(self):
        global lightsOn
        if self.expControlOn:
            self.expController.write("m")
            lightsOn = 0
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def ledCallback(self):
        if self.expControlOn:
            self.expController.write("l")          
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def lightsCallback(self):
        global lightsOn
        if self.expControlOn:
            self.expController.write("n")
            lightsOn = not lightsOn
            # if self.expActive and lightsOn:
            #     self.newline[2] = '1'
            # elif self.expActive:
            #     self.newline[2] = '0'
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def waterCallback(self):
        if self.expControlOn:
            self.expController.write("w")
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def flushCallback(self):
        if self.expControlOn:
            if not self.expActive:
                self.expController.write("f")
            else:
                pass
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def beginExpCallback(self):
        if self.expControlOn:
            if len(savedPOIs)>0:
                if not self.expBegan:
                    try:
                        global lightsOn
                        self.loadCameras()
                        if not baselineAcquired:
                            if not lightsOn:
                                self.lightsCallback()
                                lightsOn = 1
                            self.writeControl = "s"
                            self.acquireBaseline()                       
                        global obsPOIs
                        global zPOIs
                        obsPOIs = []
                        zPOIs = []
                        for i in range(numCams):
                            obsPOIs.append(np.zeros(len(savedPOIs[i])))
                            zPOIs.append(0)
                        self.buffer_full = False 
                        self.sensorpath = dataDir + "/sensor_data/"
                        self.camerapath = dataDir + "/camera/data/" + str(datetime.datetime.now())
                        if not os.path.isdir(self.sensorpath):
                            os.makedirs(self.sensorpath)
                        if not os.path.isdir(self.camerapath):
                            os.makedirs(self.camerapath)
                        sensorfile = self.sensorpath + str(datetime.datetime.now());    
                        self.outputfile = open(sensorfile, "w+")
                        # header = "time countTrials serPNS robotOutState lightsOn inRewardWin robotRZState solenoidOpen lickState zPOI"
                        header = "time trial serPNS triggered inRewardWin zPOI"
                        self.outputfile.write(header + "\n")
                        self.writeControl = "b"         
                        self.expController.write(self.writeControl)
                        while not self.expController.in_waiting:
                            pass
                        self.newline = self.expController.readline().split()
                        self.expController.flushInput()
                        print('trials completed:')
                        print(self.newline[0])
                        self.expBegan = True
                        self.expActive = True
                        self.reachDetected = False
                    except xiapi.Xi_error as err:
                        self.expActive = False
                        self.expBegan = False
                        self.unloadCameras()
                        if err.status == 10:
                            tkMessageBox.showinfo("Warning", "No image triggers detected.")
                            print (err)
                else:
                    self.expActive = True
            else:
               tkMessageBox.showinfo("Warning", "No POIs have been saved.") 
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def pauseExpCallback(self):        
        if self.expActive:
            self.expActive = False
            self.expController.write("p")
        else:
            tkMessageBox.showinfo("Warning", "No experiment to pause.")

    def endExpCallback(self):
        if self.expActive:
            self.expActive = False
            self.expEnded = True
            self.expController.write("e")
            self.expController.close()
        else:
            tkMessageBox.showinfo("Warning", "No experiment to end.")

    def loadCameras(self):              
        self.camList = []
        for i in range(numCams):
            print('opening camera %s ...' %(i))
            cam = xiapi.Camera(dev_id = i)
            cam.open_device()
            cam.set_imgdataformat(imgdataformat)
            cam.set_exposure(exposure)
            cam.set_gain(gain)
            cam.set_sensor_feature_value(sensor_feature_value)
            cam.set_gpi_selector(gpi_selector)
            cam.set_gpi_mode("XI_GPI_TRIGGER")
            cam.set_trigger_source(trigger_source)
            cam.set_gpo_selector(gpo_selector)
            cam.set_gpo_mode(gpo_mode)
            cam.set_height(512)
            cam.set_width(512)
            cam.set_offsetX(384)
            cam.set_offsetY(256)
            cam.enable_recent_frame()
            self.camList.append(cam)
            self.camList[i].start_acquisition()
        self.img = xiapi.Image()
        self.imgBuffer = deque()
        self.camerasLoaded = True

    def unloadCameras(self):
        for i in range(numCams):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camerasLoaded = False

    def acquireBaseline(self):
        if len(savedPOIs)==0:
            tkMessageBox.showinfo("Warning", "No saved POIs.")
        else:
            global poiMeans
            global poiStds
            global baselineAcquired
            cnt = 0 
            numImgs = int(np.round(float(baselineDur)*float(fps),decimals=0)) 
            baselinePOIs = []
            for i in range(numCams):
                baselinePOIs.append(np.zeros(shape = (len(savedPOIs[i]), numImgs)))
            print("Acquiring baseline...")
            for cnt in range(numImgs):
                self.expController.write(self.writeControl)
                for i in range(numCams):
                    self.camList[i].get_image(self.img,timeout = 2000)
                    npImg = self.img.get_image_data_numpy()
                    for j in range(len(savedPOIs[i])): 
                        baselinePOIs[i][j,cnt] = npImg[savedPOIs[i][j][1],savedPOIs[i][j][0]]
            poiMeans = []
            poiStds = []
            for i in range(numCams):   
                poiMeans.append(np.mean(baselinePOIs[i], axis = 1))             
                poiStds.append(np.std(np.sum(np.square(baselinePOIs[i]-poiMeans[i].reshape(len(savedPOIs[i]),1)),axis=1)))
            baselineAcquired = True
            print("Baseline acquired!")

    def runExperiment(self):
        global lightsOn  
        now = str(int(round(time.time()*1000)))   
        if self.newline[2]=='1': 
            lightsOn = 1
            for i in range(numCams):
                self.camList[i].get_image(self.img, timeout = 2000)                  
                npImg = self.img.get_image_data_numpy()
                for j in range(len(savedPOIs[i])): 
                    obsPOIs[i][j] = npImg[savedPOIs[i][j][1],savedPOIs[i][j][0]]
                zPOIs[i] = np.round(np.sum(np.square(obsPOIs[i]-poiMeans[i]))/(poiStds[i]+np.finfo(float).eps),decimals=1)
                self.imgBuffer.append(imgTup.ImageTuple(i, now, npImg))
                if len(self.imgBuffer)>numCams*bufferDur*fps and not self.reachDetected:
                    self.imgBuffer.popleft()
                    self.imgBuffer.popleft()
        else:
            lightsOn = 0
            for i in range(numCams):
                zPOIs[i] = 0     
        self.expController.write(self.writeControl) 
        while not self.expController.in_waiting:
            pass 
        self.newline = self.expController.readline() 
        self.expController.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(zPOIs))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and min(zPOIs)>poiThreshold: 
            self.reachDetected = True  
            self.reachInit = now     
            self.writeControl = 'r'
        elif self.newline[1] == 'e': 
            serBuf.serialize(self.imgBuffer,self.camerapath,self.newline)
            self.reachDetected = False
            self.writeControl = 's' 
            self.imgBuffer = deque() 
            print(self.newline[0])
        elif self.reachDetected and (int(now)-int(self.reachInit))>reachTimeout and self.newline[3]=='0':
            self.movRobCallback()

    def main(self):
        self.mainActive = True
        try:
            while self.mainActive:        
                if self.expActive and not self.expEnded:
                    try:
                        self.runExperiment()
                    except xiapi.Xi_error as err:
                        self.expActive = False
                        self.expBegan = False
                        self.unloadCameras()
                        if err.status == 10:
                            tkMessageBox.showinfo("Warning", "No image triggers detected.")
                self.window.update()
            if self.expControlOn:
                self.expController.write("e")
                self.expController.close()
            if self.robControlOn:
                    self.robController.write("e")
                    self.robController.close()
            if self.camerasLoaded:
                self.unloadCameras()
                self.imgBuffer = deque()
            self.window.destroy()
        except KeyboardInterrupt:
            answer = tkMessageBox.askyesnocancel("Question", "Save parameters?")
            if answer == True:
                if self.expControlOn:
                    self.expController.write("e")
                    self.expController.close()
                if self.robControlOn:
                    self.robController.write("e")
                    self.robController.close()
                if self.camerasLoaded:
                    self.unloadCameras()
                    self.imgBuffer = deque()
                self.saveParameters()
                self.window.destroy()
            elif answer == False:
                if self.expControlOn:
                    self.expController.write("e")
                    self.expController.close()
                if self.robControlOn:
                    self.robController.write("e")
                    self.robController.close()
                if self.camerasLoaded:
                    self.unloadCameras()
                    self.imgBuffer = deque()
                self.window.destroy()
            else:
                self.main()

class CameraSettings:

    def __init__(self, window):
        self.window = window
        self.window.title("Camera Settings")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.numCams = tk.StringVar()
        self.numCams.set(str(numCams))
        self.fps = tk.StringVar()
        self.fps.set(str(fps))
        self.exposure = tk.StringVar()
        self.exposure.set(str(exposure))
        self.gain = tk.StringVar()
        self.gain.set(str(gain))   
        self.gpi_mode = tk.StringVar()
        self.gpi_mode.set(gpi_mode)
        self.trigger_source = tk.StringVar()
        self.trigger_source.set(trigger_source)
        self.gpo_mode = tk.StringVar()
        self.gpo_mode.set(gpo_mode)
        self.baselineDur = tk.StringVar()
        self.baselineDur.set(baselineDur)
        self.bufferDur = tk.StringVar()
        self.bufferDur.set(bufferDur)
        self.poiThreshold = tk.StringVar()
        self.poiThreshold.set(poiThreshold)
        self.camsLoaded = False
        self.streaming = False
        self.streamStarted = False
        self.drawSaved = False
        self.addPOIs = False
        self.removePOIs = False
        self.addedPOIs = [[] for _ in range(numCams)]
        self.savedPOIs = [[] for _ in range(numCams)] 
        self.setup_UI()

    def onQuit(self):
        global numCams
        global fps
        global exposure
        global gain
        global baselineDur
        global bufferDur
        global trigger_source
        global gpo_mode
        global poiThreshold
        numCams = int(self.numCams.get())
        fps = int(self.fps.get())
        exposure = int(self.exposure.get())
        gain = float(self.gain.get()) 
        baselineDur = float(self.baselineDur.get())
        bufferDur = float(self.bufferDur.get())
        trigger_source = self.trigger_source.get()
        gpo_mode = self.gpo_mode.get()
        poiThreshold = float(self.poiThreshold.get())
        if self.streaming:
            self.stopStream()
        self.window.destroy()

    def setup_UI(self):        
        tk.Label(self.window,text="# Cameras:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=0, sticky='W')   
        self.numCamsMenu = tk.OptionMenu(self.window,self.numCams,"1","2","3")
        self.numCamsMenu.configure(width=12,anchor="w")
        self.numCamsMenu.grid(row=0, column=1)
        tk.Label(self.window,text="FPS:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, sticky='W')   
        tk.Entry(self.window,textvariable=self.fps,width=17).grid(row=1, column=1)
        tk.Label(self.window,text="Exposure (usec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, sticky='W')   
        tk.Entry(self.window,textvariable=self.exposure,width=17).grid(row=2, column=1)
        tk.Label(self.window,text="Gain:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, sticky='W')   
        tk.Entry(self.window,textvariable=self.gain,width=17).grid(row=3, column=1)
        tk.Label(self.window,text="Trigger Source:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, sticky='W')   
        self.gpiTrigMenu = tk.OptionMenu(self.window,self.trigger_source,
            "XI_TRG_OFF",
            "XI_TRG_EDGE_RISING",
            "XI_TRG_EDGE_FALLING",
            "XI_TRG_SOFTWARE",
            "XI_TRG_LEVEL_HIGH",
            "XI_TRG_LEVEL_LOW")
        self.gpiTrigMenu.configure(width=12,anchor="w")
        self.gpiTrigMenu.grid(row=4, column=1)
        tk.Label(self.window,text="Sync Mode:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, sticky='W')   
        self.gpoModeMenu = tk.OptionMenu(self.window,self.gpo_mode,
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
        tk.Label(self.window,text="Image Buffer (sec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, sticky='W')   
        tk.Entry(self.window,textvariable=self.bufferDur,width=17).grid(row=6, column=1)
        tk.Button(self.window,text="Start Streaming",font='Arial 10 bold',width=14,command=self.startStreamCallback).grid(row=7, column=0,sticky="e")
        tk.Button(self.window,text="Stop Streaming",font='Arial 10 bold',width=14,command=self.stopStreamCallback).grid(row=8, column=0,sticky="e")
        tk.Button(self.window,text="Load POIs",font='Arial 10 bold',width=14,command=self.loadPOIsCallback).grid(row=7, column=1)
        tk.Button(self.window,text="Save POIs",font='Arial 10 bold',width=14,command=self.savePOIsCallback).grid(row=8, column=1)
        tk.Button(self.window,text="Add POIs",font='Arial 10 bold',width=14,command=self.addPOIsCallback).grid(row=7, column=2)
        tk.Button(self.window,text="Remove POIs",font='Arial 10 bold',width=14,command=self.removePOIsCallback).grid(row=8, column=2)
        tk.Label(self.window,text="Acquire Baseline (sec):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, sticky='W')   
        tk.Entry(self.window,textvariable=self.baselineDur,width=17).grid(row=9, column=1)
        tk.Button(self.window,text="Start",font='Arial 10 bold',width=14,command=self.baselineCallback).grid(row=9, column=2)
        tk.Label(self.window,text="POI Threshold (stdev):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, sticky='W')   
        tk.Entry(self.window,textvariable=self.poiThreshold,width=17).grid(row=10, column=1)

    def startStreamCallback(self):
        if not self.streamStarted:
            global numCams
            global fps
            global exposure
            global gain
            numCams = int(self.numCams.get())
            fps = 30
            exposure = int(self.exposure.get())
            gain = float(self.gain.get())          
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
        for i in range(numCams):
            print('opening camera %s ...' %(i))
            cam = xiapi.Camera(dev_id = i)
            cam.open_device()
            cam.set_imgdataformat(imgdataformat)
            cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            cam.set_exposure(exposure)
            cam.set_gain(gain)
            cam.set_sensor_feature_value(sensor_feature_value)
            cam.set_gpi_selector(gpi_selector)
            cam.set_gpi_mode("XI_GPI_OFF")
            cam.set_trigger_source("XI_TRG_OFF")
            cam.set_gpo_selector(gpo_selector)
            cam.set_gpo_mode(gpo_mode)
            cam.set_framerate(fps)
            cam.set_region_selector(0)
            cam.set_height(512)
            cam.set_width(512)
            cam.set_offsetX(384)
            cam.set_offsetY(256)
            self.camList.append(cam)
            self.camList[i].start_acquisition()   
        self.camsLoaded = True        

    def unloadCameras(self):
        for i in range(numCams):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camsLoaded = False

    def startStream(self):
        if not self.streamStarted:
            self.imgWidth = self.camList[0].get_width()
            self.imgHeight = self.camList[0].get_height()
            self.camWindows = [0,0]
            for i in range(numCams):
                self.camWindows[i] = tk.Toplevel(self.window)
                self.camWindows[i].title("Camera"+str(i))
                self.camWindows[i].protocol("WM_DELETE_WINDOW", self.stopStream)
                self.camWindows[i].canvas = tk.Canvas(self.camWindows[i], width = self.imgWidth, height = self.imgHeight)
                self.camWindows[i].canvas.grid(row=0,column= 0)            
            self.streamStarted = True
        self.delay = 20
        self.streaming = True
        self.refresh()

    def stopStream(self):
        self.streaming = False 
        self.streamStarted = False 
        self.poiActive = False  
        self.drawSaved = False    
        for i in range(numCams):
            self.camWindows[i].destroy()
        self.unloadCameras()

    def refresh(self):
        if self.streaming:
            npImg = np.zeros(shape = (self.imgHeight, self.imgWidth)) 
            img = xiapi.Image()
            self.photoImg = [0,0]
            for i in range(numCams):
                self.camList[i].get_image(img,timeout = 2000)
                npImg = img.get_image_data_numpy()
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
            self.window.after(self.delay,self.refresh)

    def loadPOIsCallback(self):
        if self.streaming:
            if len(savedPOIs)>0:
                self.savedPOIs = savedPOIs
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
        for i in range(numCams):
            self.savedPOIs[i] += self.addedPOIs[i] 
        global savedPOIs
        savedPOIs = self.savedPOIs 
        self.addedPOIs = [[] for _ in range(numCams)]

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

    def baselineCallback(self):
        if len(savedPOIs)==0:
            tkMessageBox.showinfo("Warning", "No saved POIs.")
        else:
            if self.streaming:
                self.stopStream()
            global numCams
            global fps
            global exposure
            global gain
            global poiMeans
            global poiStds
            global baselineAcquired
            numCams = int(self.numCams.get())
            fps = int(self.fps.get())
            exposure = int(self.exposure.get())
            gain = float(self.gain.get())         
            self.loadCameras()
            cnt = 0 
            numImgs = int(np.round(float(baselineDur)*float(fps),decimals=0)) 
            img = xiapi.Image()
            baselinePOIs = []
            for i in range(numCams):
                baselinePOIs.append(np.zeros(shape = (len(savedPOIs[i]), numImgs)))
            print("Acquiring baseline...")
            for cnt in range(numImgs):
                for i in range(numCams):
                    self.camList[i].get_image(img,timeout = 20000)
                    npImg = img.get_image_data_numpy()
                    for j in range(len(savedPOIs[i])): 
                        baselinePOIs[i][j,cnt] = npImg[savedPOIs[i][j][1],savedPOIs[i][j][0]]
            poiMeans = []
            poiStds = []
            for i in range(numCams):   
                poiMeans.append(np.mean(baselinePOIs[i], axis = 1))             
                poiStds.append(np.std(np.sum(np.square(baselinePOIs[i]-poiMeans[i].reshape(len(savedPOIs[i]),1)),axis=1)))
            baselineAcquired = True
            self.unloadCameras()
            print("Baseline acquired!")

class ExperimentSettings:

    def __init__(self, window, controller):
        self.window = window
        self.window.title("Experiment Settings") 
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.controller = controller    
        self.lightsOnDur = tk.StringVar()
        self.lightsOnDur.set(str(lightsOnDur))
        self.lightsOffDur = tk.StringVar()
        self.lightsOffDur.set(str(lightsOffDur))
        self.rewardWinDur = tk.StringVar()
        self.rewardWinDur.set(str(rewardWinDur))
        self.maxRewards = tk.StringVar()
        self.maxRewards.set(str(maxRewards))
        self.solenoidOpenDur = tk.StringVar()
        self.solenoidOpenDur.set(str(solenoidOpenDur))
        self.solenoidBounceDur = tk.StringVar()
        self.solenoidBounceDur.set(str(solenoidBounceDur))
        self.flushDur = tk.StringVar()
        self.flushDur.set(str(flushDur))
        self.setup_UI()

    def onQuit(self):
        global lightsOnDur
        global lightsOffDur
        global rewardWinDur
        global maxRewards
        global solenoidOpenDur
        global solenoidBounceDur
        global flushDur
        lightsOnDur = int(self.lightsOnDur.get())
        lightsOffDur = int(self.lightsOffDur.get())
        rewardWinDur = int(self.rewardWinDur.get())
        maxRewards = int(self.maxRewards.get()) 
        solenoidOpenDur = int(self.solenoidOpenDur.get())
        solenoidBounceDur = int(self.solenoidBounceDur.get())
        flushDur = int(self.flushDur.get())
        self.window.destroy()

    def setup_UI(self):
        tk.Label(self.window,text="Lights On (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=0, column=0)   
        tk.Entry(self.window,textvariable=self.lightsOnDur,width=17).grid(row=0, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.lightsOnDurRead).grid(row=0, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.lightsOnDurWrite).grid(row=0, column=3)
        tk.Label(self.window,text="Lights Off (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, column=0)   
        tk.Entry(self.window,textvariable=self.lightsOffDur,width=17).grid(row=1, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.lightsOffDurRead).grid(row=1, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.lightsOffDurWrite).grid(row=1, column=3)
        tk.Label(self.window,text="Reward Window (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=0)   
        tk.Entry(self.window,textvariable=self.rewardWinDur,width=17).grid(row=2, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.rewardWinDurRead).grid(row=2, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.rewardWinDurWrite).grid(row=2, column=3)
        tk.Label(self.window,text="# Rewards/Trial:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=0)   
        tk.Entry(self.window,textvariable=self.maxRewards,width=17).grid(row=3, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.maxRewardsRead).grid(row=3, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.maxRewardsWrite).grid(row=3, column=3)
        tk.Label(self.window,text="Solenoid Open (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=0)   
        tk.Entry(self.window,textvariable=self.solenoidOpenDur,width=17).grid(row=4, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.solenoidOpenDurRead).grid(row=4, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.solenoidOpenDurWrite).grid(row=4, column=3)
        tk.Label(self.window,text="Solenoid Bounce (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=0)   
        tk.Entry(self.window,textvariable=self.solenoidBounceDur,width=17).grid(row=5, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.solenoidBounceDurRead).grid(row=5, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.solenoidBounceDurWrite).grid(row=5, column=3)
        tk.Label(self.window,text="Flush (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=0)   
        tk.Entry(self.window,textvariable=self.flushDur,width=17).grid(row=6, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.flushDurRead).grid(row=6, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.flushDurWrite).grid(row=6, column=3)

    def lightsOnDurRead(self):
        self.controller.write("v")

    def lightsOnDurWrite(self):
        self.controller.write("v")

    def lightsOffDurRead(self):
        self.controller.write("v")

    def lightsOffDurWrite(self):
        self.controller.write("v")

    def rewardWinDurRead(self):
        self.controller.write("v")

    def rewardWinDurWrite(self):
        self.controller.write("v")

    def maxRewardsRead(self):
        self.controller.write("v")

    def maxRewardsWrite(self):
        self.controller.write("v")

    def solenoidOpenDurRead(self):
        self.controller.write("v")

    def solenoidOpenDurWrite(self):
        self.controller.write("v")

    def solenoidBounceDurRead(self):
        self.controller.write("v")

    def solenoidBounceDurWrite(self):
        self.controller.write("v")

    def flushDurRead(self):
        self.controller.write("v")

    def flushDurWrite(self):
        self.controller.write("v")

class RobotSettings:

    def __init__(self, window, controller):
        self.window = window
        self.window.title("Robot Settings") 
        self.controller = controller    
        self.setup_UI()

    def setup_UI(self):
        print('boiler code')

#start program
ReachMaster(tk.Tk())

