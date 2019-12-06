import config
import settings.camera_settings as camset
import settings.experiment_settings as expset
import settings.robot_settings as robset
# import utils.ImageTuple as imgTup
# import utils.serializeBuffer as serBuf
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
from serial.tools import list_ports
import os 
from collections import deque
from vidgear.gears import WriteGear
import json

class ReachMaster:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("ReachMaster 1.0")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.mainActive = False
        self.cfg = config.default_cfg()
        config.save_tmp(self.cfg)
        self.output_params = self.cfg['CameraSettings']['output_params']
        self.dataDir = tk.StringVar()
        self.dataDir.set(self.cfg['ReachMaster']['dataDir']) 
        self.cfgFile = tk.StringVar()
        self.cfgFile.set(self.cfg['ReachMaster']['cfgFile'])
        self.portList = list(list_ports.comports())
        for i in range(len(self.portList)):
            self.portList[i] = self.portList[i].device
        self.expControlPath = tk.StringVar()
        self.robControlPath = tk.StringVar()
        if self.cfg['ReachMaster']['expControlPath'] in self.portList:
            self.expControlPath.set(self.cfg['ReachMaster']['expControlPath']) 
        else:
            self.expControlPath.set(self.portList[0])
        if self.cfg['ReachMaster']['robControlPath'] in self.portList:
            self.robControlPath.set(self.cfg['ReachMaster']['robControlPath']) 
        else:
            self.robControlPath.set(self.portList[0])
        self.expControlOn = False
        self.robControlOn = False
        self.camerasLoaded = False
        self.baselineAcquired = False
        self.expActive = False    
        self.lightsOn = 0  
        self.calibrationLoaded = False
        self.commandsLoaded = False
        self.setup_UI()

    def onQuit(self):        
        answer = tkMessageBox.askyesnocancel("Question", "Save Configuration?")
        if answer == True:            
            config.save_cfg(self.cfg)
            self.mainActive = False
        elif answer == False:
            self.mainActive = False
        else:
            pass

    def setup_UI(self):
        tk.Label(text="Data Directory:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=0, column=0)
        tk.Label(textvariable=self.dataDir, bg="white").grid(row=0, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.ddBrowseCallback).grid(row=0, column=2)
        tk.Label(text="Configuration File:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=1, column=0)
        tk.Label(textvariable=self.cfgFile, bg="white").grid(row=1, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.cfBrowseCallback).grid(row=1, column=2)
        tk.Label(text="Experiment Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=2, column=0)
        expControllerMenu = apply(tk.OptionMenu, (self.window, self.expControlPath) + tuple(self.portList))
        expControllerMenu.grid(row=2, column=1)
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
        tk.Button(text="Start Experiment", font='Arial 10 bold',width=14, command=self.startExpCallback).grid(row=5, column=2)

    def ddBrowseCallback(self):
        self.dataDir.set(tkFileDialog.askdirectory())
        self.cfg['ReachMaster']['dataDir'] = self.dataDir.get()

    def cfBrowseCallback(self):
        self.cfgFile.set(tkFileDialog.askopenfilename())
        self.cfg = config.json_load_byteified(open(self.cfgFile.get()))
        self.cfg['ReachMaster']['cfgFile'] = self.cfgFile.get()
        config.save_tmp(self.cfg)
        self.output_params = self.cfg['CameraSettings']['self.output_params']

    def expConnectCallback(self):
        global expController
        self.cfg['ReachMaster']['expControlPath'] = self.expControlPath.get()
        expController = serial.Serial(self.cfg['ReachMaster']['expControlPath'],
            self.cfg['ReachMaster']['serialBaud'],
            timeout=self.cfg['ReachMaster']['controlTimeout'])
        time.sleep(2) #wait for controller to wake up
        expController.flushInput()
        expController.write("h")
        response = expController.read()
        if response=="h":
            self.expControlOn = True
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def expDisconnectCallback(self):
        if self.expControlOn:
            if not self.expActive:
                expController.write("e")
                expController.close()
                self.expControlOn = False
            else:
                tkMessageBox.showinfo("Warning", "Experiment is active! Not safe to disconnect.")
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def robConnectCallback(self):
        global robController
        self.cfg['ReachMaster']['robControlPath'] = self.robControlPath.get()
        robController = serial.Serial(self.cfg['ReachMaster']['robControlPath'],
            self.cfg['ReachMaster']['serialBaud'],
            timeout=self.cfg['ReachMaster']['controlTimeout'])
        # time.sleep(2) #wait for controller to wake up
        robController.flushInput()
        robController.write("h")
        response = robController.readline()
        if response:
            self.robControlOn = True
            if self.cfg['RobotSettings']['calibrationFile'] != 'None':
                self.loadCalibration()                
            if self.cfg['RobotSettings']['commandType'] != 'None':
                self.loadCommands()
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def robDisconnectCallback(self):
        if self.robControlOn:
            robController.write("e")
            robController.close()
        else:
            tkMessageBox.showinfo("Warning", "Robot Controller not connected.")

    def loadCalibration(self):
        dis = self.cfg['RobotSettings']['dis']#.encode("utf-8")
        pos = self.cfg['RobotSettings']['pos']#.encode("utf-8")
        xPushDur = self.cfg['RobotSettings']['xPushDur']#.encode("utf-8")
        xPullDur = self.cfg['RobotSettings']['xPullDur']#.encode("utf-8")
        yPushDur = self.cfg['RobotSettings']['yPushDur']#.encode("utf-8")
        yPullDur = self.cfg['RobotSettings']['yPullDur']#.encode("utf-8")
        zPushDur = self.cfg['RobotSettings']['zPushDur']#.encode("utf-8")
        zPullDur = self.cfg['RobotSettings']['zPullDur']#.encode("utf-8")
        robController.write("c")
        if robController.read() == "c":
            robController.write("pos\n")
            if robController.read() == "c":
                robController.write(pos)
        if robController.read() == "c":
            print('pos loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load pos.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("dis\n")
            if robController.read() == "c":
                robController.write(dis)
                if robController.read() == "c":
                    print('dis loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load dis.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("xPushDur\n")
            if robController.read() == "c":
                robController.write(xPushDur)
                if robController.read() == "c":
                    print('xPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load xPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("xPullDur\n")
            if robController.read() == "c":
                robController.write(xPullDur)
                if robController.read() == "c":
                    print('xPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load xPullDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("yPushDur\n")
            if robController.read() == "c":
                robController.write(yPushDur)
                if robController.read() == "c":
                    print('yPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load yPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("yPullDur\n")
            if robController.read() == "c":
                robController.write(yPullDur)
                if robController.read() == "c":
                    print('yPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load yPullDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("zPushDur\n")
            if robController.read() == "c":
                robController.write(zPushDur)
                if robController.read() == "c":
                    print('zPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load zPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("zPullDur\n")
            if robController.read() == "c":
                robController.write(zPullDur)
                if robController.read() == "c":
                    print('zPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load zPullDur.")
        self.calibrationLoaded = True  

    def loadCommands(self):
        Ly = self.cfg['RobotSettings']['Ly']
        Lz = self.cfg['RobotSettings']['Lz']
        Axx = self.cfg['RobotSettings']['Axx']
        Ayy = self.cfg['RobotSettings']['Ayy']
        Azz = self.cfg['RobotSettings']['Azz']
        x0 = self.cfg['RobotSettings']['x0']
        y0 = self.cfg['RobotSettings']['y0']
        z0 = self.cfg['RobotSettings']['z0']      
        n = 100
        if self.cfg['RobotSettings']['commandType'] == "sampleContinuous":
            rLow = self.cfg['RobotSettings']['rLow']
            rHigh = self.cfg['RobotSettings']['rHigh']
            thetaMag = self.cfg['RobotSettings']['thetaMag']
            r = rLow + (rHigh-rLow)*np.random.uniform(low=0.0,high=1.0,size=(500*n))**(1.0/3.0)
            thetay = thetaMag*np.random.uniform(low=-1,high=1,size=500*n)
            thetaz = thetaMag*np.random.uniform(low=-1,high=1,size=500*n)
            theta = np.sqrt(thetay**2+thetaz**2)
            r = r[theta<=thetaMag][0:n]
            thetay = thetay[theta<=thetaMag][0:n]
            thetaz = thetaz[theta<=thetaMag][0:n]
        elif self.cfg['RobotSettings']['commandType'] == "sampleDiscrete":
            rSet,thetaySet,thetazSet = np.loadtxt(self.cfg['RobotSettings']['commandFile'],\
            skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
            randSample = np.random.choice(range(len(rSet)),replace=True,size=n)
            r = rSet[randSample]
            thetay = thetaySet[randSample]
            thetaz = thetazSet[randSample]
        elif self.cfg['RobotSettings']['commandType'] == "fromFile":
            r,thetay,thetaz = np.loadtxt(self.cfg['RobotSettings']['commandFile'],\
            skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
        else:
            tkMessageBox.showinfo("Warning", "Invalid command type.")
            return
        Ax = np.sqrt(Axx**2+r**2-2*Axx*r*np.cos(thetay)*np.cos(thetaz))
        gammay = -np.arcsin(np.sin(thetay)*np.sqrt((r*np.cos(thetay)*np.cos(thetaz))**2+\
            (r*np.sin(thetay)*np.cos(thetaz))**2)/np.sqrt((Axx-r*np.cos(thetay)*\
                np.cos(thetaz))**2+(r*np.sin(thetay)*np.cos(thetaz))**2))
        gammaz = -np.arcsin(r*np.sin(thetaz)/Ax)
        Ay = np.sqrt((Ly-Ly*np.cos(gammay)*np.cos(gammaz))**2+\
            (Ayy-Ly*np.sin(gammay)*np.cos(gammaz))**2+(Ly*np.sin(gammaz))**2)
        Az = np.sqrt((Lz-Lz*np.cos(gammay)*np.cos(gammaz))**2+\
            (Lz*np.sin(gammay)*np.cos(gammaz))**2+(Azz-Lz*np.sin(gammaz))**2)
        Ax = np.round((Ax-Axx)/50*1024+x0,decimals=1)
        Ay = np.round((Ay-Ayy)/50*1024+y0,decimals=1)
        Az = np.round((Az-Azz)/50*1024+z0,decimals=1)
        x = np.array2string(Ax,formatter={'float_kind':lambda Ax: "%.1f" % Ax})
        y = np.array2string(Ay,formatter={'float_kind':lambda Ay: "%.1f" % Ay})
        z = np.array2string(Az,formatter={'float_kind':lambda Az: "%.1f" % Az})
        r = np.array2string(r,formatter={'float_kind':lambda r: "%.1f" % r})
        thetay = np.array2string(thetay,formatter={'float_kind':lambda thetay: "%.1f" % thetay})
        thetaz = np.array2string(thetaz,formatter={'float_kind':lambda thetaz: "%.1f" % thetaz})
        x = x[1:-1]+' '
        y = y[1:-1]+' '
        z = z[1:-1]+' '
        r = r[1:-1]+' '
        thetay = thetay[1:-1]+' '
        thetaz = thetaz[1:-1]+' '
        self.cfg['RobotSettings']['x'] = x
        self.cfg['RobotSettings']['y'] = y
        self.cfg['RobotSettings']['z'] = z
        self.cfg['RobotSettings']['r'] = r
        self.cfg['RobotSettings']['thetay'] = thetay
        self.cfg['RobotSettings']['thetaz'] = thetaz
        robController.write("p")
        if robController.read() == "p":
            robController.write("xCommandPos\n")
            if robController.read() == "p":
                robController.write(x)
        if robController.read() == "p":
            print('x commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load x commands.")
        robController.write("p")
        if robController.read() == "p":
            robController.write("yCommandPos\n")
            if robController.read() == "p":
                robController.write(y)
        if robController.read() == "p":
            print('y commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load y commands.")
        robController.write("p")
        if robController.read() == "p":
            robController.write("zCommandPos\n")
            if robController.read() == "p":
                robController.write(z)
        if robController.read() == "p":
            print('z commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load z commands.") 
        self.commandsLoaded = True

    def camSetCallback(self):  
        if not self.expActive:
            if self.expControlOn:
                self.expDisconnectCallback()
                time.sleep(2)
                camset.CameraSettings(self.window)
            else:
                tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")
        else:
            pass

    def expSetCallback(self):
        if not self.expActive:
            if self.expControlOn:
                self.expDisconnectCallback()
                time.sleep(2)   
                expset.ExperimentSettings(self.window)
            else:
                tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")
        else:
            pass

    def robSetCallback(self):
        if self.robControlOn:
            self.robDisconnectCallback()
            time.sleep(2)   
            robset.RobotSettings(self.window)
        else:
            tkMessageBox.showinfo("Warning", "Robot Controller not connected.")

    def movRobCallback(self):
        if self.expControlOn:
            expController.write("m")
            if self.cfg['CameraSettings']['vidMode'] == "TRIALS":
                self.lightsOn = 0
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def ledCallback(self):
        if self.expControlOn:
            expController.write("l")          
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def lightsCallback(self):
        if self.expControlOn:
            expController.write("n")
            self.lightsOn = not self.lightsOn
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def waterCallback(self):
        if self.expControlOn:
            expController.write("w")
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def flushCallback(self):
        if self.expControlOn:
            if not self.expActive:
                expController.write("f")
            else:
                pass
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def startExpCallback(self):
        if self.expControlOn:          
            if len(self.cfg['CameraSettings']['savedPOIs'])>0:
                if not self.expActive:
                    try:
                        self.loadCameras()
                        if not self.baselineAcquired:
                            if not self.lightsOn:
                                self.lightsCallback()
                            self.writeControl = "s"
                            self.acquireBaseline()                       
                        global obsPOIs
                        global zPOIs
                        obsPOIs = []
                        zPOIs = []
                        for i in range(self.cfg['CameraSettings']['numCams']):
                            obsPOIs.append(np.zeros(len(self.cfg['CameraSettings']['savedPOIs'][i])))
                            zPOIs.append(0)
                        self.buffer_full = False 
                        self.sensorpath = self.cfg['ReachMaster']['dataDir'] + "/sensor_data/"
                        self.camerapath = self.cfg['ReachMaster']['dataDir'] + "/videos/" 
                        if not os.path.isdir(self.sensorpath):
                            os.makedirs(self.sensorpath)
                        if not os.path.isdir(self.camerapath):
                            os.makedirs(self.camerapath)
                        sensorfile = self.sensorpath + str(datetime.datetime.now())    
                        self.outputfile = open(sensorfile, "w+")
                        # header = "time countTrials serPNS robotOutState lightsOn inRewardWin robotRZState solenoidOpen lickState zPOI"
                        header = "time trial serPNS triggered inRewardWin zPOI"
                        self.outputfile.write(header + "\n")
                        self.writeControl = "b"         
                        expController.write(self.writeControl)
                        while not expController.in_waiting:
                            pass
                        self.newline = expController.readline().split()
                        expController.flushInput()
                        if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS":
                            self.vid_fn = self.camerapath + str(datetime.datetime.now()) + '.mp4' 
                            self.video = WriteGear(
                                output_filename = self.vid_fn,
                                compression_mode = True,
                                logging=False,
                                **self.output_params)
                        print('trials completed:')
                        print(self.newline[0])
                        self.expActive = True
                        self.reachDetected = False
                    except xiapi.Xi_error as err:
                        self.expActive = False
                        self.unloadCameras()
                        if err.status == 10:
                            tkMessageBox.showinfo("Warning", "No image triggers detected.")
                            print (err)
            else:
               tkMessageBox.showinfo("Warning", "No POIs have been saved.") 
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

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
            cam.set_gpi_mode(self.cfg['CameraSettings']['gpi_mode'])
            cam.set_trigger_source(self.cfg['CameraSettings']['trigger_source'])
            cam.set_gpo_selector(self.cfg['CameraSettings']['gpo_selector'])
            cam.set_gpo_mode(self.cfg['CameraSettings']['gpo_mode'])
            if self.cfg['CameraSettings']['downsampling'] == "XI_DWN_2x2":
                cam.set_downsampling(self.cfg['CameraSettings']['downsampling'])
            else:
                cam.set_height(self.cfg['CameraSettings']['imgHeight'])
                cam.set_width(self.cfg['CameraSettings']['imgWidth'])
                cam.set_offsetX(self.cfg['CameraSettings']['offsetX'])
                cam.set_offsetY(self.cfg['CameraSettings']['offsetY'])
            cam.enable_recent_frame()
            self.camList.append(cam)
            self.camList[i].start_acquisition()
        self.img = xiapi.Image()
        self.imgBuffer = deque()
        self.camerasLoaded = True

    def unloadCameras(self):
        for i in range(self.cfg['CameraSettings']['numCams']):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camerasLoaded = False

    def acquireBaseline(self):
        if len(self.cfg['CameraSettings']['savedPOIs'])==0:
            tkMessageBox.showinfo("Warning", "No saved POIs.")
        else:
            global poiMeans
            global poiStds
            global baselineAcquired
            cnt = 0 
            numImgs = int(np.round(float(self.cfg['ExperimentSettings']['baselineDur'])*\
                float(self.cfg['CameraSettings']['fps']),decimals=0)) 
            baselinePOIs = []
            for i in range(self.cfg['CameraSettings']['numCams']):
                baselinePOIs.append(np.zeros(shape = (len(self.cfg['CameraSettings']['savedPOIs'][i]), numImgs)))
            print("Acquiring baseline...")
            for cnt in range(numImgs):
                time.sleep(0.005)
                expController.write(self.writeControl)
                for i in range(self.cfg['CameraSettings']['numCams']):
                    self.camList[i].get_image(self.img,timeout = 2000)
                    npImg = self.img.get_image_data_numpy()
                    for j in range(len(self.cfg['CameraSettings']['savedPOIs'][i])): 
                        baselinePOIs[i][j,cnt] = npImg[self.cfg['CameraSettings']['savedPOIs'][i][j][1],
                            self.cfg['CameraSettings']['savedPOIs'][i][j][0]]
            poiMeans = []
            poiStds = []
            for i in range(self.cfg['CameraSettings']['numCams']):   
                poiMeans.append(np.mean(baselinePOIs[i], axis = 1))             
                poiStds.append(np.std(np.sum(np.square(baselinePOIs[i]-
                    poiMeans[i].reshape(len(self.cfg['CameraSettings']['savedPOIs'][i]),1)),axis=0)))
            self.baselineAcquired = True
            print("Baseline acquired!")

    def runConintuous(self):
        now = str(int(round(time.time()*1000)))  
        if self.newline[3]=='1': 
            self.lightsOn = 1 
            for i in range(self.cfg['CameraSettings']['numCams']):
                self.camList[i].get_image(self.img, timeout = 2000)                  
                npImg = self.img.get_image_data_numpy()
                for j in range(len(self.cfg['CameraSettings']['savedPOIs'][i])): 
                    obsPOIs[i][j] = npImg[self.cfg['CameraSettings']['savedPOIs'][i][j][1],
                        self.cfg['CameraSettings']['savedPOIs'][i][j][0]]
                zPOIs[i] = np.round(np.sum(np.square(obsPOIs[i]-poiMeans[i]))/(poiStds[i]+np.finfo(float).eps),decimals=1)
                if i == 0:
                    frame = npImg
                else:
                    frame = np.hstack((frame,npImg))
        else:
            self.lightsOn = 0
            for i in range(self.cfg['CameraSettings']['numCams']):
                zPOIs[i] = 0
        expController.write(self.writeControl) 
        if self.newline[3]=='1':
            self.video.write(frame)
        while not expController.in_waiting:
            pass 
        self.newline = expController.readline() 
        expController.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(zPOIs))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and self.newline[2] == '0' and min(zPOIs)>self.cfg['CameraSettings']['poiThreshold']: 
            self.reachDetected = True  
            self.reachInit = now     
            self.writeControl = 'r'
        elif self.newline[1] == 'e': 
            self.reachDetected = False
            self.writeControl = 's'   
            print(self.newline[0])
        elif self.reachDetected and\
         (int(now)-int(self.reachInit))>self.cfg['ExperimentSettings']['reachTimeout'] and self.newline[4]=='0':
            self.movRobCallback()

    def runTrials(self): 
        now = str(int(round(time.time()*1000)))   
        if self.newline[3]=='1': 
            self.lightsOn = 1
            for i in range(self.cfg['CameraSettings']['numCams']):
                self.camList[i].get_image(self.img, timeout = 2000)                  
                npImg = self.img.get_image_data_numpy()
                for j in range(len(self.cfg['CameraSettings']['savedPOIs'][i])): 
                    obsPOIs[i][j] = npImg[self.cfg['CameraSettings']['savedPOIs'][i][j][1],
                        self.cfg['CameraSettings']['savedPOIs'][i][j][0]]
                zPOIs[i] = np.round(np.sum(np.square(obsPOIs[i]-poiMeans[i]))/(poiStds[i]+np.finfo(float).eps),decimals=1)
                self.imgBuffer.append(npImg)
                if len(self.imgBuffer)>self.cfg['CameraSettings']['numCams']*\
                self.cfg['ExperimentSettings']['bufferDur']*self.cfg['CameraSettings']['fps'] and not self.reachDetected:
                    self.imgBuffer.popleft()
        else:
            self.lightsOn = 0
            for i in range(self.cfg['CameraSettings']['numCams']):
                zPOIs[i] = 0     
        expController.write(self.writeControl) 
        while not expController.in_waiting:
            pass 
        self.newline = expController.readline() 
        expController.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(zPOIs))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and self.newline[2] == '0' and min(zPOIs)>self.cfg['CameraSettings']['poiThreshold']: 
            self.reachDetected = True  
            self.reachInit = now     
            self.writeControl = 'r'
        elif self.newline[1] == 'e': 
            # serBuf.serialize(self.imgBuffer,self.camerapath,self.newline)
            if not os.path.isdir(self.camerapath):
                os.makedirs(self.camerapath)
            trial_fn = self.camerapath + 'trial: ' + str(self.newline[0]) + '.mp4' 
            self.video = WriteGear(output_filename = trial_fn,compression_mode = True,logging=False,**self.output_params)
            for i in range(len(self.imgBuffer)/self.cfg['CameraSettings']['numCams']):
                # frame = cv2.cvtColor(self.imgBuffer[(i+1)*numCams-numCams],cv2.COLOR_BAYER_BG2BGR)
                frame = self.imgBuffer[(i+1)*self.cfg['CameraSettings']['numCams']-self.cfg['CameraSettings']['numCams']]
                for f in range(self.cfg['CameraSettings']['numCams']-1):
                    # frame = np.hstack((frame,cv2.cvtColor(self.imgBuffer[(i+1)*numCams-numCams+f+1],cv2.COLOR_BAYER_BG2BGR))) 
                    frame = np.hstack((frame,self.imgBuffer[(i+1)*\
                        self.cfg['CameraSettings']['numCams']-self.cfg['CameraSettings']['numCams']+f+1])) 
                self.video.write(frame)   
            self.video.close()
            self.reachDetected = False
            self.writeControl = 's' 
            self.imgBuffer = deque() 
            print(self.newline[0])
        elif self.reachDetected and (int(now)-int(self.reachInit))>\
        self.cfg['ExperimentSettings']['reachTimeout'] and self.newline[4]=='0':
            self.movRobCallback()

    def run(self):
        self.mainActive = True
        try:
            while self.mainActive:        
                if self.expActive:
                    try:
                        if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS":
                            self.runConintuous()
                        elif self.cfg['CameraSettings']['vidMode'] == "TRIALS":
                            self.runTrials()
                    except xiapi.Xi_error as err:
                        self.expActive = False
                        self.unloadCameras()
                        if err.status == 10:
                            tkMessageBox.showinfo("Warning", "No image triggers detected.")
                self.window.update()
            if self.expControlOn:
                expController.write("e")
                expController.close()
            if self.robControlOn:
                    robController.write("e")
                    robController.close()
            if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS" and self.baselineAcquired:
                self.video.close()
            if self.camerasLoaded:
                self.unloadCameras()
                self.imgBuffer = deque()
            self.window.destroy()
        except KeyboardInterrupt:
            answer = tkMessageBox.askyesnocancel("Question", "Save self.cfg?")
            if answer == True:
                if self.expControlOn:
                    expController.write("e")
                    expController.close()
                if self.robControlOn:
                    robController.write("e")
                    robController.close()
                if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS":
                    self.video.close()
                if self.camerasLoaded:
                    self.unloadCameras()
                    self.imgBuffer = deque()
                self.savecfg()
                self.window.destroy()
            elif answer == False:
                if self.expControlOn:
                    expController.write("e")
                    expController.close()
                if self.robControlOn:
                    robController.write("e")
                    robController.close()
                if self.cfg['CameraSettings']['vidMode'] == "CONTINUOUS":
                    self.video.close()
                if self.camerasLoaded:
                    self.unloadCameras()
                    self.imgBuffer = deque()
                self.window.destroy()
            else:
                self.run()

if __name__ == '__main__':
    rm = ReachMaster()
    rm.run()