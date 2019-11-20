import Tkinter as tk 
import tkFileDialog
import tkMessageBox
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
import os 
from collections import deque
from vidgear.gears import WriteGear
import json
import pandas as pd

#load default workspace 
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

workspace = json_load_byteified(open('defaultWorkspace.txt'))
output_params = workspace['CameraSettings']['output_params']

class ReachMaster:

    def __init__(self, window):
        #setup root UI
        self.window = window
        self.window.title("ReachMaster 1.0")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.mainActive = False
        self.dataDir = tk.StringVar()
        self.dataDir.set(workspace['ReachMaster']['dataDir']) 
        self.workspaceFile = tk.StringVar()
        self.workspaceFile.set(workspace['ReachMaster']['workspaceFile'])
        self.portList = list(list_ports.comports())
        for i in range(len(self.portList)):
            self.portList[i] = self.portList[i].device
        self.expControlPath = tk.StringVar()
        self.robControlPath = tk.StringVar()
        if workspace['ReachMaster']['expControlPath'] in self.portList:
            self.expControlPath.set(workspace['ReachMaster']['expControlPath']) 
        else:
            self.expControlPath.set(self.portList[0])
        if workspace['ReachMaster']['robControlPath'] in self.portList:
            self.robControlPath.set(workspace['ReachMaster']['robControlPath']) 
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
        self.main()

    def onQuit(self):
        workspace['ReachMaster']['workspaceFile'] = self.workspaceFile.get()
        answer = tkMessageBox.askyesnocancel("Question", "Save Workspace?")
        if answer == True:
            self.saveWorkspace()
            self.mainActive = False
        elif answer == False:
            self.mainActive = False
        else:
            pass

    def saveWorkspace(self):
        workspacePath = workspace['ReachMaster']['dataDir']+"/workspaces/"
        if not os.path.isdir(workspacePath):
                os.makedirs(workspacePath)
        fn = workspacePath + 'Workspace: ' + str(datetime.datetime.now()) + '.txt'
        with open(fn, 'w') as outfile:
            json.dump(workspace, outfile, indent=4)

    def setup_UI(self):
        tk.Label(text="Data Directory:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=0, column=0)
        tk.Label(textvariable=self.dataDir, bg="white").grid(row=0, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.ddBrowseCallback).grid(row=0, column=2)
        tk.Label(text="Workspace File:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=1, column=0)
        tk.Label(textvariable=self.workspaceFile, bg="white").grid(row=1, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.wfBrowseCallback).grid(row=1, column=2)
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
        global workspace
        self.dataDir.set(tkFileDialog.askdirectory())
        workspace['ReachMaster']['dataDir'] = self.dataDir.get()

    def wfBrowseCallback(self):
        self.workspaceFile.set(tkFileDialog.askopenfilename())
        #open file and read in workspace
        global workspace
        workspace = json_load_byteified(open(self.workspaceFile.get()))
        output_params = workspace['CameraSettings']['output_params']

    def expConnectCallback(self):
        global workspace
        global expController
        workspace['ReachMaster']['expControlPath'] = self.expControlPath.get()
        expController = serial.Serial(workspace['ReachMaster']['expControlPath'],
            workspace['ReachMaster']['serialBaud'],
            timeout=workspace['ReachMaster']['controlTimeout'])
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
            else:
                tkMessageBox.showinfo("Warning", "Experiment is active! Not safe to disconnect.")
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def robConnectCallback(self):
        global workspace
        global robController
        workspace['ReachMaster']['robControlPath'] = self.robControlPath.get()
        robController = serial.Serial(workspace['ReachMaster']['robControlPath'],
            workspace['ReachMaster']['serialBaud'],
            timeout=workspace['ReachMaster']['controlTimeout'])
        time.sleep(2) #wait for controller to wake up
        robController.flushInput()
        robController.write("h")
        response = robController.readline()
        if response:
            self.robControlOn = True
            if workspace['RobotSettings']['calibrationFile'] != 'None':
                self.loadCalibration()                
            if workspace['RobotSettings']['commandType'] != 'None':
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
        dis = workspace['RobotSettings']['dis']#.encode("utf-8")
        pos = workspace['RobotSettings']['pos']#.encode("utf-8")
        xPushDur = workspace['RobotSettings']['xPushDur']#.encode("utf-8")
        xPullDur = workspace['RobotSettings']['xPullDur']#.encode("utf-8")
        yPushDur = workspace['RobotSettings']['yPushDur']#.encode("utf-8")
        yPullDur = workspace['RobotSettings']['yPullDur']#.encode("utf-8")
        zPushDur = workspace['RobotSettings']['zPushDur']#.encode("utf-8")
        zPullDur = workspace['RobotSettings']['zPullDur']#.encode("utf-8")
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
        Ly = workspace['RobotSettings']['Ly']
        Lz = workspace['RobotSettings']['Lz']
        Axx = workspace['RobotSettings']['Axx']
        Ayy = workspace['RobotSettings']['Ayy']
        Azz = workspace['RobotSettings']['Azz']
        x0 = workspace['RobotSettings']['x0']
        y0 = workspace['RobotSettings']['y0']
        z0 = workspace['RobotSettings']['z0']      
        n = 100
        if workspace['RobotSettings']['commandType'] == "sampleContinuous":
            rLow = workspace['RobotSettings']['rLow']
            rHigh = workspace['RobotSettings']['rHigh']
            thetaMag = workspace['RobotSettings']['thetaMag']
            r = rLow + (rHigh-rLow)*np.random.uniform(low=0.0,high=1.0,size=(500*n))**(1.0/3.0)
            thetay = thetaMag*np.random.uniform(low=-1,high=1,size=500*n)
            thetaz = thetaMag*np.random.uniform(low=-1,high=1,size=500*n)
            theta = np.sqrt(thetay**2+thetaz**2)
            r = r[theta<=thetaMag][0:n]
            thetay = thetay[theta<=thetaMag][0:n]
            thetaz = thetaz[theta<=thetaMag][0:n]
        elif workspace['RobotSettings']['commandType'] == "sampleDiscrete":
            rSet,thetaySet,thetazSet = np.loadtxt(workspace['RobotSettings']['commandFile'],\
            skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
            randSample = np.random.choice(range(len(rSet)),replace=True,size=n)
            r = rSet[randSample]
            thetay = thetaySet[randSample]
            thetaz = thetazSet[randSample]
        elif workspace['RobotSettings']['commandType'] == "fromFile":
            r,thetay,thetaz = np.loadtxt(workspace['RobotSettings']['commandFile'],\
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
        workspace['RobotSettings']['x'] = x
        workspace['RobotSettings']['y'] = y
        workspace['RobotSettings']['z'] = z
        workspace['RobotSettings']['r'] = r
        workspace['RobotSettings']['thetay'] = thetay
        workspace['RobotSettings']['thetaz'] = thetaz
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
                camSetRoot = tk.Toplevel(self.window)     
                CameraSettings(camSetRoot)
            else:
                tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")
        else:
            pass

    def expSetCallback(self):
        if not self.expActive:
            if self.expControlOn:
                expSetRoot = tk.Toplevel(self.window)     
                ExperimentSettings(expSetRoot)
            else:
                tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")
        else:
            pass

    def robSetCallback(self):
        if self.robControlOn:
            robSetRoot = tk.Toplevel(self.window)     
            RobotSettings(robSetRoot)
        else:
            tkMessageBox.showinfo("Warning", "Robot Controller not connected.")

    def movRobCallback(self):
        if self.expControlOn:
            expController.write("m")
            if workspace['CameraSettings']['vidMode'] == "TRIALS":
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
            if len(workspace['CameraSettings']['savedPOIs'])>0:
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
                        for i in range(workspace['CameraSettings']['numCams']):
                            obsPOIs.append(np.zeros(len(workspace['CameraSettings']['savedPOIs'][i])))
                            zPOIs.append(0)
                        self.buffer_full = False 
                        self.sensorpath = workspace['ReachMaster']['dataDir'] + "/sensor_data/"
                        self.camerapath = workspace['ReachMaster']['dataDir'] + "/videos/" 
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
                        if workspace['CameraSettings']['vidMode'] == "CONTINUOUS":
                            self.vid_fn = self.camerapath + str(datetime.datetime.now()) + '.mp4' 
                            self.video = WriteGear(
                                output_filename = self.vid_fn,
                                compression_mode = True,
                                logging=False,
                                **output_params)
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
        for i in range(workspace['CameraSettings']['numCams']):
            print('opening camera %s ...' %(i))
            cam = xiapi.Camera(dev_id = i)
            cam.open_device()
            cam.set_imgdataformat(workspace['CameraSettings']['imgdataformat'])
            cam.set_exposure(workspace['CameraSettings']['exposure'])
            cam.set_gain(workspace['CameraSettings']['gain'])
            cam.set_sensor_feature_value(workspace['CameraSettings']['sensor_feature_value'])
            cam.set_gpi_selector(workspace['CameraSettings']['gpi_selector'])
            cam.set_gpi_mode(workspace['CameraSettings']['gpi_mode'])
            cam.set_trigger_source(workspace['CameraSettings']['trigger_source'])
            cam.set_gpo_selector(workspace['CameraSettings']['gpo_selector'])
            cam.set_gpo_mode(workspace['CameraSettings']['gpo_mode'])
            if workspace['CameraSettings']['downsampling'] == "XI_DWN_2x2":
                cam.set_downsampling(workspace['CameraSettings']['downsampling'])
            else:
                cam.set_height(workspace['CameraSettings']['imgHeight'])
                cam.set_width(workspace['CameraSettings']['imgWidth'])
                cam.set_offsetX(workspace['CameraSettings']['offsetX'])
                cam.set_offsetY(workspace['CameraSettings']['offsetY'])
            cam.enable_recent_frame()
            self.camList.append(cam)
            self.camList[i].start_acquisition()
        self.img = xiapi.Image()
        self.imgBuffer = deque()
        self.camerasLoaded = True

    def unloadCameras(self):
        for i in range(workspace['CameraSettings']['numCams']):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camerasLoaded = False

    def acquireBaseline(self):
        if len(workspace['CameraSettings']['savedPOIs'])==0:
            tkMessageBox.showinfo("Warning", "No saved POIs.")
        else:
            global poiMeans
            global poiStds
            global baselineAcquired
            cnt = 0 
            numImgs = int(np.round(float(workspace['ExperimentSettings']['baselineDur'])*\
                float(workspace['CameraSettings']['fps']),decimals=0)) 
            baselinePOIs = []
            for i in range(workspace['CameraSettings']['numCams']):
                baselinePOIs.append(np.zeros(shape = (len(workspace['CameraSettings']['savedPOIs'][i]), numImgs)))
            print("Acquiring baseline...")
            for cnt in range(numImgs):
                time.sleep(0.005)
                expController.write(self.writeControl)
                for i in range(workspace['CameraSettings']['numCams']):
                    self.camList[i].get_image(self.img,timeout = 2000)
                    npImg = self.img.get_image_data_numpy()
                    for j in range(len(workspace['CameraSettings']['savedPOIs'][i])): 
                        baselinePOIs[i][j,cnt] = npImg[workspace['CameraSettings']['savedPOIs'][i][j][1],
                            workspace['CameraSettings']['savedPOIs'][i][j][0]]
            poiMeans = []
            poiStds = []
            for i in range(workspace['CameraSettings']['numCams']):   
                poiMeans.append(np.mean(baselinePOIs[i], axis = 1))             
                poiStds.append(np.std(np.sum(np.square(baselinePOIs[i]-
                    poiMeans[i].reshape(len(workspace['CameraSettings']['savedPOIs'][i]),1)),axis=0)))
            self.baselineAcquired = True
            print("Baseline acquired!")

    def runConintuous(self):
        now = str(int(round(time.time()*1000)))  
        if self.newline[3]=='1': 
            self.lightsOn = 1 
            for i in range(workspace['CameraSettings']['numCams']):
                self.camList[i].get_image(self.img, timeout = 2000)                  
                npImg = self.img.get_image_data_numpy()
                for j in range(len(workspace['CameraSettings']['savedPOIs'][i])): 
                    obsPOIs[i][j] = npImg[workspace['CameraSettings']['savedPOIs'][i][j][1],
                        workspace['CameraSettings']['savedPOIs'][i][j][0]]
                zPOIs[i] = np.round(np.sum(np.square(obsPOIs[i]-poiMeans[i]))/(poiStds[i]+np.finfo(float).eps),decimals=1)
                if i == 0:
                    frame = npImg
                else:
                    frame = np.hstack((frame,npImg))
        else:
            self.lightsOn = 0
            for i in range(workspace['CameraSettings']['numCams']):
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
        if self.newline[1] == 's' and self.newline[2] == '0' and min(zPOIs)>workspace['CameraSettings']['poiThreshold']: 
            self.reachDetected = True  
            self.reachInit = now     
            self.writeControl = 'r'
        elif self.newline[1] == 'e': 
            self.reachDetected = False
            self.writeControl = 's'   
            print(self.newline[0])
        elif self.reachDetected and\
         (int(now)-int(self.reachInit))>workspace['ExperimentSettings']['reachTimeout'] and self.newline[4]=='0':
            self.movRobCallback()

    def runTrials(self): 
        now = str(int(round(time.time()*1000)))   
        if self.newline[3]=='1': 
            self.lightsOn = 1
            for i in range(workspace['CameraSettings']['numCams']):
                self.camList[i].get_image(self.img, timeout = 2000)                  
                npImg = self.img.get_image_data_numpy()
                for j in range(len(workspace['CameraSettings']['savedPOIs'][i])): 
                    obsPOIs[i][j] = npImg[workspace['CameraSettings']['savedPOIs'][i][j][1],
                        workspace['CameraSettings']['savedPOIs'][i][j][0]]
                zPOIs[i] = np.round(np.sum(np.square(obsPOIs[i]-poiMeans[i]))/(poiStds[i]+np.finfo(float).eps),decimals=1)
                self.imgBuffer.append(npImg)
                if len(self.imgBuffer)>workspace['CameraSettings']['numCams']*\
                workspace['ExperimentSettings']['bufferDur']*workspace['CameraSettings']['fps'] and not self.reachDetected:
                    self.imgBuffer.popleft()
        else:
            self.lightsOn = 0
            for i in range(workspace['CameraSettings']['numCams']):
                zPOIs[i] = 0     
        expController.write(self.writeControl) 
        while not expController.in_waiting:
            pass 
        self.newline = expController.readline() 
        expController.flushInput()
        self.outputfile.write(now+" "+self.newline[0:-2:1]+" "+str(min(zPOIs))+"\n")
        self.newline = self.newline.split() 
        if self.newline[1] == 's' and self.newline[2] == '0' and min(zPOIs)>workspace['CameraSettings']['poiThreshold']: 
            self.reachDetected = True  
            self.reachInit = now     
            self.writeControl = 'r'
        elif self.newline[1] == 'e': 
            # serBuf.serialize(self.imgBuffer,self.camerapath,self.newline)
            if not os.path.isdir(self.camerapath):
                os.makedirs(self.camerapath)
            trial_fn = self.camerapath + 'trial: ' + str(self.newline[0]) + '.mp4' 
            self.video = WriteGear(output_filename = trial_fn,compression_mode = True,logging=False,**output_params)
            for i in range(len(self.imgBuffer)/workspace['CameraSettings']['numCams']):
                # frame = cv2.cvtColor(self.imgBuffer[(i+1)*numCams-numCams],cv2.COLOR_BAYER_BG2BGR)
                frame = self.imgBuffer[(i+1)*workspace['CameraSettings']['numCams']-workspace['CameraSettings']['numCams']]
                for f in range(workspace['CameraSettings']['numCams']-1):
                    # frame = np.hstack((frame,cv2.cvtColor(self.imgBuffer[(i+1)*numCams-numCams+f+1],cv2.COLOR_BAYER_BG2BGR))) 
                    frame = np.hstack((frame,self.imgBuffer[(i+1)*\
                        workspace['CameraSettings']['numCams']-workspace['CameraSettings']['numCams']+f+1])) 
                self.video.write(frame)   
            self.video.close()
            self.reachDetected = False
            self.writeControl = 's' 
            self.imgBuffer = deque() 
            print(self.newline[0])
        elif self.reachDetected and (int(now)-int(self.reachInit))>\
        workspace['ExperimentSettings']['reachTimeout'] and self.newline[4]=='0':
            self.movRobCallback()

    def main(self):
        self.mainActive = True
        try:
            while self.mainActive:        
                if self.expActive:
                    try:
                        if workspace['CameraSettings']['vidMode'] == "CONTINUOUS":
                            self.runConintuous()
                        elif workspace['CameraSettings']['vidMode'] == "TRIALS":
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
            if workspace['CameraSettings']['vidMode'] == "CONTINUOUS" and self.baselineAcquired:
                self.video.close()
            if self.camerasLoaded:
                self.unloadCameras()
                self.imgBuffer = deque()
            self.window.destroy()
        except KeyboardInterrupt:
            answer = tkMessageBox.askyesnocancel("Question", "Save Workspace?")
            if answer == True:
                if self.expControlOn:
                    expController.write("e")
                    expController.close()
                if self.robControlOn:
                    robController.write("e")
                    robController.close()
                if workspace['CameraSettings']['vidMode'] == "CONTINUOUS":
                    self.video.close()
                if self.camerasLoaded:
                    self.unloadCameras()
                    self.imgBuffer = deque()
                self.saveWorkspace()
                self.window.destroy()
            elif answer == False:
                if self.expControlOn:
                    expController.write("e")
                    expController.close()
                if self.robControlOn:
                    robController.write("e")
                    robController.close()
                if workspace['CameraSettings']['vidMode'] == "CONTINUOUS":
                    self.video.close()
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
        self.numCams.set(str(workspace['CameraSettings']['numCams']))
        self.fps = tk.StringVar()
        self.fps.set(str(workspace['CameraSettings']['fps']))
        self.exposure = tk.StringVar()
        self.exposure.set(str(workspace['CameraSettings']['exposure']))
        self.gain = tk.StringVar()
        self.gain.set(str(workspace['CameraSettings']['gain']))   
        self.gpi_mode = tk.StringVar()
        self.gpi_mode.set(workspace['CameraSettings']['gpi_mode'])
        self.trigger_source = tk.StringVar()
        self.trigger_source.set(workspace['CameraSettings']['trigger_source'])
        self.gpo_mode = tk.StringVar()
        self.gpo_mode.set(workspace['CameraSettings']['gpo_mode'])
        self.baselineDur = tk.StringVar()
        self.baselineDur.set(str(workspace['ExperimentSettings']['baselineDur']))
        self.bufferDur = tk.StringVar()
        self.bufferDur.set(str(workspace['ExperimentSettings']['bufferDur']))
        self.imgWidth = tk.StringVar()
        self.imgWidth.set(str(workspace['CameraSettings']['imgWidth']))
        self.imgHeight = tk.StringVar()
        self.imgHeight.set(str(workspace['CameraSettings']['imgHeight']))
        self.offsetX = tk.StringVar()
        self.offsetX.set(str(workspace['CameraSettings']['offsetX']))
        self.offsetY = tk.StringVar()
        self.offsetY.set(str(workspace['CameraSettings']['offsetY']))
        self.downsampling = tk.StringVar()
        self.downsampling.set(str(workspace['CameraSettings']['downsampling']))
        self.poiThreshold = tk.StringVar()
        self.poiThreshold.set(str(workspace['CameraSettings']['poiThreshold']))
        self.camsLoaded = False
        self.streaming = False
        self.streamStarted = False
        self.drawSaved = False
        self.addPOIs = False
        self.removePOIs = False
        self.addedPOIs = [[] for _ in range(workspace['CameraSettings']['numCams'])]
        self.savedPOIs = [[] for _ in range(workspace['CameraSettings']['numCams'])] 
        self.capture = False
        self.record = False
        self.imgNum = [1]
        self.vidMode = tk.StringVar()
        self.vidMode.set(workspace['CameraSettings']['vidMode'])
        self.setup_UI()

    def onQuit(self):
        global workspace
        global output_params
        workspace['CameraSettings']['numCams'] = int(self.numCams.get())
        workspace['CameraSettings']['fps'] = int(self.fps.get())
        workspace['CameraSettings']['exposure'] = int(self.exposure.get())
        workspace['CameraSettings']['gain'] = float(self.gain.get()) 
        workspace['ExperimentSettings']['baselineDur'] = float(self.baselineDur.get())
        workspace['ExperimentSettings']['bufferDur'] = float(self.bufferDur.get())
        workspace['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
        workspace['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
        workspace['CameraSettings']['offsetX'] = int(self.offsetX.get())
        workspace['CameraSettings']['offsetY'] = int(self.offsetY.get())
        workspace['CameraSettings']['downsampling'] = self.downsampling.get()
        workspace['CameraSettings']['trigger_source'] = self.trigger_source.get()
        workspace['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
        workspace['CameraSettings']['poiThreshold'] = float(self.poiThreshold.get())
        output_params["-output_dimensions"] = (workspace['CameraSettings']['numCams']*
            workspace['CameraSettings']['imgWidth'],workspace['CameraSettings']['imgHeight'])
        workspace['CameraSettings']['vidMode'] = self.vidMode.get()
        if workspace['CameraSettings']['vidMode'] == "CONTINUOUS":
            output_params["-crf"] = 28
            self.contModeWrite()
        elif workspace['CameraSettings']['vidMode'] == "TRIALS":
            output_params["-crf"] = 23
        workspace['CameraSettings']['output_params'] = output_params
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
        tk.Label(self.window,text="Image Width (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, sticky='W')   
        tk.Entry(self.window,textvariable=self.imgWidth,width=17).grid(row=7, column=1)
        tk.Label(self.window,text="Image Height (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, sticky='W')   
        tk.Entry(self.window,textvariable=self.imgHeight,width=17).grid(row=8, column=1)
        tk.Label(self.window,text="Image X Offest (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, sticky='W')   
        tk.Entry(self.window,textvariable=self.offsetX,width=17).grid(row=9, column=1)
        tk.Label(self.window,text="Image Y Offset (pix):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, sticky='W')   
        tk.Entry(self.window,textvariable=self.offsetY,width=17).grid(row=10, column=1)
        tk.Label(self.window,text="Downsampling:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, sticky='W') 
        self.downsamplingMenu = tk.OptionMenu(self.window,self.downsampling,
            "XI_DWN_1x1",
            "XI_DWN_2x2")
        self.downsamplingMenu.configure(width=12,anchor="w")
        self.downsamplingMenu.grid(row=11, column=1)
        tk.Button(self.window,text="Start Streaming",font='Arial 10 bold',width=14,command=self.startStreamCallback).grid(row=12, column=0,sticky="e")
        tk.Button(self.window,text="Stop Streaming",font='Arial 10 bold',width=14,command=self.stopStreamCallback).grid(row=13, column=0,sticky="e")
        tk.Button(self.window,text="Load POIs",font='Arial 10 bold',width=14,command=self.loadPOIsCallback).grid(row=12, column=1)
        tk.Button(self.window,text="Save POIs",font='Arial 10 bold',width=14,command=self.savePOIsCallback).grid(row=13, column=1)
        tk.Button(self.window,text="Add POIs",font='Arial 10 bold',width=14,command=self.addPOIsCallback).grid(row=12, column=2)
        tk.Button(self.window,text="Remove POIs",font='Arial 10 bold',width=14,command=self.removePOIsCallback).grid(row=13, column=2)
        tk.Button(self.window,text="Capture Image",font='Arial 10 bold',width=14,command=self.captureImgCallback).grid(row=14, column=0,sticky="e")
        tk.Button(self.window,text="Start Record",font='Arial 10 bold',width=14,command=self.startRecCallback).grid(row=14, column=1)
        tk.Button(self.window,text="Stop Record",font='Arial 10 bold',width=14,command=self.stopRecCallback).grid(row=14, column=2)        
        tk.Label(self.window,text="POI Threshold (stdev):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, sticky='W')   
        tk.Entry(self.window,textvariable=self.poiThreshold,width=17).grid(row=15, column=1)
        tk.Label(self.window,text="Video Mode:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=16, sticky='W')   
        self.vidModeMenu = tk.OptionMenu(self.window,self.vidMode,
            "CONTINUOUS",
            "TRIALS")
        self.vidModeMenu.configure(width=12,anchor="w")
        self.vidModeMenu.grid(row=16, column=1)

    def startStreamCallback(self):
        if not self.streamStarted:
            global workspace
            workspace['CameraSettings']['numCams'] = int(self.numCams.get())
            workspace['CameraSettings']['fps'] = int(self.fps.get())
            workspace['CameraSettings']['exposure'] = int(self.exposure.get())
            workspace['CameraSettings']['gain'] = float(self.gain.get())   
            workspace['CameraSettings']['trigger_source'] = self.trigger_source.get()
            workspace['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            workspace['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
            workspace['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
            workspace['CameraSettings']['offsetX'] = int(self.offsetX.get())
            workspace['CameraSettings']['offsetY'] = int(self.offsetY.get())  
            workspace['CameraSettings']['downsampling'] = self.downsampling.get()
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
        for i in range(workspace['CameraSettings']['numCams']):
            print('opening camera %s ...' %(i))
            cam = xiapi.Camera(dev_id = i)
            cam.open_device()
            cam.set_imgdataformat(workspace['CameraSettings']['imgdataformat'])
            cam.set_exposure(workspace['CameraSettings']['exposure'])
            cam.set_gain(workspace['CameraSettings']['gain'])
            cam.set_sensor_feature_value(workspace['CameraSettings']['sensor_feature_value'])
            cam.set_gpi_selector(workspace['CameraSettings']['gpi_selector'])
            # cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            # cam.set_framerate(fps)
            cam.set_gpi_mode("XI_GPI_TRIGGER")
            cam.set_trigger_source(workspace['CameraSettings']['trigger_source'])
            cam.set_gpo_selector(workspace['CameraSettings']['gpo_selector'])
            cam.set_gpo_mode(workspace['CameraSettings']['gpo_mode'])
            if workspace['CameraSettings']['downsampling'] == "XI_DWN_2x2":
                cam.set_downsampling(workspace['CameraSettings']['downsampling'])
            else:
                widthIncrement = cam.get_width_increment()
                heightIncrement = cam.get_height_increment()
                if (workspace['CameraSettings']['imgWidth']%widthIncrement)!=0:
                    tkMessageBox.showinfo("Warning", "Image width not divisible by "+str(widthIncrement))
                    break
                elif (workspace['CameraSettings']['imgHeight']%heightIncrement)!=0:
                    tkMessageBox.showinfo("Warning", "Image height not divisible by "+str(heightIncrement))
                    break
                elif (workspace['CameraSettings']['imgWidth']+workspace['CameraSettings']['offsetX'])>1280:
                    tkMessageBox.showinfo("Warning", "Image width + x offset > 1280") 
                    break
                elif (workspace['CameraSettings']['imgHeight']+workspace['CameraSettings']['offsetY'])>1024:
                    tkMessageBox.showinfo("Warning", "Image height + y offset > 1024") 
                    break
                else:
                    cam.set_height(workspace['CameraSettings']['imgHeight'])
                    cam.set_width(workspace['CameraSettings']['imgWidth'])
                    cam.set_offsetX(workspace['CameraSettings']['offsetX'])
                    cam.set_offsetY(workspace['CameraSettings']['offsetY'])                
            self.camList.append(cam)
            self.camList[i].start_acquisition()   
        self.imgBuffer = deque()
        self.camsLoaded = True        

    def unloadCameras(self):
        for i in range(workspace['CameraSettings']['numCams']):
            print('Stopping acquisition for camera %d ...' %i)
            self.camList[i].stop_acquisition()
            self.camList[i].close_device()
        self.camsLoaded = False

    def startStream(self):
        if not self.streamStarted:
            self.buffer_full = False 
            self.camWindows = [0 for _ in range(workspace['CameraSettings']['numCams'])]
            for i in range(workspace['CameraSettings']['numCams']):
                self.camWindows[i] = tk.Toplevel(self.window)
                self.camWindows[i].title("Camera"+str(i))
                self.camWindows[i].protocol("WM_DELETE_WINDOW", self.stopStream)
                self.camWindows[i].canvas = tk.Canvas(self.camWindows[i], 
                    width = workspace['CameraSettings']['imgWidth'], 
                    height = workspace['CameraSettings']['imgHeight'])
                self.camWindows[i].canvas.grid(row=0,column= 0)            
            self.streamStarted = True
        self.delay = int(np.round(1.0/float(workspace['CameraSettings']['fps'])*1000.0))
        self.streaming = True
        self.refresh()

    def stopStream(self):
        self.streaming = False 
        self.streamStarted = False 
        self.poiActive = False  
        self.drawSaved = False    
        for i in range(workspace['CameraSettings']['numCams']):
            self.camWindows[i].destroy()
        self.unloadCameras()

    def refresh(self):
        if self.streaming:
            expController.write("t")
            now = str(int(round(time.time()*1000)))
            npImg = np.zeros(shape = (workspace['CameraSettings']['imgHeight'], workspace['CameraSettings']['imgWidth'])) 
            img = xiapi.Image()
            self.photoImg = [0 for _ in range(workspace['CameraSettings']['numCams'])]
            for i in range(workspace['CameraSettings']['numCams']):
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
                self.calipath = workspace['ReachMaster']['dataDir'] + "/calibration_images/"
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
            if len(workspace['CameraSettings']['savedPOIs'])>0:
                self.savedPOIs = workspace['CameraSettings']['savedPOIs']
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
        global workspace
        baselineAcquired = False
        for i in range(workspace['CameraSettings']['numCams']):
            self.savedPOIs[i] += self.addedPOIs[i] 
        workspace['CameraSettings']['savedPOIs'] = self.savedPOIs 
        self.addedPOIs = [[] for _ in range(workspace['CameraSettings']['numCams'])]

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
            global workspace
            workspace['CameraSettings']['numCams'] = int(self.numCams.get())
            workspace['CameraSettings']['fps'] = int(self.fps.get())
            workspace['CameraSettings']['exposure'] = int(self.exposure.get())
            workspace['CameraSettings']['gain'] = float(self.gain.get())   
            workspace['CameraSettings']['trigger_source'] = self.trigger_source.get()
            workspace['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            workspace['CameraSettings']['imgWidth'] = int(self.imgWidth.get())
            workspace['CameraSettings']['imgHeight'] = int(self.imgHeight.get())
            workspace['CameraSettings']['offsetX'] = int(self.offsetX.get())
            workspace['CameraSettings']['offsetY'] = int(self.offsetY.get())  
            self.record = True
            self.loadCameras() 
            self.calipath = workspace['ReachMaster']['dataDir'] + "/calibration_videos/"
            if not os.path.isdir(self.calipath):
                os.makedirs(self.calipath)
            self.vid_fn = self.calipath + str(datetime.datetime.now()) + '.mp4' 
            self.video = WriteGear(
                output_filename = self.vid_fn,
                compression_mode = True,
                logging=False,
                **output_params)
            self.delay = int(np.round(1.0/float(workspace['CameraSettings']['fps'])*1000.0))
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
            npImg = np.zeros(shape = (workspace['CameraSettings']['imgHeight'], workspace['CameraSettings']['imgWidth'])) 
            img = xiapi.Image()            
            for i in range(workspace['CameraSettings']['numCams']):
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


class ExperimentSettings:

    def __init__(self, window):
        self.window = window
        self.window.title("Experiment Settings") 
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit) 
        self.lightsOnDur = tk.StringVar()
        self.lightsOnDur.set(str(workspace['ExperimentSettings']['lightsOnDur']))
        self.lightsOffDur = tk.StringVar()
        self.lightsOffDur.set(str(workspace['ExperimentSettings']['lightsOffDur']))
        self.rewardWinDur = tk.StringVar()
        self.rewardWinDur.set(str(workspace['ExperimentSettings']['rewardWinDur']))
        self.maxRewards = tk.StringVar()
        self.maxRewards.set(str(workspace['ExperimentSettings']['maxRewards']))
        self.solenoidOpenDur = tk.StringVar()
        self.solenoidOpenDur.set(str(workspace['ExperimentSettings']['solenoidOpenDur']))
        self.solenoidBounceDur = tk.StringVar()
        self.solenoidBounceDur.set(str(workspace['ExperimentSettings']['solenoidBounceDur']))
        self.flushDur = tk.StringVar()
        self.flushDur.set(str(workspace['ExperimentSettings']['flushDur']))
        self.reachDelay = tk.StringVar()
        self.reachDelay.set(str(workspace['ExperimentSettings']['reachDelay']))
        self.setup_UI()

    def onQuit(self):
        global workspace
        workspace['ExperimentSettings']['lightsOnDur'] = int(self.lightsOnDur.get())
        workspace['ExperimentSettings']['lightsOffDur'] = int(self.lightsOffDur.get())
        workspace['ExperimentSettings']['rewardWinDur'] = int(self.rewardWinDur.get())
        workspace['ExperimentSettings']['maxRewards'] = int(self.maxRewards.get()) 
        workspace['ExperimentSettings']['solenoidOpenDur'] = int(self.solenoidOpenDur.get())
        workspace['ExperimentSettings']['solenoidBounceDur'] = int(self.solenoidBounceDur.get())
        workspace['ExperimentSettings']['flushDur'] = int(self.flushDur.get())
        workspace['ExperimentSettings']['reachDelay'] = int(self.reachDelay.get())
        self.window.destroy()

    def setup_UI(self):
        tk.Button(self.window,text="Read All",font='Arial 10 bold',width=14,command=self.readAllCallback).grid(row=0,column=2)
        tk.Button(self.window,text="Write All",font='Arial 10 bold',width=14,command=self.writeAllCallback).grid(row=0,column=3)
        tk.Label(self.window,text="Lights On (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, column=0)   
        tk.Entry(self.window,textvariable=self.lightsOnDur,width=17).grid(row=1, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.lightsOnDurRead).grid(row=1, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.lightsOnDurWrite).grid(row=1, column=3)
        tk.Label(self.window,text="Lights Off (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=0)   
        tk.Entry(self.window,textvariable=self.lightsOffDur,width=17).grid(row=2, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.lightsOffDurRead).grid(row=2, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.lightsOffDurWrite).grid(row=2, column=3)
        tk.Label(self.window,text="Reward Window (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=0)   
        tk.Entry(self.window,textvariable=self.rewardWinDur,width=17).grid(row=3, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.rewardWinDurRead).grid(row=3, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.rewardWinDurWrite).grid(row=3, column=3)
        tk.Label(self.window,text="# Rewards/Trial:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=0)   
        tk.Entry(self.window,textvariable=self.maxRewards,width=17).grid(row=4, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.maxRewardsRead).grid(row=4, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.maxRewardsWrite).grid(row=4, column=3)
        tk.Label(self.window,text="Solenoid Open (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=0)   
        tk.Entry(self.window,textvariable=self.solenoidOpenDur,width=17).grid(row=5, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.solenoidOpenDurRead).grid(row=5, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.solenoidOpenDurWrite).grid(row=5, column=3)
        tk.Label(self.window,text="Solenoid Bounce (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=0)   
        tk.Entry(self.window,textvariable=self.solenoidBounceDur,width=17).grid(row=6, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.solenoidBounceDurRead).grid(row=6, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.solenoidBounceDurWrite).grid(row=6, column=3)
        tk.Label(self.window,text="Flush (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=0)   
        tk.Entry(self.window,textvariable=self.flushDur,width=17).grid(row=7, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.flushDurRead).grid(row=7, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.flushDurWrite).grid(row=7, column=3)
        tk.Label(self.window,text="Reach Delay (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=0)   
        tk.Entry(self.window,textvariable=self.reachDelay,width=17).grid(row=8, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.reachDelayRead).grid(row=8, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.reachDelayWrite).grid(row=8, column=3)

    def lightsOnDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("lightsOnDur\n")
            self.lightsOnDur.set(expController.readline()[:-2])

    def lightsOnDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("lightsOnDur\n")
            if expController.read() == "v":
                expController.write(self.lightsOnDur.get()+"\n")

    def lightsOffDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("lightsOffDur\n")
            self.lightsOffDur.set(expController.readline()[:-2])

    def lightsOffDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("lightsOffDur\n")
            if expController.read() == "v":
                expController.write(self.lightsOffDur.get()+"\n")

    def rewardWinDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("rewardWinDur\n")
            self.rewardWinDur.set(expController.readline()[:-2])

    def rewardWinDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("rewardWinDur\n")
            if expController.read() == "v":
                expController.write(self.rewardWinDur.get()+"\n")

    def maxRewardsRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("maxRewards\n")
            self.maxRewards.set(expController.readline()[:-2])

    def maxRewardsWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("maxRewards\n")
            if expController.read() == "v":
                expController.write(self.maxRewards.get()+"\n")

    def solenoidOpenDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("solenoidOpenDur\n")
            self.solenoidOpenDur.set(expController.readline()[:-2])

    def solenoidOpenDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("SolenoidOpenDur\n")
            if expController.read() == "v":
                expController.write(self.solenoidOpenDur.get()+"\n")

    def solenoidBounceDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("solenoidBounceDur\n")
            self.solenoidBounceDur.set(expController.readline()[:-2])

    def solenoidBounceDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("solenoidBounceDur\n")
            if expController.read() == "v":
                expController.write(self.solenoidBounceDur.get()+"\n")

    def flushDurRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("rewardWinDur\n")
            self.flushDur.set(expController.readline()[:-2])

    def flushDurWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("flushDur\n")
            if expController.read() == "v":
                expController.write(self.flushDur.get()+"\n")

    def reachDelayRead(self):
        expController.write("g")
        if expController.read() == "g":
            expController.write("reachDelay\n")
            self.reachDelay.set(expController.readline()[:-2])

    def reachDelayWrite(self):
        expController.write("v")
        if expController.read() == "v":
            expController.write("reachDelay\n")
            if expController.read() == "v":
                expController.write(self.reachDelay.get()+"\n")

    def readAllCallback(self):
        self.flushDurRead()
        self.solenoidBounceDurRead()
        self.solenoidOpenDurRead()
        self.maxRewardsRead()
        self.rewardWinDurRead()
        self.lightsOffDurRead()
        self.lightsOnDurRead()
        self.reachDelayRead()


    def writeAllCallback(self):
        self.flushDurWrite()
        self.solenoidBounceDurWrite()
        self.solenoidOpenDurWrite()
        self.maxRewardsWrite()
        self.rewardWinDurWrite()
        self.lightsOffDurWrite()
        self.lightsOnDurWrite()
        self.reachDelayWrite()

class RobotSettings:

    def __init__(self, window):
        self.window = window
        self.window.title("Robot Settings")   
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.alpha = tk.StringVar()
        self.alpha.set(str(workspace['RobotSettings']['alpha']))
        self.tol = tk.StringVar()
        self.tol.set(str(workspace['RobotSettings']['tol']))
        self.period = tk.StringVar()
        self.period.set(str(workspace['RobotSettings']['period']))
        self.offDur = tk.StringVar()
        self.offDur.set(str(workspace['RobotSettings']['offDur']))
        self.numTol = tk.StringVar()
        self.numTol.set(str(workspace['RobotSettings']['numTol']))
        self.xPushWt = tk.StringVar()
        self.xPushWt.set(str(workspace['RobotSettings']['xPushWt']))
        self.xPullWt = tk.StringVar()
        self.xPullWt.set(str(workspace['RobotSettings']['xPullWt']))
        self.yPushWt = tk.StringVar()
        self.yPushWt.set(str(workspace['RobotSettings']['yPushWt']))
        self.yPullWt = tk.StringVar()
        self.yPullWt.set(str(workspace['RobotSettings']['yPullWt']))
        self.zPushWt = tk.StringVar()
        self.zPushWt.set(str(workspace['RobotSettings']['zPushWt']))
        self.zPullWt = tk.StringVar()
        self.zPullWt.set(str(workspace['RobotSettings']['zPullWt']))
        self.RZx = tk.StringVar()
        self.RZx.set(str(workspace['RobotSettings']['RZx']))
        self.RZy_low = tk.StringVar()
        self.RZy_low.set(str(workspace['RobotSettings']['RZy_low']))
        self.RZy_high = tk.StringVar()
        self.RZy_high.set(str(workspace['RobotSettings']['RZy_high']))
        self.RZz_low = tk.StringVar()
        self.RZz_low.set(str(workspace['RobotSettings']['RZz_low']))
        self.RZz_high = tk.StringVar()
        self.RZz_high.set(str(workspace['RobotSettings']['RZz_high']))
        self.calibrationFile = tk.StringVar()
        self.calibrationFile.set(str(workspace['RobotSettings']['calibrationFile']))
        self.dis = workspace['RobotSettings']['dis']
        self.pos = workspace['RobotSettings']['pos']
        self.xPushDur = workspace['RobotSettings']['xPushDur']
        self.xPullDur = workspace['RobotSettings']['xPullDur']
        self.yPushDur = workspace['RobotSettings']['yPushDur']
        self.yPullDur = workspace['RobotSettings']['yPullDur']
        self.zPushDur = workspace['RobotSettings']['zPushDur']
        self.zPullDur = workspace['RobotSettings']['zPullDur'] 
        self.commandFile = tk.StringVar()
        self.commandFile.set(str(workspace['RobotSettings']['commandFile']))
        self.commandType = tk.StringVar()
        self.commandType.set(workspace['RobotSettings']['commandType'])
        self.Ly = tk.StringVar()
        self.Ly.set(str(workspace["RobotSettings"]["Ly"]))
        self.Lz = tk.StringVar()
        self.Lz.set(str(workspace["RobotSettings"]["Lz"]))
        self.Axx = tk.StringVar()
        self.Axx.set(str(workspace["RobotSettings"]["Axx"]))
        self.Ayy = tk.StringVar()
        self.Ayy.set(str(workspace["RobotSettings"]["Ayy"]))
        self.Azz = tk.StringVar()
        self.Azz.set(str(workspace["RobotSettings"]["Azz"]))
        self.x0 = tk.StringVar()
        self.x0.set(str(workspace["RobotSettings"]["x0"]))
        self.y0 = tk.StringVar()
        self.y0.set(str(workspace["RobotSettings"]["y0"]))
        self.z0 = tk.StringVar()
        self.z0.set(str(workspace["RobotSettings"]["z0"]))
        self.rLow = tk.StringVar()
        self.rLow.set(str(workspace["RobotSettings"]["rLow"]))
        self.rHigh = tk.StringVar()
        self.rHigh.set(str(workspace["RobotSettings"]["rHigh"]))
        self.thetaMag = tk.StringVar()
        self.thetaMag.set(str(workspace["RobotSettings"]["thetaMag"]))
        self.r =  workspace["RobotSettings"]["rCommandPos"]
        self.thetay =  workspace["RobotSettings"]["thetayCommandPos"]
        self.thetaz =  workspace["RobotSettings"]["thetazCommandPos"]
        self.x =  workspace["RobotSettings"]["xCommandPos"]
        self.y =  workspace["RobotSettings"]["yCommandPos"]
        self.z =  workspace["RobotSettings"]["zCommandPos"]
        self.setup_UI()

    def onQuit(self):
        global workspace
        workspace['RobotSettings']['alpha'] = float(self.alpha.get())
        workspace['RobotSettings']['tol'] = float(self.tol.get())
        workspace['RobotSettings']['period'] = float(self.period.get())
        workspace['RobotSettings']['offDur'] = int(self.offDur.get()) 
        workspace['RobotSettings']['numTol'] = int(self.numTol.get())
        workspace['RobotSettings']['xPushWt'] = float(self.xPushWt.get())
        workspace['RobotSettings']['xPullWt'] = float(self.xPullWt.get())
        workspace['RobotSettings']['yPushWt'] = float(self.yPushWt.get())
        workspace['RobotSettings']['yPullWt'] = float(self.yPullWt.get())
        workspace['RobotSettings']['zPushWt'] = float(self.zPushWt.get())
        workspace['RobotSettings']['zPullWt'] = float(self.zPullWt.get())
        workspace['RobotSettings']['RZx'] = int(self.RZx.get())
        workspace['RobotSettings']['RZy_low'] = int(self.RZy_low.get())
        workspace['RobotSettings']['RZy_high'] = int(self.RZy_high.get())
        workspace['RobotSettings']['RZz_low'] = int(self.RZz_low.get())
        workspace['RobotSettings']['RZz_high'] = int(self.RZz_high.get())
        workspace["RobotSettings"]["calibrationFile"] = self.calibrationFile.get()
        workspace["RobotSettings"]["dis"] = self.dis
        workspace["RobotSettings"]["pos"] = self.pos
        workspace["RobotSettings"]["xPushDur"] = self.xPushDur
        workspace["RobotSettings"]["xPullDur"] = self.xPullDur
        workspace["RobotSettings"]["yPushDur"] = self.yPushDur
        workspace["RobotSettings"]["yPullDur"] = self.yPullDur
        workspace["RobotSettings"]["zPushDur"] = self.zPushDur
        workspace["RobotSettings"]["zPullDur"] = self.zPullDur
        workspace["RobotSettings"]["commandFile"] = self.commandFile.get()
        workspace["RobotSettings"]["commandType"] = self.commandType.get()        
        workspace["RobotSettings"]["Ly"] = int(self.Ly.get())
        workspace["RobotSettings"]["Lz"] = int(self.Lz.get())
        workspace["RobotSettings"]["Axx"] = int(self.Axx.get())
        workspace["RobotSettings"]["Ayy"] = int(self.Ayy.get())
        workspace["RobotSettings"]["Azz"] = int(self.Azz.get())
        workspace["RobotSettings"]["x0"] = int(self.x0.get())
        workspace["RobotSettings"]["y0"] = int(self.y0.get())
        workspace["RobotSettings"]["z0"] = int(self.z0.get())
        workspace["RobotSettings"]["rLow"] = int(self.rLow.get())
        workspace["RobotSettings"]["rHigh"] = int(self.rHigh.get())
        workspace["RobotSettings"]["thetaMag"] = float(self.thetaMag.get())
        workspace["RobotSettings"]["xCommandPos"] = self.x
        workspace["RobotSettings"]["yCommandPos"] = self.y
        workspace["RobotSettings"]["zCommandPos"] = self.z
        workspace["RobotSettings"]["rCommandPos"] = self.r
        workspace["RobotSettings"]["thetayCommandPos"] = self.thetay
        workspace["RobotSettings"]["thetazCommandPos"] = self.thetaz        
        self.window.destroy()

    def setup_UI(self):
        tk.Button(self.window,text="Read All",font='Arial 10 bold',width=14,command=self.readAllCallback).grid(row=0,column=2)
        tk.Button(self.window,text="Write All",font='Arial 10 bold',width=14,command=self.writeAllCallback).grid(row=0,column=3)
        tk.Label(self.window,text="Position Smoothing:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, column=0)   
        tk.Entry(self.window,textvariable=self.alpha,width=17).grid(row=1, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.alphaRead).grid(row=1, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.alphaWrite).grid(row=1, column=3)
        tk.Label(self.window,text="Valve Period:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=0)   
        tk.Entry(self.window,textvariable=self.period,width=17).grid(row=2, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.periodRead).grid(row=2, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.periodWrite).grid(row=2, column=3)
        tk.Label(self.window,text="Off Duration:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=0)   
        tk.Entry(self.window,textvariable=self.offDur,width=17).grid(row=3, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.offDurRead).grid(row=3, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.offDurWrite).grid(row=3, column=3)
        tk.Label(self.window,text="Converge Tolerance:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=0)   
        tk.Entry(self.window,textvariable=self.tol,width=17).grid(row=4, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.tolRead).grid(row=4, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.tolWrite).grid(row=4, column=3)
        tk.Label(self.window,text="# w/in Tolerance:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=0)   
        tk.Entry(self.window,textvariable=self.numTol,width=17).grid(row=5, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.numTolRead).grid(row=5, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.numTolWrite).grid(row=5, column=3)
        tk.Label(self.window,text="Push Weight (x):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=0)   
        tk.Entry(self.window,textvariable=self.xPushWt,width=17).grid(row=6, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.xPushWtRead).grid(row=6, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.xPushWtWrite).grid(row=6, column=3)
        tk.Label(self.window,text="Pull Weight (x):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=0)   
        tk.Entry(self.window,textvariable=self.xPullWt,width=17).grid(row=7, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.xPullWtRead).grid(row=7, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.xPullWtWrite).grid(row=7, column=3)
        tk.Label(self.window,text="Push Weight (y):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=0)   
        tk.Entry(self.window,textvariable=self.yPushWt,width=17).grid(row=8, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.yPushWtRead).grid(row=8, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.yPushWtWrite).grid(row=8, column=3)
        tk.Label(self.window,text="Pull Weight (y):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, column=0)   
        tk.Entry(self.window,textvariable=self.yPullWt,width=17).grid(row=9, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.yPullWtRead).grid(row=9, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.yPullWtWrite).grid(row=9, column=3)
        tk.Label(self.window,text="Push Weight (z):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, column=0)   
        tk.Entry(self.window,textvariable=self.zPushWt,width=17).grid(row=10, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.zPushWtRead).grid(row=10, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.zPushWtWrite).grid(row=10, column=3)
        tk.Label(self.window,text="Pull Weight (z):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, column=0)   
        tk.Entry(self.window,textvariable=self.zPullWt,width=17).grid(row=11, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.zPullWtRead).grid(row=11, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.zPullWtWrite).grid(row=11, column=3)
        tk.Label(self.window,text="Reward Zone (xmin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=12, column=0)   
        tk.Entry(self.window,textvariable=self.RZx,width=17).grid(row=12, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.RZxRead).grid(row=12, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.RZxWrite).grid(row=12, column=3)
        tk.Label(self.window,text="Reward Zone (ymin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=13, column=0)   
        tk.Entry(self.window,textvariable=self.RZy_low,width=17).grid(row=13, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.RZy_lowRead).grid(row=13, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.RZy_lowWrite).grid(row=13, column=3)
        tk.Label(self.window,text="Reward Zone (ymax):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=14, column=0)   
        tk.Entry(self.window,textvariable=self.RZy_high,width=17).grid(row=14, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.RZy_highRead).grid(row=14, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.RZy_highWrite).grid(row=14, column=3)
        tk.Label(self.window,text="Reward Zone (zmin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, column=0)   
        tk.Entry(self.window,textvariable=self.RZz_low,width=17).grid(row=15, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.RZz_lowRead).grid(row=15, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.RZz_lowWrite).grid(row=15, column=3)
        tk.Label(self.window,text="Reward Zone (zmax):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=16, column=0)   
        tk.Entry(self.window,textvariable=self.RZz_high,width=17).grid(row=16, column=1)
        tk.Button(self.window,text="Read",font='Arial 10 bold',width=14,command=self.RZz_highRead).grid(row=16, column=2)
        tk.Button(self.window,text="Write",font='Arial 10 bold',width=14,command=self.RZz_highWrite).grid(row=16, column=3)
        tk.Button(self.window,text="Run Calibration",font='Arial 10 bold',width=14,command=self.runCalCallback).grid(row=1, column=6)
        tk.Label(self.window,text="Calibration File:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=4)
        tk.Label(self.window,textvariable=self.calibrationFile, bg="white").grid(row=2, column=5)
        tk.Button(self.window,text="Browse", font='Arial 10 bold',width=14, command=self.calBrowseCallback).grid(row=2, column=6)
        tk.Button(self.window,text="Load", font='Arial 10 bold',width=14, command=self.calLoadCallback).grid(row=2, column=7)
        tk.Label(self.window,text="Command File:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=4)
        tk.Label(self.window,textvariable=self.commandFile, bg="white").grid(row=3, column=5)
        tk.Button(self.window,text="Browse", font='Arial 10 bold',width=14, command=self.comBrowseCallback).grid(row=3, column=6)
        tk.Label(self.window,text="Command Type:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=4)
        self.comTypeMenu = tk.OptionMenu(self.window,self.commandType,
            "sampleContinuous",
            "sampleDiscrete",
            "fromFile")
        self.comTypeMenu.configure(width=23)
        self.comTypeMenu.grid(row=4, column=5)
        tk.Button(self.window,text="Load", font='Arial 10 bold',width=14, command=self.loadComTypeCallback).grid(row=4, column=6)
        tk.Label(self.window,text="rLow:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=4)   
        tk.Entry(self.window,textvariable=self.rLow,width=17).grid(row=5, column=5)
        tk.Label(self.window,text="rHigh:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=4)   
        tk.Entry(self.window,textvariable=self.rHigh,width=17).grid(row=6, column=5)
        tk.Label(self.window,text="thetaMag:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=4)   
        tk.Entry(self.window,textvariable=self.thetaMag,width=17).grid(row=7, column=5)
        tk.Label(self.window,text="Ly:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=4)   
        tk.Entry(self.window,textvariable=self.Ly,width=17).grid(row=8, column=5)
        tk.Label(self.window,text="Lz:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, column=4)   
        tk.Entry(self.window,textvariable=self.Lz,width=17).grid(row=9, column=5)
        tk.Label(self.window,text="Axx:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, column=4)   
        tk.Entry(self.window,textvariable=self.Axx,width=17).grid(row=10, column=5)
        tk.Label(self.window,text="Ayy:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, column=4)   
        tk.Entry(self.window,textvariable=self.Ayy,width=17).grid(row=11, column=5)
        tk.Label(self.window,text="Azz:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=12, column=4)   
        tk.Entry(self.window,textvariable=self.Azz,width=17).grid(row=12, column=5)
        tk.Label(self.window,text="x0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=13, column=4)   
        tk.Entry(self.window,textvariable=self.x0,width=17).grid(row=13, column=5)
        tk.Label(self.window,text="y0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=14, column=4)   
        tk.Entry(self.window,textvariable=self.y0,width=17).grid(row=14, column=5)
        tk.Label(self.window,text="z0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, column=4)   
        tk.Entry(self.window,textvariable=self.z0,width=17).grid(row=15, column=5)

    def alphaRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("alpha\n")
            self.alpha.set(robController.readline()[:-2])

    def alphaWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("alpha\n")
            if robController.read() == "v":
                robController.write(self.alpha.get()+"\n")

    def tolRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("tol\n")
            self.tol.set(robController.readline()[:-2])

    def tolWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("tol\n")
            if robController.read() == "v":
                robController.write(self.tol.get()+"\n")

    def periodRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("period\n")
            self.period.set(robController.readline()[:-2])

    def periodWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("period\n")
            if robController.read() == "v":
                robController.write(self.period.get()+"\n")

    def offDurRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("offDur\n")
            self.offDur.set(robController.readline()[:-2])

    def offDurWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("offDur\n")
            if robController.read() == "v":
                robController.write(self.offDur.get()+"\n")

    def numTolRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("numTol\n")
            self.numTol.set(robController.readline()[:-2])

    def numTolWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("numTol\n")
            if robController.read() == "v":
                robController.write(self.numTol.get()+"\n")

    def xPushWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("xPushWt\n")
            self.xPushWt.set(robController.readline()[:-2])

    def xPushWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("xPushWt\n")
            if robController.read() == "v":
                robController.write(self.xPushWt.get()+"\n")

    def xPullWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("xPullWt\n")
            self.xPullWt.set(robController.readline()[:-2])

    def xPullWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("xPullWt\n")
            if robController.read() == "v":
                robController.write(self.xPullWt.get()+"\n")


    def yPushWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("yPushWt\n")
            self.yPushWt.set(robController.readline()[:-2])

    def yPushWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("yPushWt\n")
            if robController.read() == "v":
                robController.write(self.yPushWt.get()+"\n")

    def yPullWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("yPullWt\n")
            self.yPullWt.set(robController.readline()[:-2])

    def yPullWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("yPullWt\n")
            if robController.read() == "v":
                robController.write(self.yPullWt.get()+"\n")

    def zPushWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("zPushWt\n")
            self.zPushWt.set(robController.readline()[:-2])

    def zPushWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("zPushWt\n")
            if robController.read() == "v":
                robController.write(self.zPushWt.get()+"\n")

    def zPullWtRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("zPullWt\n")
            self.zPullWt.set(robController.readline()[:-2])

    def zPullWtWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("zPullWt\n")
            if robController.read() == "v":
                robController.write(self.zPullWt.get()+"\n")

    def RZxRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("RZx\n")
            self.RZx.set(robController.readline()[:-2])

    def RZxWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("RZx\n")
            if robController.read() == "v":
                robController.write(self.RZx.get()+"\n")

    def RZy_lowRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("RZy_low\n")
            self.RZy_low.set(robController.readline()[:-2])

    def RZy_lowWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("RZy_low\n")
            if robController.read() == "v":
                robController.write(self.RZy_low.get()+"\n")

    def RZy_highRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("RZy_high\n")
            self.RZy_high.set(robController.readline()[:-2])

    def RZy_highWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("RZy_high\n")
            if robController.read() == "v":
                robController.write(self.RZy_high.get()+"\n")

    def RZz_lowRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("RZz_low\n")
            self.RZz_low.set(robController.readline()[:-2])

    def RZz_lowWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("RZz_low\n")
            if robController.read() == "v":
                robController.write(self.RZz_low.get()+"\n")

    def RZz_highRead(self):
        robController.write("g")
        if robController.read() == "g":
            robController.write("RZz_high\n")
            self.RZz_high.set(robController.readline()[:-2])

    def RZz_highWrite(self):
        robController.write("v")
        if robController.read() == "v":
            robController.write("RZz_high\n")
            if robController.read() == "v":
                robController.write(self.RZz_high.get()+"\n")

    def readAllCallback(self):
        self.alphaRead()
        self.tolRead()
        self.periodRead()
        self.offDurRead()
        self.numTolRead()
        self.xPushWtRead()
        self.xPullWtRead()
        self.yPushWtRead()
        self.yPullWtRead()
        self.zPushWtRead()
        self.zPullWtRead()
        self.RZxRead()
        self.RZy_lowRead()
        self.RZy_highRead()
        self.RZz_lowRead()
        self.RZz_highRead()

    def writeAllCallback(self):
        self.alphaWrite()
        self.tolWrite()
        self.periodWrite()
        self.offDurWrite()
        self.numTolWrite()
        self.xPushWtWrite()
        self.xPullWtWrite()
        self.yPushWtWrite()
        self.yPullWtWrite()
        self.zPushWtWrite()
        self.zPullWtWrite()
        self.RZxWrite()
        self.RZy_lowWrite()
        self.RZy_highWrite()
        self.RZz_lowWrite()
        self.RZz_highWrite()

    def runCalCallback(self):
        print("not implemented")

    def calBrowseCallback(self):
        global workspace
        self.calibrationFile.set(tkFileDialog.askopenfilename())
        workspace['RobotSettings']['calibrationFile'] = self.calibrationFile.get()

    def calLoadCallback(self):
        dis,pos,xPushDur,xPullDur,yPushDur,yPullDur,zPushDur,zPullDur =\
         np.loadtxt(self.calibrationFile.get(),skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3,4,5,6,7,8))
        self.dis = np.array2string(dis[0:-1:50],formatter={'float_kind':lambda dis: "%.1f" % dis})
        self.pos = np.array2string(pos[0:50],formatter={'float_kind':lambda pos: "%.1f" % pos})
        self.xPushDur = np.array2string(xPushDur,formatter={'float_kind':lambda xPushDur: "%.1f" % xPushDur})
        self.xPullDur = np.array2string(xPullDur,formatter={'float_kind':lambda xPullDur: "%.1f" % xPullDur})
        self.yPushDur = np.array2string(yPushDur,formatter={'float_kind':lambda yPushDur: "%.1f" % yPushDur})
        self.yPullDur = np.array2string(yPullDur,formatter={'float_kind':lambda yPullDur: "%.1f" % yPullDur})
        self.zPushDur = np.array2string(zPushDur,formatter={'float_kind':lambda zPushDur: "%.1f" % zPushDur})
        self.zPullDur = np.array2string(zPullDur,formatter={'float_kind':lambda zPullDur: "%.1f" % zPullDur})
        self.dis = self.dis[1:-1]+' '
        self.pos = self.pos[1:-1]+' '
        self.xPushDur = self.xPushDur[1:-1]+' '
        self.xPullDur = self.xPullDur[1:-1]+' '
        self.yPushDur = self.yPushDur[1:-1]+' '
        self.yPullDur = self.yPullDur[1:-1]+' '
        self.zPushDur = self.zPushDur[1:-1]+' '
        self.zPullDur = self.zPullDur[1:-1]+' '
        robController.write("c")
        if robController.read() == "c":
            robController.write("pos\n")
            if robController.read() == "c":
                robController.write(self.pos)
        if robController.read() == "c":
            print('pos loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load pos.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("dis\n")
            if robController.read() == "c":
                robController.write(self.dis)
                if robController.read() == "c":
                    print('dis loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load dis.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("xPushDur\n")
            if robController.read() == "c":
                robController.write(self.xPushDur)
                if robController.read() == "c":
                    print('xPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load xPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("xPullDur\n")
            if robController.read() == "c":
                robController.write(self.xPullDur)
                if robController.read() == "c":
                    print('xPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load xPullDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("yPushDur\n")
            if robController.read() == "c":
                robController.write(self.yPushDur)
                if robController.read() == "c":
                    print('yPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load yPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("yPullDur\n")
            if robController.read() == "c":
                robController.write(self.yPullDur)
                if robController.read() == "c":
                    print('yPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load yPullDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("zPushDur\n")
            if robController.read() == "c":
                robController.write(self.zPushDur)
                if robController.read() == "c":
                    print('zPushDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load zPushDur.")
        robController.write("c")
        if robController.read() == "c":
            robController.write("zPullDur\n")
            if robController.read() == "c":
                robController.write(self.zPullDur)
                if robController.read() == "c":
                    print('zPullDur loaded')
                else:
                    tkMessageBox.showinfo("Warning", "Failed to load zPullDur.")                  

    def comBrowseCallback(self):
        global workspace
        self.commandFile.set(tkFileDialog.askopenfilename())
        workspace['RobotSettings']['commandFile'] = self.commandFile.get()

    def loadComTypeCallback(self): 
        Ly = int(self.Ly.get())
        Lz = int(self.Lz.get())
        Axx = int(self.Axx.get())
        Ayy = int(self.Ayy.get())
        Azz = int(self.Azz.get()) 
        x0 = int(self.x0.get())
        y0 = int(self.y0.get())
        z0 = int(self.z0.get())       
        n = 100
        if self.commandType.get() == "sampleContinuous":
            rLow = int(self.rLow.get())
            rHigh = int(self.rHigh.get())
            thetaMag = float(self.thetaMag.get())
            r = rLow + (rHigh-rLow)*np.random.uniform(low=0.0,high=1.0,size=(500*n))**(1.0/3.0)
            thetay = thetaMag*np.random.uniform(low=-1.0,high=1.0,size=500*n)
            thetaz = thetaMag*np.random.uniform(low=-1.0,high=1.0,size=500*n)
            theta = np.sqrt(thetay**2+thetaz**2)
            r = r[theta<=thetaMag][0:n]
            thetay = thetay[theta<=thetaMag][0:n]
            thetaz = thetaz[theta<=thetaMag][0:n]
        elif self.commandType.get() == "sampleDiscrete":
            rSet,thetaySet,thetazSet = np.loadtxt(self.commandFile.get(),\
            skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
            randSample = np.random.choice(range(len(rSet)),replace=True,size=n)
            r = rSet[randSample]
            thetay = thetaySet[randSample]
            thetaz = thetazSet[randSample]
        elif self.commandType.get() == "fromFile":
            r,thetay,thetaz = np.loadtxt(self.commandFile.get(),\
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
        self.x = np.array2string(Ax,formatter={'float_kind':lambda Ax: "%.1f" % Ax})
        self.y = np.array2string(Ay,formatter={'float_kind':lambda Ay: "%.1f" % Ay})
        self.z = np.array2string(Az,formatter={'float_kind':lambda Az: "%.1f" % Az})
        self.r = np.array2string(r,formatter={'float_kind':lambda r: "%.1f" % r})
        self.thetay = np.array2string(thetay,formatter={'float_kind':lambda thetay: "%.2f" % thetay})
        self.thetaz = np.array2string(thetaz,formatter={'float_kind':lambda thetaz: "%.2f" % thetaz})
        self.x = self.x[1:-1]+' '
        self.y = self.y[1:-1]+' '
        self.z = self.z[1:-1]+' '
        self.r = self.r[1:-1]+' '
        self.thetay = self.thetay[1:-1]+' '
        self.thetaz = self.thetaz[1:-1]+' '
        robController.write("p")
        if robController.read() == "p":
            robController.write("xCommandPos\n")
            if robController.read() == "p":
                robController.write(self.x)
        if robController.read() == "p":
            print('x commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load x commands.")
        robController.write("p")
        if robController.read() == "p":
            robController.write("yCommandPos\n")
            if robController.read() == "p":
                robController.write(self.y)
        if robController.read() == "p":
            print('y commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load y commands.")
        robController.write("p")
        if robController.read() == "p":
            robController.write("zCommandPos\n")
            if robController.read() == "p":
                robController.write(self.z)
        if robController.read() == "p":
            print('z commands loaded')
        else:
            tkMessageBox.showinfo("Warning", "Failed to load z commands.") 
        self.commandsLoaded = True

#start program
ReachMaster(tk.Tk())