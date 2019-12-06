from .. import config
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import numpy as np
import serial
import os 
import json

class RobotSettings(tk.Toplevel):

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()
        self.title("Robot Settings")   
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
        self.alpha = tk.StringVar()
        self.alpha.set(str(self.cfg['RobotSettings']['alpha']))
        self.tol = tk.StringVar()
        self.tol.set(str(self.cfg['RobotSettings']['tol']))
        self.period = tk.StringVar()
        self.period.set(str(self.cfg['RobotSettings']['period']))
        self.offDur = tk.StringVar()
        self.offDur.set(str(self.cfg['RobotSettings']['offDur']))
        self.numTol = tk.StringVar()
        self.numTol.set(str(self.cfg['RobotSettings']['numTol']))
        self.xPushWt = tk.StringVar()
        self.xPushWt.set(str(self.cfg['RobotSettings']['xPushWt']))
        self.xPullWt = tk.StringVar()
        self.xPullWt.set(str(self.cfg['RobotSettings']['xPullWt']))
        self.yPushWt = tk.StringVar()
        self.yPushWt.set(str(self.cfg['RobotSettings']['yPushWt']))
        self.yPullWt = tk.StringVar()
        self.yPullWt.set(str(self.cfg['RobotSettings']['yPullWt']))
        self.zPushWt = tk.StringVar()
        self.zPushWt.set(str(self.cfg['RobotSettings']['zPushWt']))
        self.zPullWt = tk.StringVar()
        self.zPullWt.set(str(self.cfg['RobotSettings']['zPullWt']))
        self.RZx = tk.StringVar()
        self.RZx.set(str(self.cfg['RobotSettings']['RZx']))
        self.RZy_low = tk.StringVar()
        self.RZy_low.set(str(self.cfg['RobotSettings']['RZy_low']))
        self.RZy_high = tk.StringVar()
        self.RZy_high.set(str(self.cfg['RobotSettings']['RZy_high']))
        self.RZz_low = tk.StringVar()
        self.RZz_low.set(str(self.cfg['RobotSettings']['RZz_low']))
        self.RZz_high = tk.StringVar()
        self.RZz_high.set(str(self.cfg['RobotSettings']['RZz_high']))
        self.calibrationFile = tk.StringVar()
        self.calibrationFile.set(str(self.cfg['RobotSettings']['calibrationFile']))
        self.dis = self.cfg['RobotSettings']['dis']
        self.pos = self.cfg['RobotSettings']['pos']
        self.xPushDur = self.cfg['RobotSettings']['xPushDur']
        self.xPullDur = self.cfg['RobotSettings']['xPullDur']
        self.yPushDur = self.cfg['RobotSettings']['yPushDur']
        self.yPullDur = self.cfg['RobotSettings']['yPullDur']
        self.zPushDur = self.cfg['RobotSettings']['zPushDur']
        self.zPullDur = self.cfg['RobotSettings']['zPullDur'] 
        self.commandFile = tk.StringVar()
        self.commandFile.set(str(self.cfg['RobotSettings']['commandFile']))
        self.commandType = tk.StringVar()
        self.commandType.set(self.cfg['RobotSettings']['commandType'])
        self.Ly = tk.StringVar()
        self.Ly.set(str(self.cfg["RobotSettings"]["Ly"]))
        self.Lz = tk.StringVar()
        self.Lz.set(str(self.cfg["RobotSettings"]["Lz"]))
        self.Axx = tk.StringVar()
        self.Axx.set(str(self.cfg["RobotSettings"]["Axx"]))
        self.Ayy = tk.StringVar()
        self.Ayy.set(str(self.cfg["RobotSettings"]["Ayy"]))
        self.Azz = tk.StringVar()
        self.Azz.set(str(self.cfg["RobotSettings"]["Azz"]))
        self.x0 = tk.StringVar()
        self.x0.set(str(self.cfg["RobotSettings"]["x0"]))
        self.y0 = tk.StringVar()
        self.y0.set(str(self.cfg["RobotSettings"]["y0"]))
        self.z0 = tk.StringVar()
        self.z0.set(str(self.cfg["RobotSettings"]["z0"]))
        self.rLow = tk.StringVar()
        self.rLow.set(str(self.cfg["RobotSettings"]["rLow"]))
        self.rHigh = tk.StringVar()
        self.rHigh.set(str(self.cfg["RobotSettings"]["rHigh"]))
        self.thetaMag = tk.StringVar()
        self.thetaMag.set(str(self.cfg["RobotSettings"]["thetaMag"]))
        self.r =  self.cfg["RobotSettings"]["rCommandPos"]
        self.thetay =  self.cfg["RobotSettings"]["thetayCommandPos"]
        self.thetaz =  self.cfg["RobotSettings"]["thetazCommandPos"]
        self.x =  self.cfg["RobotSettings"]["xCommandPos"]
        self.y =  self.cfg["RobotSettings"]["yCommandPos"]
        self.z =  self.cfg["RobotSettings"]["zCommandPos"]
        self.robConnect()
        self.setup_UI()

    def onQuit(self):
        self.cfg['RobotSettings']['alpha'] = float(self.alpha.get())
        self.cfg['RobotSettings']['tol'] = float(self.tol.get())
        self.cfg['RobotSettings']['period'] = float(self.period.get())
        self.cfg['RobotSettings']['offDur'] = int(self.offDur.get()) 
        self.cfg['RobotSettings']['numTol'] = int(self.numTol.get())
        self.cfg['RobotSettings']['xPushWt'] = float(self.xPushWt.get())
        self.cfg['RobotSettings']['xPullWt'] = float(self.xPullWt.get())
        self.cfg['RobotSettings']['yPushWt'] = float(self.yPushWt.get())
        self.cfg['RobotSettings']['yPullWt'] = float(self.yPullWt.get())
        self.cfg['RobotSettings']['zPushWt'] = float(self.zPushWt.get())
        self.cfg['RobotSettings']['zPullWt'] = float(self.zPullWt.get())
        self.cfg['RobotSettings']['RZx'] = int(self.RZx.get())
        self.cfg['RobotSettings']['RZy_low'] = int(self.RZy_low.get())
        self.cfg['RobotSettings']['RZy_high'] = int(self.RZy_high.get())
        self.cfg['RobotSettings']['RZz_low'] = int(self.RZz_low.get())
        self.cfg['RobotSettings']['RZz_high'] = int(self.RZz_high.get())
        self.cfg["RobotSettings"]["calibrationFile"] = self.calibrationFile.get()
        self.cfg["RobotSettings"]["dis"] = self.dis
        self.cfg["RobotSettings"]["pos"] = self.pos
        self.cfg["RobotSettings"]["xPushDur"] = self.xPushDur
        self.cfg["RobotSettings"]["xPullDur"] = self.xPullDur
        self.cfg["RobotSettings"]["yPushDur"] = self.yPushDur
        self.cfg["RobotSettings"]["yPullDur"] = self.yPullDur
        self.cfg["RobotSettings"]["zPushDur"] = self.zPushDur
        self.cfg["RobotSettings"]["zPullDur"] = self.zPullDur
        self.cfg["RobotSettings"]["commandFile"] = self.commandFile.get()
        self.cfg["RobotSettings"]["commandType"] = self.commandType.get()        
        self.cfg["RobotSettings"]["Ly"] = int(self.Ly.get())
        self.cfg["RobotSettings"]["Lz"] = int(self.Lz.get())
        self.cfg["RobotSettings"]["Axx"] = int(self.Axx.get())
        self.cfg["RobotSettings"]["Ayy"] = int(self.Ayy.get())
        self.cfg["RobotSettings"]["Azz"] = int(self.Azz.get())
        self.cfg["RobotSettings"]["x0"] = int(self.x0.get())
        self.cfg["RobotSettings"]["y0"] = int(self.y0.get())
        self.cfg["RobotSettings"]["z0"] = int(self.z0.get())
        self.cfg["RobotSettings"]["rLow"] = int(self.rLow.get())
        self.cfg["RobotSettings"]["rHigh"] = int(self.rHigh.get())
        self.cfg["RobotSettings"]["thetaMag"] = float(self.thetaMag.get())
        self.cfg["RobotSettings"]["xCommandPos"] = self.x
        self.cfg["RobotSettings"]["yCommandPos"] = self.y
        self.cfg["RobotSettings"]["zCommandPos"] = self.z
        self.cfg["RobotSettings"]["rCommandPos"] = self.r
        self.cfg["RobotSettings"]["thetayCommandPos"] = self.thetay
        self.cfg["RobotSettings"]["thetazCommandPos"] = self.thetaz        
        config.save_tmp(self.cfg)
        self.robDisconnect()
        self.destroy()

    def robConnect(self):
        global robController
        robController = serial.Serial(self.cfg['ReachMaster']['robControlPath'],
            self.cfg['ReachMaster']['serialBaud'],
            timeout=self.cfg['ReachMaster']['controlTimeout'])
        # time.sleep(2) #wait for controller to wake up
        robController.flushInput()
        robController.write("h")
        response = robController.read()
        if response == "h":
            self.robControlOn = True
        else:
            tkMessageBox.showinfo("Warning", "Failed to connect.")

    def robDisconnect(self):
        if self.robControlOn:
            robController.write("e")
            robController.close()
        else:
            tkMessageBox.showinfo("Warning", "Robot Controller not connected.")

    def setup_UI(self):
        tk.Button(self,text="Read All",font='Arial 10 bold',width=14,command=self.readAllCallback).grid(row=0,column=2)
        tk.Button(self,text="Write All",font='Arial 10 bold',width=14,command=self.writeAllCallback).grid(row=0,column=3)
        tk.Label(self,text="Position Smoothing:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, column=0)   
        tk.Entry(self,textvariable=self.alpha,width=17).grid(row=1, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.alphaRead).grid(row=1, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.alphaWrite).grid(row=1, column=3)
        tk.Label(self,text="Valve Period:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=0)   
        tk.Entry(self,textvariable=self.period,width=17).grid(row=2, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.periodRead).grid(row=2, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.periodWrite).grid(row=2, column=3)
        tk.Label(self,text="Off Duration:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=0)   
        tk.Entry(self,textvariable=self.offDur,width=17).grid(row=3, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.offDurRead).grid(row=3, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.offDurWrite).grid(row=3, column=3)
        tk.Label(self,text="Converge Tolerance:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=0)   
        tk.Entry(self,textvariable=self.tol,width=17).grid(row=4, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.tolRead).grid(row=4, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.tolWrite).grid(row=4, column=3)
        tk.Label(self,text="# w/in Tolerance:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=0)   
        tk.Entry(self,textvariable=self.numTol,width=17).grid(row=5, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.numTolRead).grid(row=5, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.numTolWrite).grid(row=5, column=3)
        tk.Label(self,text="Push Weight (x):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=0)   
        tk.Entry(self,textvariable=self.xPushWt,width=17).grid(row=6, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.xPushWtRead).grid(row=6, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.xPushWtWrite).grid(row=6, column=3)
        tk.Label(self,text="Pull Weight (x):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=0)   
        tk.Entry(self,textvariable=self.xPullWt,width=17).grid(row=7, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.xPullWtRead).grid(row=7, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.xPullWtWrite).grid(row=7, column=3)
        tk.Label(self,text="Push Weight (y):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=0)   
        tk.Entry(self,textvariable=self.yPushWt,width=17).grid(row=8, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.yPushWtRead).grid(row=8, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.yPushWtWrite).grid(row=8, column=3)
        tk.Label(self,text="Pull Weight (y):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, column=0)   
        tk.Entry(self,textvariable=self.yPullWt,width=17).grid(row=9, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.yPullWtRead).grid(row=9, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.yPullWtWrite).grid(row=9, column=3)
        tk.Label(self,text="Push Weight (z):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, column=0)   
        tk.Entry(self,textvariable=self.zPushWt,width=17).grid(row=10, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.zPushWtRead).grid(row=10, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.zPushWtWrite).grid(row=10, column=3)
        tk.Label(self,text="Pull Weight (z):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, column=0)   
        tk.Entry(self,textvariable=self.zPullWt,width=17).grid(row=11, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.zPullWtRead).grid(row=11, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.zPullWtWrite).grid(row=11, column=3)
        tk.Label(self,text="Reward Zone (xmin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=12, column=0)   
        tk.Entry(self,textvariable=self.RZx,width=17).grid(row=12, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.RZxRead).grid(row=12, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.RZxWrite).grid(row=12, column=3)
        tk.Label(self,text="Reward Zone (ymin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=13, column=0)   
        tk.Entry(self,textvariable=self.RZy_low,width=17).grid(row=13, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.RZy_lowRead).grid(row=13, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.RZy_lowWrite).grid(row=13, column=3)
        tk.Label(self,text="Reward Zone (ymax):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=14, column=0)   
        tk.Entry(self,textvariable=self.RZy_high,width=17).grid(row=14, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.RZy_highRead).grid(row=14, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.RZy_highWrite).grid(row=14, column=3)
        tk.Label(self,text="Reward Zone (zmin):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, column=0)   
        tk.Entry(self,textvariable=self.RZz_low,width=17).grid(row=15, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.RZz_lowRead).grid(row=15, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.RZz_lowWrite).grid(row=15, column=3)
        tk.Label(self,text="Reward Zone (zmax):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=16, column=0)   
        tk.Entry(self,textvariable=self.RZz_high,width=17).grid(row=16, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.RZz_highRead).grid(row=16, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.RZz_highWrite).grid(row=16, column=3)
        tk.Button(self,text="Run Calibration",font='Arial 10 bold',width=14,command=self.runCalCallback).grid(row=1, column=6)
        tk.Label(self,text="Calibration File:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=4)
        tk.Label(self,textvariable=self.calibrationFile, bg="white").grid(row=2, column=5)
        tk.Button(self,text="Browse", font='Arial 10 bold',width=14, command=self.calBrowseCallback).grid(row=2, column=6)
        tk.Button(self,text="Load", font='Arial 10 bold',width=14, command=self.calLoadCallback).grid(row=2, column=7)
        tk.Label(self,text="Command File:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=4)
        tk.Label(self,textvariable=self.commandFile, bg="white").grid(row=3, column=5)
        tk.Button(self,text="Browse", font='Arial 10 bold',width=14, command=self.comBrowseCallback).grid(row=3, column=6)
        tk.Label(self,text="Command Type:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=4)
        self.comTypeMenu = tk.OptionMenu(self,self.commandType,
            "sampleContinuous",
            "sampleDiscrete",
            "fromFile")
        self.comTypeMenu.configure(width=23)
        self.comTypeMenu.grid(row=4, column=5)
        tk.Button(self,text="Load", font='Arial 10 bold',width=14, command=self.loadComTypeCallback).grid(row=4, column=6)
        tk.Label(self,text="rLow:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=4)   
        tk.Entry(self,textvariable=self.rLow,width=17).grid(row=5, column=5)
        tk.Label(self,text="rHigh:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=4)   
        tk.Entry(self,textvariable=self.rHigh,width=17).grid(row=6, column=5)
        tk.Label(self,text="thetaMag:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=4)   
        tk.Entry(self,textvariable=self.thetaMag,width=17).grid(row=7, column=5)
        tk.Label(self,text="Ly:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=4)   
        tk.Entry(self,textvariable=self.Ly,width=17).grid(row=8, column=5)
        tk.Label(self,text="Lz:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=9, column=4)   
        tk.Entry(self,textvariable=self.Lz,width=17).grid(row=9, column=5)
        tk.Label(self,text="Axx:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=10, column=4)   
        tk.Entry(self,textvariable=self.Axx,width=17).grid(row=10, column=5)
        tk.Label(self,text="Ayy:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=11, column=4)   
        tk.Entry(self,textvariable=self.Ayy,width=17).grid(row=11, column=5)
        tk.Label(self,text="Azz:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=12, column=4)   
        tk.Entry(self,textvariable=self.Azz,width=17).grid(row=12, column=5)
        tk.Label(self,text="x0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=13, column=4)   
        tk.Entry(self,textvariable=self.x0,width=17).grid(row=13, column=5)
        tk.Label(self,text="y0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=14, column=4)   
        tk.Entry(self,textvariable=self.y0,width=17).grid(row=14, column=5)
        tk.Label(self,text="z0:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=15, column=4)   
        tk.Entry(self,textvariable=self.z0,width=17).grid(row=15, column=5)

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
        self.calibrationFile.set(tkFileDialog.askopenfilename())
        self.cfg['RobotSettings']['calibrationFile'] = self.calibrationFile.get()

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
        self.commandFile.set(tkFileDialog.askopenfilename())
        self.cfg['RobotSettings']['commandFile'] = self.commandFile.get()

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