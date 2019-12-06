from .. import config
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import numpy as np
import serial
import os 
import json

class ExperimentSettings(tk.Toplevel):

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()
        self.title("Experiment Settings") 
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.onQuit) 
        self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
        self.lightsOnDur = tk.StringVar()
        self.lightsOnDur.set(str(self.cfg['ExperimentSettings']['lightsOnDur']))
        self.lightsOffDur = tk.StringVar()
        self.lightsOffDur.set(str(self.cfg['ExperimentSettings']['lightsOffDur']))
        self.rewardWinDur = tk.StringVar()
        self.rewardWinDur.set(str(self.cfg['ExperimentSettings']['rewardWinDur']))
        self.maxRewards = tk.StringVar()
        self.maxRewards.set(str(self.cfg['ExperimentSettings']['maxRewards']))
        self.solenoidOpenDur = tk.StringVar()
        self.solenoidOpenDur.set(str(self.cfg['ExperimentSettings']['solenoidOpenDur']))
        self.solenoidBounceDur = tk.StringVar()
        self.solenoidBounceDur.set(str(self.cfg['ExperimentSettings']['solenoidBounceDur']))
        self.flushDur = tk.StringVar()
        self.flushDur.set(str(self.cfg['ExperimentSettings']['flushDur']))
        self.reachDelay = tk.StringVar()
        self.reachDelay.set(str(self.cfg['ExperimentSettings']['reachDelay']))
        self.expConnect()
        self.setup_UI()

    def onQuit(self):
        self.cfg['ExperimentSettings']['lightsOnDur'] = int(self.lightsOnDur.get())
        self.cfg['ExperimentSettings']['lightsOffDur'] = int(self.lightsOffDur.get())
        self.cfg['ExperimentSettings']['rewardWinDur'] = int(self.rewardWinDur.get())
        self.cfg['ExperimentSettings']['maxRewards'] = int(self.maxRewards.get()) 
        self.cfg['ExperimentSettings']['solenoidOpenDur'] = int(self.solenoidOpenDur.get())
        self.cfg['ExperimentSettings']['solenoidBounceDur'] = int(self.solenoidBounceDur.get())
        self.cfg['ExperimentSettings']['flushDur'] = int(self.flushDur.get())
        self.cfg['ExperimentSettings']['reachDelay'] = int(self.reachDelay.get())
        config.save_tmp(self.cfg)
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
        tk.Button(self,text="Read All",font='Arial 10 bold',width=14,command=self.readAllCallback).grid(row=0,column=2)
        tk.Button(self,text="Write All",font='Arial 10 bold',width=14,command=self.writeAllCallback).grid(row=0,column=3)
        tk.Label(self,text="Lights On (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=1, column=0)   
        tk.Entry(self,textvariable=self.lightsOnDur,width=17).grid(row=1, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.lightsOnDurRead).grid(row=1, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.lightsOnDurWrite).grid(row=1, column=3)
        tk.Label(self,text="Lights Off (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=2, column=0)   
        tk.Entry(self,textvariable=self.lightsOffDur,width=17).grid(row=2, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.lightsOffDurRead).grid(row=2, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.lightsOffDurWrite).grid(row=2, column=3)
        tk.Label(self,text="Reward Window (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=3, column=0)   
        tk.Entry(self,textvariable=self.rewardWinDur,width=17).grid(row=3, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.rewardWinDurRead).grid(row=3, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.rewardWinDurWrite).grid(row=3, column=3)
        tk.Label(self,text="# Rewards/Trial:", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=4, column=0)   
        tk.Entry(self,textvariable=self.maxRewards,width=17).grid(row=4, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.maxRewardsRead).grid(row=4, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.maxRewardsWrite).grid(row=4, column=3)
        tk.Label(self,text="Solenoid Open (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=5, column=0)   
        tk.Entry(self,textvariable=self.solenoidOpenDur,width=17).grid(row=5, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.solenoidOpenDurRead).grid(row=5, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.solenoidOpenDurWrite).grid(row=5, column=3)
        tk.Label(self,text="Solenoid Bounce (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=6, column=0)   
        tk.Entry(self,textvariable=self.solenoidBounceDur,width=17).grid(row=6, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.solenoidBounceDurRead).grid(row=6, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.solenoidBounceDurWrite).grid(row=6, column=3)
        tk.Label(self,text="Flush (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=7, column=0)   
        tk.Entry(self,textvariable=self.flushDur,width=17).grid(row=7, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.flushDurRead).grid(row=7, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.flushDurWrite).grid(row=7, column=3)
        tk.Label(self,text="Reach Delay (ms):", font='Arial 10 bold', bg="white",width=23,anchor="e").grid(row=8, column=0)   
        tk.Entry(self,textvariable=self.reachDelay,width=17).grid(row=8, column=1)
        tk.Button(self,text="Read",font='Arial 10 bold',width=14,command=self.reachDelayRead).grid(row=8, column=2)
        tk.Button(self,text="Write",font='Arial 10 bold',width=14,command=self.reachDelayWrite).grid(row=8, column=3)

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