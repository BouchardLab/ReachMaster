"""This root application for ReachMaster.

This module implements the root application window for ReachMaster. The
window provides functionality for selecting a data output directory, a
configuration file, connecting to the experiment and robot controllers, 
accessing the settings windows, basic rig control, as well as selecting 
and running protocols.

Examples:
    From command line::

        $ python application.py

    From python::

        import reachmaster.application
        rm = application.ReachMaster()
        rm.run()

Todo:
    * Create executable
    * You have to also use ``sphinx.ext.todo`` extension

"""

import config
import settings.camera_settings as camset
import settings.experiment_settings as expset
import settings.robot_settings as robset
import interfaces.experiment_interface as expint
import interfaces.robot_interface as robint
import protocols
import Tkinter as tk 
import tkFileDialog
import tkMessageBox
import time

class ReachMaster:

    def __init__(self):
        #configure root window
        self.window = tk.Tk()
        self.window.title("ReachMaster 1.0")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)        
        self.cfg = config.default_cfg()
        config.save_tmp(self.cfg)
        self.data_dir = tk.StringVar()
        self.data_dir.set(self.cfg['ReachMaster']['data_dir']) 
        self.cfg_file = tk.StringVar()
        self.cfg_file.set(self.cfg['ReachMaster']['cfg_file'])
        self.port_list = expint.get_ports()
        self.exp_control_path = tk.StringVar()
        self.rob_control_path = tk.StringVar()
        if self.cfg['ReachMaster']['exp_control_path'] in self.port_list:
            self.exp_control_path.set(self.cfg['ReachMaster']['exp_control_path']) 
        else:
            self.exp_control_path.set(self.port_list[0])
        if self.cfg['ReachMaster']['rob_control_path'] in self.port_list:
            self.rob_control_path.set(self.cfg['ReachMaster']['rob_control_path']) 
        else:
            self.rob_control_path.set(self.port_list[0])        
        self.protocol_list = protocols.list_protocols()
        self.protocol = tk.StringVar()
        self.protocol.set(self.protocol_list[0])
        self.running = False  
        self.exp_connected = False
        self.rob_connected = False 
        self.protocol_running = False
        self.child = None 
        self.configure_window()                   

    def on_quit(self):  
        #prevents response when child window/process is active
        if (self.child is not None) and self.child.winfo_exists(): 
            return     
        #normal close when all children are dead
        answer = tkMessageBox.askyesnocancel("Question", "Save Configuration?")
        if answer == True:            
            config.save_cfg(self.cfg)            
        elif answer == False:
            pass
        else:
            self.run()
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)       
        self.running = False       
        self.window.destroy()

    def configure_window(self):
        tk.Label(text="Data Directory:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=0, column=0)
        tk.Label(textvariable=self.data_dir, bg="white").grid(row=0, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.ddbrowse_callback).grid(row=0, column=2)
        tk.Label(text="Configuration File:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=1, column=0)
        tk.Label(textvariable=self.cfg_file, bg="white").grid(row=1, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.cfbrowse_callback).grid(row=1, column=2)
        tk.Label(text="Experiment Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=2, column=0)
        self.exp_controller_menu = apply(tk.OptionMenu, (self.window, self.exp_control_path) + tuple(self.port_list))
        self.exp_controller_menu.grid(row=2, column=1)
        tk.Button(text="Connect", font='Arial 10 bold',width=14, command=self.exp_connect_callback).grid(row=2, column=2)
        tk.Button(text="Disconnect", font='Arial 10 bold',width=14, command=self.exp_disconnect_callback).grid(row=2, column=3)
        tk.Label(text="Robot Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=3, column=0)
        self.rob_controller_menu = apply(tk.OptionMenu, (self.window, self.rob_control_path) + tuple(self.port_list))
        self.rob_controller_menu.grid(row=3, column=1)
        tk.Button(text="Connect", font='Arial 10 bold',width=14, command=self.rob_connect_callback).grid(row=3, column=2)
        tk.Button(text="Disconnect", font='Arial 10 bold',width=14, command=self.rob_disconnect_callback).grid(row=3, column=3)
        tk.Button(text="Camera Settings", font='Arial 10 bold',width=16, command=self.cam_settings_callback).grid(row=4, sticky='W')
        tk.Button(text="Experiment Settings", font='Arial 10 bold',width=16, command=self.exp_settings_callback).grid(row=5, sticky='W')
        tk.Button(text="Robot Settings", font='Arial 10 bold',width=16, command=self.rob_settings_callback).grid(row=6, sticky='W')
        tk.Button(text="Move Robot", font='Arial 10 bold',width=16, command=self.move_rob_callback).grid(row=7, sticky='W')
        tk.Button(text="Toggle LED", font='Arial 10 bold',width=14, command=self.toggle_led_callback).grid(row=4, column=1)
        tk.Button(text="Toggle Lights", font='Arial 10 bold',width=14, command=self.toggle_lights_callback).grid(row=5, column=1)
        tk.Button(text="Deliver Water", font='Arial 10 bold',width=14, command=self.deliver_water_callback).grid(row=6, column=1)
        tk.Button(text="Flush Water", font='Arial 10 bold',width=14, command=self.flush_water_callback).grid(row=7, column=1)
        self.protocol_menu = apply(tk.OptionMenu, (self.window, self.protocol) + tuple(self.protocol_list))
        self.protocol_menu.grid(row=5, column=2)
        tk.Button(text="Run Protocol", font='Arial 10 bold',width=14, command=self.run_protocol_callback).grid(row=5, column=3)

    def ddbrowse_callback(self):
        self.data_dir.set(tkFileDialog.askdirectory())
        self.cfg['ReachMaster']['data_dir'] = self.data_dir.get()
        config.save_tmp(self.cfg)

    def cfbrowse_callback(self):
        self.cfg_file.set(tkFileDialog.askopenfilename())
        self.cfg = config.json_load_byteified(open(self.cfg_file.get()))
        self.cfg['ReachMaster']['cfg_file'] = self.cfg_file.get()
        self.cfg['ReachMaster']['data_dir'] = self.data_dir.get()
        config.save_tmp(self.cfg)

    def exp_connect_callback(self):
        if not self.exp_connected:
            try:
                self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
                self.cfg['ReachMaster']['exp_control_path'] = self.exp_control_path.get()
                self.exp_controller = expint.start_interface(self.cfg)
                self.exp_connected = True                
                config.save_tmp(self.cfg)
            except Exception as err:
                tkMessageBox.showinfo("Warning", err)

    def exp_disconnect_callback(self):
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
            self.exp_connected = False

    def rob_connect_callback(self):
        if not self.rob_connected:
            try:
                self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
                self.cfg['ReachMaster']['rob_control_path'] = self.rob_control_path.get()
                self.rob_controller = robint.start_interface(self.cfg)
                self.rob_connected = True
                config.save_tmp(self.cfg)
            except Exception as err:
                tkMessageBox.showinfo("Warning", err)

    def rob_disconnect_callback(self):
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
            self.rob_connected = False

    def cam_settings_callback(self):  
        if self.exp_connected:
            self.exp_disconnect_callback()
            time.sleep(2)
            self.child = camset.CameraSettings(self.window)
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def exp_settings_callback(self):
        self.child = expset.ExperimentSettings(self.window)

    def rob_settings_callback(self):   
        self.child = robset.RobotSettings(self.window)

    def move_rob_callback(self):
        if self.exp_connected:
            expint.move_robot(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def toggle_led_callback(self):
        if self.exp_connected:
            expint.toggle_led(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def toggle_lights_callback(self):
        if self.exp_connected:
            expint.toggle_lights(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")
    
    def deliver_water_callback(self):
        if self.exp_connected:
            expint.deliver_water(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def flush_water_callback(self):
        if self.exp_connected:
            expint.flush_water(self.exp_controller)
        else:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")

    def run_protocol_callback(self):  
        if self.exp_connected and self.rob_connected:
            self.cfg = config.json_load_byteified(open('./temp/tmp_config.txt'))
            self.cfg['Protocol']['type'] = self.protocol.get()
            config.save_tmp(self.cfg)
            expint.stop_interface(self.exp_controller)
            self.exp_connected = False
            robint.stop_interface(self.rob_controller)
            self.rob_connected = False
            time.sleep(2)
            self.child = protocols.Protocols(self.window)
            self.protocol_running = True
        elif not self.exp_connected:
            tkMessageBox.showinfo("Warning", "Experiment controller not connected.")
        else:
            tkMessageBox.showinfo("Warning", "Robot controller not connected.")

    def run(self):
        self.running = True
        try:
            while self.running:        
                if self.protocol_running:
                    try:
                        self.child.run()
                    except Exception as err:
                        self.protocol_running = False
                        print(err)
                self.window.update()
        except KeyboardInterrupt:
            self.on_quit()
                

if __name__ == '__main__':
    rm = ReachMaster()
    rm.run()