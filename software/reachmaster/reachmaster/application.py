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
from serial.tools import list_ports

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
        self.port_list = list(list_ports.comports())
        for i in range(len(self.port_list)):
            self.port_list[i] = self.port_list[i].device
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

    def _configure_window(self):
        tk.Label(text="Data Directory:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=0, column=0)
        tk.Label(textvariable=self.data_dir, bg="white").grid(row=0, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.ddbrowse_callback).grid(row=0, column=2)
        tk.Label(text="Configuration File:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=1, column=0)
        tk.Label(textvariable=self.cfg_file, bg="white").grid(row=1, column=1)
        tk.Button(text="Browse", font='Arial 10 bold',width=14, command=self.cfbrowse_callback).grid(row=1, column=2)
        tk.Label(text="Experiment Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=2, column=0)
        self.exp_controllerMenu = apply(tk.OptionMenu, (self.window, self.exp_control_path) + tuple(self.port_list))
        self.exp_controllerMenu.grid(row=2, column=1)
        tk.Button(text="Connect", font='Arial 10 bold',width=14, command=self.exp_connect_callback).grid(row=2, column=2)
        tk.Button(text="Disconnect", font='Arial 10 bold',width=14, command=self.exp_disconnect_callback).grid(row=2, column=3)
        tk.Label(text="Robot Controller:", font='Arial 10 bold', bg="white",width=22,anchor="e").grid(row=3, column=0)
        self.rob_controllerMenu = apply(tk.OptionMenu, (self.window, self.rob_control_path) + tuple(self.port_list))
        self.rob_controllerMenu.grid(row=3, column=1)
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
        tk.Button(text="Run Protocol", font='Arial 10 bold',width=14, command=self.run_protocol_callback).grid(row=5, column=2)

    def ddbrowse_callback(self):
        self.data_dir.set(tkFileDialog.askdirectory())
        self.cfg['ReachMaster']['data_dir'] = self.data_dir.get()

    def cfbrowse_callback(self):
        self.cfg_file.set(tkFileDialog.askopenfilename())
        self.cfg = config.json_load_byteified(open(self.cfg_file.get()))
        self.cfg['ReachMaster']['cfg_file'] = self.cfg_file.get()
        config.save_tmp(self.cfg)
        self.output_params = self.cfg['CameraSettings']['self.output_params']

    def exp_connect_callback(self):
        try:
            self.exp_controller = expint.start_interface(self.cfg)
            self.exp_connected = True
        except Exception as err:
            tkMessageBox.showinfo("Warning", err)

    def exp_disconnect_callback(self):
        expint.stop_interface(self.exp_controller)
        self.exp_connected = False

    def rob_connect_callback(self):
        try:
            self.rob_controller = robint.start_interface(self.cfg)
            self.rob_connected = True
        except Exception as err:
            tkMessageBox.showinfo("Warning", err)

    def rob_disconnect_callback(self):
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
        if self.exp_connected:
            self.exp_disconnect_callback()
            time.sleep(2)   
            self.child = expset.ExperimentSettings(self.window)
        else:
            tkMessageBox.showinfo("Warning", "Experiment Controller not connected.")

    def rob_settings_callback(self):
        if self.rob_connected:
            self.rob_disconnect_callback()
            time.sleep(2)   
            self.child = robset.RobotSettings(self.window)
        else:
            tkMessageBox.showinfo("Warning", "Robot controller not connected.")

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
            expint.stop_interface(self.exp_controller)
            robint.stop_interface(self.rob_controller)
            time.sleep(2)
            self.protocol = protocols.Protocols(self.window)
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
                        self.protocol.run()
                    except Exception as err:
                        self.protocol_running = False
                        if err.status == 10:
                            tkMessageBox.showinfo("Warning", err)
                self.window.update()
        except KeyboardInterrupt:
            self.on_quit()
                

if __name__ == '__main__':
    rm = ReachMaster()
    rm.run()