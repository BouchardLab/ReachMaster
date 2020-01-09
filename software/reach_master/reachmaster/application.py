"""The root application window provides functionality for selecting a 
data output directory, a configuration file, connecting to the 
experiment and robot controllers, accessing the settings windows, 
basic rig control, as well as selecting and running protocols.

Examples:
    From command line::

        $ python application.py

    From python::

        >>> import reachmaster.application
        >>> rm = application.ReachMaster()
        >>> rm.run()

Todo:
    * Automated unit tests

"""

from . import config as cfg
from .settings import camera_settings as camset
from .settings import experiment_settings as expset
from .settings import robot_settings as robset
from .interfaces import experiment_interface as expint
from .interfaces import robot_interface as robint
from . import protocols
import tkinter as tk 
import tkinter.filedialog
import tkinter.messagebox
import time
from os import path

class ReachMaster:
    """The primary class for the root ReachMaster application.

    Configures and provides callback methods for the root application 
    window. The `run()` method detects events during normal operation 
    and the `on_quit()` method is called prior to application 
    destruction.   

    Attributes
    ----------
    window : instance 
        Tkinter object that constructs the root application window.
    config : dict 
        The current configuration settings for the application.
    data_dir : instance
        Tkinter StringVar that captures the user-selected data 
        output directory.
    config_file : instance
        Tkinter StringVar that captures the user-selected 
        configuration file.
    port_list : list(str) 
        List of serial port names with connected devices.
    exp_control_port : instance
        Tkinter StringVar that captures the user-selected port from 
        the port list for the experiment microcontroller.
    rob_control_port : insstance 
        Tkinter StringVar that captures the user-selected port from 
        the port list for the robot microcontroller.
    protocol_list : list(str) 
        List of protocol options. 
    protocol : instance
        Tkinter StringVar that captures the user-selected protocol.
    running : bool 
        True if the root application window has not been destroyed. 
    exp_connected : bool 
        True if interface to the experiment microcontroller is active.
    rob_connected : bool 
        True if interface to the robot microcontroller is active.
    protocol_running : bool 
        True while an experiment protocol is being executed. 
    child : NoneType or instance 
        Tkinter Toplevel object that constructs child window selected 
        by the user from the root application window. Defaults to None 
        when child window is closed. 

    """

    def __init__(self):
        self.window = tk.Tk()        
        self.window.title("ReachMaster")
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)        
        self.config = cfg.default_config()    
        cfg.save_tmp(self.config)
        self.data_dir = tk.StringVar()        
        self.data_dir.set(self.config['ReachMaster']['data_dir']) 
        self.config_file = tk.StringVar()
        self.config_file.set(self.config['ReachMaster']['config_file'])
        self.port_list = expint.get_ports()      
        self.exp_control_port = tk.StringVar()
        self.rob_control_port = tk.StringVar()
        if self.config['ReachMaster']['exp_control_port'] in self.port_list:
            self.exp_control_port.set(self.config['ReachMaster']['exp_control_port']) 
        else:
            self.exp_control_port.set(self.port_list[0])
        if self.config['ReachMaster']['rob_control_port'] in self.port_list:
            self.rob_control_port.set(self.config['ReachMaster']['rob_control_port']) 
        else:
            self.rob_control_port.set(self.port_list[0])        
        self.protocol_list = protocols.list_protocols()
        self.protocol = tk.StringVar()
        self.protocol.set(self.protocol_list[0])
        self.running = False  
        self.exp_connected = False
        self.rob_connected = False 
        self.protocol_running = False        
        self.child = None         
        self._configure_window()                   

    def on_quit(self): 
        """Called prior to destruction of the ReachMaster application.

        Prevents destruction if any child windows are open. Asks user 
        to save the configuration file to the data directory. Stops 
        all active interfaces. 

        """ 
        if (self.child is not None) and self.child.winfo_exists(): 
            return     
        answer = tkinter.messagebox.askyesnocancel("Question", "Save Configuration?")
        if answer == True:            
            cfg.save_config(self.config)            
        elif answer == False:
            pass
        else:
            return
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)       
        self.running = False       
        self.window.destroy()

    def run(self):
        """The application main loop."""
        self.running = True
        try:
            while self.running:        
                if self.protocol_running and self.child.baseline_acquired:
                    try:
                        self.child.run()
                    except Exception as err:
                        self.protocol_running = False
                        tkinter.messagebox.showinfo("Warning", err)
                self.window.update()
        except KeyboardInterrupt:
            self.on_quit()

    def _configure_window(self):
        tk.Label(
            text = "Data Directory:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 22,
            anchor = "e"
            ).grid(row = 0, column = 0)
        tk.Label(textvariable = self.data_dir, bg = "white").grid(row = 0, column = 1)
        tk.Button(
            text = "Browse", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.data_dir_browse_callback
            ).grid(row = 0, column = 2)
        tk.Label(
            text = "Configuration File:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 22,
            anchor = "e"
            ).grid(row = 1, column = 0)
        tk.Label(textvariable = self.config_file, bg = "white").grid(row = 1, column = 1)
        tk.Button(
            text = "Browse", 
            font = 'Arial 10 bold',
            width = 14,
            command = self.config_file_browse_callback
            ).grid(row = 1, column = 2)
        tk.Label(
            text = "Experiment Controller:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 22,
            anchor = "e"
            ).grid(row = 2, column = 0)
        self.exp_controller_menu = tk.OptionMenu(*(self.window, self.exp_control_port) + tuple(self.port_list))
        self.exp_controller_menu.grid(row = 2, column = 1)
        tk.Button(
            text = "Connect", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.exp_connect_callback
            ).grid(row = 2, column = 2)
        tk.Button(
            text = "Disconnect", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.exp_disconnect_callback
            ).grid(row = 2, column = 3)
        tk.Label(
            text = "Robot Controller:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 22,
            anchor = "e"
            ).grid(row = 3, column = 0)
        self.rob_controller_menu = tk.OptionMenu(*(self.window, self.rob_control_port) + tuple(self.port_list))
        self.rob_controller_menu.grid(row = 3, column = 1)
        tk.Button(
            text = "Connect", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.rob_connect_callback
            ).grid(row = 3, column = 2)
        tk.Button(
            text = "Disconnect", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.rob_disconnect_callback
            ).grid(row = 3, column = 3)
        tk.Button(
            text = "Camera Settings", 
            font = 'Arial 10 bold',
            width = 16, 
            command = self.cam_settings_callback
            ).grid(row = 4, sticky = 'W')
        tk.Button(
            text = "Experiment Settings", 
            font = 'Arial 10 bold',
            width = 16, 
            command = self.exp_settings_callback
            ).grid(row = 5, sticky = 'W')
        tk.Button(
            text = "Robot Settings", 
            font = 'Arial 10 bold',
            width = 16, 
            command = self.rob_settings_callback
            ).grid(row = 6, sticky = 'W')
        tk.Button(
            text = "Move Robot", 
            font = 'Arial 10 bold',
            width = 16, 
            command = self.move_rob_callback
            ).grid(row = 7, sticky = 'W')
        tk.Button(
            text = "Toggle LED", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.toggle_led_callback
            ).grid(row = 4, column = 1)
        tk.Button(
            text = "Toggle Lights", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.toggle_lights_callback
            ).grid(row = 5, column = 1)
        tk.Button(
            text = "Deliver Water", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.deliver_water_callback
            ).grid(row = 6, column = 1)
        tk.Button(
            text = "Flush Water", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.flush_water_callback
            ).grid(row = 7, column = 1)
        self.protocol_menu = tk.OptionMenu(*(self.window, self.protocol) + tuple(self.protocol_list))
        self.protocol_menu.grid(row = 5, column = 2)
        tk.Button(
            text = "Run Protocol", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.run_protocol_callback
            ).grid(row = 5, column = 3)

    # Callbacks ------------------------------------------------------------------------------

    def data_dir_browse_callback(self):
        """Allows user to set the data output directory."""
        self.data_dir.set(tkinter.filedialog.askdirectory())
        self.config['ReachMaster']['data_dir'] = self.data_dir.get()
        cfg.save_tmp(self.config)

    def config_file_browse_callback(self):
        """Allows user to load a previously saved configuration file."""
        self.config_file.set(tkinter.filedialog.askopenfilename())
        try:
            self.config = cfg.load_config(self.config_file.get())
            self.config['ReachMaster']['config_file'] = self.config_file.get()
            self.exp_control_port.set(self.config['ReachMaster']['exp_control_port']) 
            self.rob_control_port.set(self.config['ReachMaster']['rob_control_port'])
            self.protocol.set(self.config['Protocol']['type'])
            #prefer some of the user's current directory selection
            self.config['ReachMaster']['data_dir'] = self.data_dir.get()           
            cfg.save_tmp(self.config)
        except Exception as err:
            tkinter.messagebox.showinfo("Warning", err)

    def exp_connect_callback(self):
        """Connects to the experiment microcontroller located at the
        user selected port."""
        if not self.exp_connected:
            try:
                self.config = cfg.load_config('./temp/tmp_config.json') ##byte conversion off
                self.config['ReachMaster']['exp_control_port'] = self.exp_control_port.get()
                self.exp_controller = expint.start_interface(self.config)
                self.exp_connected = True                
                cfg.save_tmp(self.config)
            except Exception as err:
                print(err)
                tkinter.messagebox.showinfo("Warning", err)

    def exp_disconnect_callback(self):
        """Disconncects from the experiment microcontroller."""
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
            self.exp_connected = False

    def rob_connect_callback(self):
        """Connects to the robot microcontroller located at the user 
        selected port."""
        if not self.rob_connected:
            try:
                self.config = cfg.load_config('./temp/tmp_config.json')
                self.config['ReachMaster']['rob_control_port'] = self.rob_control_port.get()
                self.rob_controller = robint.start_interface(self.config)
                self.rob_connected = True
                cfg.save_tmp(self.config)
            except Exception as err:
                tkinter.messagebox.showinfo("Warning", err)

    def rob_disconnect_callback(self):
        """Disconncects from the robot microcontroller."""
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
            self.rob_connected = False

    def cam_settings_callback(self):  
        """Opens the camera settings window."""
        if self.exp_connected:
            self.exp_disconnect_callback()
            time.sleep(2)
            self.child = camset.CameraSettings(self.window)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment Controller not connected.")

    def exp_settings_callback(self):
        """Opens the experiment settings window."""
        self.child = expset.ExperimentSettings(self.window)

    def rob_settings_callback(self): 
        """Opens the robot settings window."""  
        self.child = robset.RobotSettings(self.window)

    def move_rob_callback(self):
        """Commands the robot to move to the next position in
        its currently loaded command sequence."""
        if self.exp_connected:
            expint.move_robot(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")

    def toggle_led_callback(self):
        """Toggles the LED located on the robot handle."""
        if self.exp_connected:
            expint.toggle_led(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")

    def toggle_lights_callback(self):
        """Toggles the neopixel lights."""
        if self.exp_connected:
            expint.toggle_lights(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")
    
    def deliver_water_callback(self):
        """Opens the reward solenoid for the reward duration set 
        in experiment settings."""
        if self.exp_connected:
            expint.deliver_water(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")

    def flush_water_callback(self):
        """Opens the reward solenoid for the flush duration set in
        experiment settings."""
        if self.exp_connected:
            expint.flush_water(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")

    def run_protocol_callback(self):  
        """Initiates the user-selected experimental protocol."""
        if self.exp_connected and self.rob_connected:
            self.config = cfg.load_config('./temp/tmp_config.json')
            self.config['Protocol']['type'] = self.protocol.get()
            cfg.save_tmp(self.config)
            expint.stop_interface(self.exp_controller)
            self.exp_connected = False
            robint.stop_interface(self.rob_controller)
            self.rob_connected = False
            time.sleep(2)
            self.child = protocols.Protocols(self.window)
            self.protocol_running = True
        elif not self.exp_connected:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")
        else:
            tkinter.messagebox.showinfo("Warning", "Robot controller not connected.")                

if __name__ == '__main__':
    rm = ReachMaster()
    rm.run()