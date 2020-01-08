"""This module provides a set of functions for interfacing
with the experiment microcontroller. It provides functions
to recognize and establish serial connections to the 
controller, read/write settings, exectute common callbacks,
and read/write custom messages. 

Todo:
    * Object orient
    * Automate unit tests
    * Python 3 compatibility
    * PEP 8

"""

import serial
from serial.tools import list_ports

#private functions-----------------------------------------------------------

def _variable_read(exp_controller, varname):
    exp_controller.write("g")
    if exp_controller.read() == "g":
        exp_controller.write(varname+"\n")
        return exp_controller.readline()[:-2]

def _variable_write(exp_controller, varname, value):
    exp_controller.write("v")
    if exp_controller.read() == "v":
        exp_controller.write(varname+"\n")
        if exp_controller.read() == "v":
            exp_controller.write(value+"\n")

#public functions------------------------------------------------------------

def start_interface(config): 
    """Establish a serial connection with the experiment
    controller.

    Parameters
    ----------
    config : dict
        The currently loaded configuration file.

    Returns
    -------
    exp_controller : serial.serialposix.Serial  
        The serial interface to the experiment controller.      

    """    
    exp_controller = serial.Serial(config['ReachMaster']['exp_control_port'],
        config['ReachMaster']['serial_baud'],
        timeout=config['ReachMaster']['control_timeout'])
    exp_controller.flushInput()
    exp_controller.write("h")
    response = exp_controller.read()
    if response=="h":
        return exp_controller
    else:
        raise Exception("Experiment controller failed to connect.")

def stop_interface(exp_controller):
    """Perform a soft reboot of the experiment controller
    and close the serial connection.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("e")
    exp_controller.close()

def get_ports():
    """List all serial port with connected devices.

    Returns
    -------
    port_list : list(str)  
        Names of all the serial ports with connected devices. 

    """
    port_list = list(list_ports.comports())
    for i in range(len(port_list)):
        port_list[i] = port_list[i].device
    return port_list

#Simple callbacks -----------------------------------------------------------

def move_robot(exp_controller):
    """Send a message to experiment controller to command the
    robot to move to the next position in its command 
    sequence.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("m")

def toggle_led(exp_controller):
    """Send a message to experiment controller to toggle the 
    LED located on the robot handle.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("l")          

def toggle_lights(exp_controller):
    """Send a message to experiment controller to toggle the 
    neopixel lights.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("n")

def deliver_water(exp_controller):
    """Send a message to experiment controller to open the 
    reward solenoid for the user-selected duration.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("w")

def flush_water(exp_controller):
    """Send a message to experiment controller to opens the 
    reward solenoid for the flush user-selected flush 
    duration.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("f")

def trigger_image(exp_controller):
    """Send a message to experiment controller to trigger the 
    cameras to capture an image.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    """
    exp_controller.write("t")

#Typically by protocols ------------------------------------------------------------------

def set_exp_controller(exp_controller, config):
    """Load the experiment settings to the experiment controller.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.
    config : dict
        The currently loaded configuration file.

    """
    _variable_write(exp_controller, 'lights_on_dur', str(config['ExperimentSettings']['lights_on_dur']))
    _variable_write(exp_controller, 'lights_off_dur', str(config['ExperimentSettings']['lights_off_dur']))
    _variable_write(exp_controller, 'reward_win_dur', str(config['ExperimentSettings']['reward_win_dur']))
    _variable_write(exp_controller, 'max_rewards', str(config['ExperimentSettings']['max_rewards']))
    _variable_write(exp_controller, 'solenoid_open_dur', str(config['ExperimentSettings']['solenoid_open_dur']))
    _variable_write(exp_controller, 'solenoid_bounce_dur', str(config['ExperimentSettings']['solenoid_bounce_dur']))
    _variable_write(exp_controller, 'flush_dur', str(config['ExperimentSettings']['flush_dur']))
    _variable_write(exp_controller, 'reach_delay', str(config['ExperimentSettings']['reach_delay']))
    _variable_write(exp_controller, 'protocol', config['Protocol']['type'])

def start_experiment(exp_controller):    
    """Send a message to experiment controller to begin 
    executing an experiment protocol.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    Returns
    -------
    response : str
        The first line of data received from the 
        experiment controller verifying that the protocol
        has been initiated.

    """
    exp_controller.write("b")
    while not exp_controller.in_waiting:
        pass
    response = exp_controller.readline().split()
    exp_controller.flushInput()  
    print('trials completed:')
    print(response[0])
    return response

def read_response(exp_controller):
    """Read a line of data from the experiment controller
    in response to the most recently written message.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.

    Returns
    -------
    response : str
        The line of data received from the experiment 
        controller in response to the most recently 
        written message.

    """
    response = exp_controller.readline()
    exp_controller.flushInput()
    return response

def write_message(exp_controller, message):
    """Read a line of data from the experiment controller
    in response to the most recently written message.

    Parameters
    ----------
    exp_controller : serial.serialposix.Serial
        The serial interface to the experiment controller.
    message : str
        A string that is recognizeable by the experiment
        controller.

    """
    exp_controller.write(message)
    while not exp_controller.in_waiting:
        pass