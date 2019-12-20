import serial
from serial.tools import list_ports

def start_interface(cfg):     
    exp_controller = serial.Serial(cfg['ReachMaster']['exp_control_port'],
        cfg['ReachMaster']['serial_baud'],
        timeout=cfg['ReachMaster']['control_timeout'])
    exp_controller.flushInput()
    exp_controller.write("h")
    response = exp_controller.read()
    if response=="h":
        return exp_controller
    else:
        raise Exception("Experiment controller failed to connect.")

def stop_interface(exp_controller):
    exp_controller.write("e")
    exp_controller.close()

def get_ports():
    port_list = list(list_ports.comports())
    for i in range(len(port_list)):
        port_list[i] = port_list[i].device
    return port_list

def move_robot(exp_controller):
    exp_controller.write("m")

def toggle_led(exp_controller):
    exp_controller.write("l")          

def toggle_lights(exp_controller):
    exp_controller.write("n")

def deliver_water(exp_controller):
    exp_controller.write("w")

def flush_water(exp_controller):
    exp_controller.write("f")

def trigger_image(exp_controller):
    exp_controller.write("t")

def variable_read(exp_controller, varname):
    exp_controller.write("g")
    if exp_controller.read() == "g":
        exp_controller.write(varname+"\n")
        return exp_controller.readline()[:-2]

def variable_write(exp_controller, varname, value):
    exp_controller.write("v")
    if exp_controller.read() == "v":
        exp_controller.write(varname+"\n")
        if exp_controller.read() == "v":
            exp_controller.write(value+"\n")

def set_exp_controller(exp_controller, cfg):
    variable_write(exp_controller, 'lights_on_dur', str(cfg['ExperimentSettings']['lights_on_dur']))
    variable_write(exp_controller, 'lights_off_dur', str(cfg['ExperimentSettings']['lights_off_dur']))
    variable_write(exp_controller, 'reward_win_dur', str(cfg['ExperimentSettings']['reward_win_dur']))
    variable_write(exp_controller, 'max_rewards', str(cfg['ExperimentSettings']['max_rewards']))
    variable_write(exp_controller, 'solenoid_open_dur', str(cfg['ExperimentSettings']['solenoid_open_dur']))
    variable_write(exp_controller, 'solenoid_bounce_dur', str(cfg['ExperimentSettings']['solenoid_bounce_dur']))
    variable_write(exp_controller, 'flush_dur', str(cfg['ExperimentSettings']['flush_dur']))
    variable_write(exp_controller, 'reach_delay', str(cfg['ExperimentSettings']['reach_delay']))
    variable_write(exp_controller, 'protocol', cfg['Protocol']['type'])

def start_experiment(exp_controller):    
    exp_controller.write("b")
    while not exp_controller.in_waiting:
        pass
    response = exp_controller.readline().split()
    exp_controller.flushInput()  
    print('trials completed:')
    print(response[0])
    return response

def read_message(exp_controller):
    response = exp_controller.readline()
    exp_controller.flushInput()
    return response

def write_message(exp_controller, message):
    exp_controller.write(message)
    while not exp_controller.in_waiting:
        pass