import serial
from serial.tools import list_ports

def start_interface(cfg):     
    exp_controller = serial.Serial(cfg['ReachMaster']['exp_control_path'],
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

def set_protocol(exp_controller, protocol):
    exp_controller.write("v")
    if exp_controller.read() == "v":
        exp_controller.write(protocol)
        if exp_controller.readline() == "v":
            exp_controller.write("1")

def var_read(exp_controller,varname):
    exp_controller.write("g")
    if exp_controller.read() == "g":
        exp_controller.write(varname+"\n")
        return exp_controller.readline()[:-2]

def var_write(exp_controller, varname, value):
    exp_controller.write("v")
    if exp_controller.read() == "v":
        exp_controller.write(varname+"\n")
        if exp_controller.read() == "v":
            exp_controller.write(value+"\n")