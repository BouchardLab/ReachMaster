import serial

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