import numpy as np
import serial
from serial.tools import list_ports

def start_interface(cfg):
        rob_controller = serial.Serial(cfg['ReachMaster']['rob_control_path'],
            cfg['ReachMaster']['serial_baud'],
            timeout=cfg['ReachMaster']['control_timeout'])
        rob_controller.flushInput()
        rob_controller.write("h")
        response = rob_controller.read()
        if response == "h":
            rob_connected = True
            return rob_controller
        else:
            raise Exception("Robot controller failed to connect.")

def stop_interface(rob_controller):    
    rob_controller.write("e")
    rob_controller.close()

def get_ports():
    port_list = list(list_ports.comports())
    for i in range(len(port_list)):
        port_list[i] = port_list[i].device
    return port_list

def load_calibration_var(robot, varname, value):
    robot.write("c")
    if robot.read() == "c":
        robot.write(varname + "\n")
        if robot.read() == "c":
            robot.write(value)
    if robot.read() == "c":
        print(varname + ' loaded')
    else:
        raise Exception(varname)

def load_config_calibration(robot, cfg):
    try:
        load_calibration_var(robot,'dis',cfg['RobotSettings']['dis'])
        load_calibration_var(robot,'pos',cfg['RobotSettings']['pos'])
        load_calibration_var(robot,'x_push_dur',cfg['RobotSettings']['x_push_dur'])
        load_calibration_var(robot,'x_pull_dur',cfg['RobotSettings']['x_pull_dur'])
        load_calibration_var(robot,'y_push_dur',cfg['RobotSettings']['y_push_dur'])
        load_calibration_var(robot,'y_pull_dur',cfg['RobotSettings']['y_pull_dur'])
        load_calibration_var(robot,'z_push_dur',cfg['RobotSettings']['z_push_dur'])
        load_calibration_var(robot,'z_pull_dur',cfg['RobotSettings']['z_pull_dur'])
    except Exception as varname:
        raise Exception(varname) 

def load_commands_var(robot, varname, value):  
    robot.write("p")
    if robot.read() == "p":
        robot.write(varname + "\n")
        if robot.read() == "p":
            robot.write(value)
    if robot.read() == "p":
        print(varname + ' commands loaded')
    else:
        raise Exception(varname)     

def load_config_commands(robot, cfg):
    #extract robot settings
    Ly = cfg['RobotSettings']['Ly']
    Lz = cfg['RobotSettings']['Lz']
    xgimbal_xoffset = cfg['RobotSettings']['xgimbal_xoffset']
    ygimbal_yoffset = cfg['RobotSettings']['ygimbal_yoffset']
    zgimbal_zoffset = cfg['RobotSettings']['zgimbal_zoffset']
    x_origin = cfg['RobotSettings']['x_origin']
    y_origin = cfg['RobotSettings']['y_origin']
    z_origin = cfg['RobotSettings']['z_origin'] 
    #derive desired commands     
    n = 100
    if cfg['RobotSettings']['command_source'] == "parametric_sample":
        reach_dist_min = cfg['RobotSettings']['reach_dist_min']
        reach_dist_max = cfg['RobotSettings']['reach_dist_max']
        reach_angle_max = cfg['RobotSettings']['reach_angle_max']
        r = reach_dist_min + (reach_dist_max-reach_dist_min)*np.random.uniform(low=0.0,high=1.0,size=(500*n))**(1.0/3.0)
        theta_y = reach_angle_max*np.random.uniform(low=-1,high=1,size=500*n)
        theta_z = reach_angle_max*np.random.uniform(low=-1,high=1,size=500*n)
        theta = np.sqrt(theta_y**2+theta_z**2)
        r = r[theta<=reach_angle_max][0:n]
        theta_y = theta_y[theta<=reach_angle_max][0:n]
        theta_z = theta_z[theta<=reach_angle_max][0:n]
    elif cfg['RobotSettings']['command_source'] == "sample_from_file":
        r_set,theta_y_set,theta_z_set = np.loadtxt(cfg['RobotSettings']['command_file'],\
        skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
        rand_sample = np.random.choice(range(len(r_set)),replace=True,size=n)
        r = r_set[rand_sample]
        theta_y = theta_y_set[rand_sample]
        theta_z = theta_z_set[rand_sample]
    elif cfg['RobotSettings']['command_source'] == "read_from_file":
        r,theta_y,theta_z = np.loadtxt(cfg['RobotSettings']['command_file'],\
        skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
    else:
        raise Exception("Invalid command type.") 
    #pass commands though inverse kinematics       
    Ax = np.sqrt(xgimbal_xoffset**2+r**2-2*xgimbal_xoffset*r*np.cos(theta_y)*np.cos(theta_z))
    gammay = -np.arcsin(np.sin(theta_y)*np.sqrt((r*np.cos(theta_y)*np.cos(theta_z))**2+\
        (r*np.sin(theta_y)*np.cos(theta_z))**2)/np.sqrt((xgimbal_xoffset-r*np.cos(theta_y)*\
            np.cos(theta_z))**2+(r*np.sin(theta_y)*np.cos(theta_z))**2))
    gammaz = -np.arcsin(r*np.sin(theta_z)/Ax)
    Ay = np.sqrt((Ly-Ly*np.cos(gammay)*np.cos(gammaz))**2+\
        (ygimbal_yoffset-Ly*np.sin(gammay)*np.cos(gammaz))**2+(Ly*np.sin(gammaz))**2)
    Az = np.sqrt((Lz-Lz*np.cos(gammay)*np.cos(gammaz))**2+\
        (Lz*np.sin(gammay)*np.cos(gammaz))**2+(zgimbal_zoffset-Lz*np.sin(gammaz))**2)
    Ax = np.round((Ax-xgimbal_xoffset)/50*1024+x_origin,decimals=1)
    Ay = np.round((Ay-ygimbal_yoffset)/50*1024+y_origin,decimals=1)
    Az = np.round((Az-zgimbal_zoffset)/50*1024+z_origin,decimals=1)
    #convert data types
    x = np.array2string(Ax,formatter={'float_kind':lambda Ax: "%.1f" % Ax})
    y = np.array2string(Ay,formatter={'float_kind':lambda Ay: "%.1f" % Ay})
    z = np.array2string(Az,formatter={'float_kind':lambda Az: "%.1f" % Az})
    r = np.array2string(r,formatter={'float_kind':lambda r: "%.1f" % r})
    theta_y = np.array2string(theta_y,formatter={'float_kind':lambda theta_y: "%.1f" % theta_y})
    theta_z = np.array2string(theta_z,formatter={'float_kind':lambda theta_z: "%.1f" % theta_z})
    x = x[1:-1]+' '
    y = y[1:-1]+' '
    z = z[1:-1]+' '
    r = r[1:-1]+' '
    theta_y = theta_y[1:-1]+' '
    theta_z = theta_z[1:-1]+' '    
    #load commands to robot
    try:
        load_commands_var(robot, 'x_command_pos', x)
        load_commands_var(robot, 'y_command_pos', y)
        load_commands_var(robot, 'z_command_pos', z)
    except Exception as varname:
        raise Exception("Failed to load: " + varname)
    #record to config and return changes
    cfg['RobotSettings']['x'] = x
    cfg['RobotSettings']['y'] = y
    cfg['RobotSettings']['z'] = z
    cfg['RobotSettings']['r'] = r
    cfg['RobotSettings']['theta_y'] = theta_y
    cfg['RobotSettings']['theta_z'] = theta_z
    return cfg

def var_read(rob_controller, varname):
    rob_controller.write("g")
    if rob_controller.read() == "g":
        rob_controller.write(varname+"\n")
        return rob_controller.readline()[:-2]

def var_write(rob_controller, varname, value):
    rob_controller.write("v")
    if rob_controller.read() == "v":
        rob_controller.write(varname+"\n")
        if rob_controller.read() == "v":
            rob_controller.write(value+"\n")
