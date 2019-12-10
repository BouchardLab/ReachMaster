import numpy as np
import serial

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
    self.rob_controller.write("e")
    self.rob_controller.close()

def load_calibration_one(robot, varname, value):
    robot.write("c")
    if robot.read() == "c":
        robot.write(varname + "\n")
        if robot.read() == "c":
            robot.write(value)
    if robot.read() == "c":
        print(varname + ' loaded')
    else:
        raise Exception(varname)

def load_calibration_all(robot, cfg):
    try:
        load_calibration_one(robot,'dis',cfg['RobotSettings']['dis'])
        load_calibration_one(robot,'pos',cfg['RobotSettings']['pos'])
        load_calibration_one(robot,'xpush_dur',cfg['RobotSettings']['xpush_dur'])
        load_calibration_one(robot,'xpull_dur',cfg['RobotSettings']['xpull_dur'])
        load_calibration_one(robot,'ypush_dur',cfg['RobotSettings']['ypush_dur'])
        load_calibration_one(robot,'ypull_dur',cfg['RobotSettings']['ypull_dur'])
        load_calibration_one(robot,'zpush_dur',cfg['RobotSettings']['zpush_dur'])
        load_calibration_one(robot,'zpull_dur',cfg['RobotSettings']['zpull_dur'])
    except Exception as varname:
        raise Exception(varname) 

def load_commands_one(robot, varname, value):  
    robot.write("p")
    if robot.read() == "p":
        robot.write(varname + "\n")
        if robot.read() == "p":
            robot.write(value)
    if robot.read() == "p":
        print(varname + ' commands loaded')
    else:
        raise Exception(varname)     

def load_commands_all(robot, cfg):
    #extract robot settings
    Ly = cfg['RobotSettings']['Ly']
    Lz = cfg['RobotSettings']['Lz']
    Axx = cfg['RobotSettings']['Axx']
    Ayy = cfg['RobotSettings']['Ayy']
    Azz = cfg['RobotSettings']['Azz']
    x0 = cfg['RobotSettings']['x0']
    y0 = cfg['RobotSettings']['y0']
    z0 = cfg['RobotSettings']['z0'] 
    #derive desired commands     
    n = 100
    if cfg['RobotSettings']['command_type'] == "sample_continuous":
        r_low = cfg['RobotSettings']['r_low']
        r_high = cfg['RobotSettings']['r_high']
        theta_mag = cfg['RobotSettings']['theta_mag']
        r = r_low + (r_high-r_low)*np.random.uniform(low=0.0,high=1.0,size=(500*n))**(1.0/3.0)
        thetay = theta_mag*np.random.uniform(low=-1,high=1,size=500*n)
        thetaz = theta_mag*np.random.uniform(low=-1,high=1,size=500*n)
        theta = np.sqrt(thetay**2+thetaz**2)
        r = r[theta<=theta_mag][0:n]
        thetay = thetay[theta<=theta_mag][0:n]
        thetaz = thetaz[theta<=theta_mag][0:n]
    elif cfg['RobotSettings']['command_type'] == "sample_discrete":
        r_set,thetay_set,thetaz_set = np.loadtxt(cfg['RobotSettings']['command_file'],\
        skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
        rand_sample = np.random.choice(range(len(r_set)),replace=True,size=n)
        r = r_set[rand_sample]
        thetay = thetay_set[rand_sample]
        thetaz = thetaz_set[rand_sample]
    elif cfg['RobotSettings']['command_type'] == "from_file":
        r,thetay,thetaz = np.loadtxt(cfg['RobotSettings']['command_file'],\
        skiprows=1,delimiter=',',unpack=True,usecols=(1,2,3))
    else:
        raise Exception("Invalid command type.") 
    #pass commands though inverse kinematics       
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
    #convert data types
    x = np.array2string(Ax,formatter={'float_kind':lambda Ax: "%.1f" % Ax})
    y = np.array2string(Ay,formatter={'float_kind':lambda Ay: "%.1f" % Ay})
    z = np.array2string(Az,formatter={'float_kind':lambda Az: "%.1f" % Az})
    r = np.array2string(r,formatter={'float_kind':lambda r: "%.1f" % r})
    thetay = np.array2string(thetay,formatter={'float_kind':lambda thetay: "%.1f" % thetay})
    thetaz = np.array2string(thetaz,formatter={'float_kind':lambda thetaz: "%.1f" % thetaz})
    x = x[1:-1]+' '
    y = y[1:-1]+' '
    z = z[1:-1]+' '
    r = r[1:-1]+' '
    thetay = thetay[1:-1]+' '
    thetaz = thetaz[1:-1]+' '    
    #load commands to robot
    try:
        load_commands_one(robot, 'xCommandPos', x)
        load_commands_one(robot, 'yCommandPos', y)
        load_commands_one(robot, 'zCommandPos', z)
    except Exception as varname:
        raise Exception("Failed to load: " + varname)
    #record to config and return changes
    cfg['RobotSettings']['x'] = x
    cfg['RobotSettings']['y'] = y
    cfg['RobotSettings']['z'] = z
    cfg['RobotSettings']['r'] = r
    cfg['RobotSettings']['thetay'] = thetay
    cfg['RobotSettings']['thetaz'] = thetaz
    return cfg