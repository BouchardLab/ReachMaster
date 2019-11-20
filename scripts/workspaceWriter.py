import os
import numpy as np
import json

#specify default workspace 
workspace = {
             'ReachMaster':       {
                                   'dataDir':os.getcwd(),
                                   'workspaceFile':'Default',
                                   'expControlPath':'/dev/ttyACM0',
                                   'robControlPath':'/dev/ttyACM1',
                                   'serialBaud':2000000,
                                   'controlTimeout':5
                                   },
             'CameraSettings':    {
                                   'numCams':3,
                                   'imgdataformat':'XI_RAW8',
                                   'fps':200,
                                   'exposure':2000,
                                   'gain':15.0,
                                   'sensor_feature_value':1,
                                   'gpi_selector':'XI_GPI_PORT1',
                                   'gpi_mode':'XI_GPI_TRIGGER',
                                   'trigger_source':'XI_TRG_EDGE_RISING',
                                   'gpo_selector':'XI_GPO_PORT1',
                                   'gpo_mode':'XI_GPO_EXPOSURE_ACTIVE',
                                   'imgWidth':688,
                                   'imgHeight':688,
                                   'offsetX':304,
                                   'offsetY':168,
                                   'downsampling':'XI_DWN_1x1',
                                   'savedPOIs':[],
                                   'poiThreshold':15,
                                   'vidMode':'TRIALS'                                   
                                   },
             'ExperimentSettings':{
                                   'baselineDur':5.0,
                                   'bufferDur':0.5,
                                   'reachTimeout':4000,
                                   'flushDur':10000,
                                   'solenoidOpenDur':75,
                                   'solenoidBounceDur':500,
                                   'rewardWinDur':3000,
                                   'maxRewards':3,
                                   'lightsOffDur':3000,
                                   'lightsOnDur':5000,
                                   'reachDelay':100
                                   },
             'RobotSettings':     {
                                   'calibrationFile':'None',
                                   'dis':'None',
                                   'pos':'None',
                                   'xPushDur':'None',
                                   'xPullDur':'None',
                                   'yPushDur':'None',
                                   'yPullDur':'None',
                                   'zPushDur':'None',
                                   'zPullDur':'None',
                                   'commandType':'None',
                                   'commandSet':'None',
                                   'commandFile':'None',
                                   'xCommandPos':'None',
                                   'yCommandPos':'None',
                                   'zCommandPos':'None',
                                   'rCommandPos':'None',
                                   'thetayCommandPos':'None',
                                   'thetazCommandPos':'None',
                                   'rLow':10,
                                   'rHigh':40,
                                   'thetaMag':np.pi/3,
                                   'Ly':64,
                                   'Lz':47,
                                   'Axx':168,
                                   'Ayy':100,
                                   'Azz':117,
                                   'x0':1024,
                                   'y0':608,
                                   'z0':531,
                                   'alpha':0.1,
                                   'tol':np.round(1023.0/50.0/3.0,decimals=1),
                                   'period':125.0*1000.0,
                                   'offDur':1000,
                                   'numTol':5,
                                   'xPushWt':1.0,
                                   'xPullWt':1.0,
                                   'yPushWt':1.0,
                                   'yPullWt':1.0,
                                   'zPushWt':1.0,
                                   'zPullWt':1.0,
                                   'RZx':1000,
                                   'RZy_low':558,
                                   'RZy_high':658,
                                   'RZz_low':481,
                                   'RZz_high':581
                                   }
             }

workspace['CameraSettings']['output_params'] = {
                                                "-vcodec":"libx264", 
                                                "-crf": 28,
                                                "-preset":"ultrafast", 
                                                "-tune":"zerolatency",
                                                "-output_dimensions": (
                                                                       workspace['CameraSettings']['numCams']*
                                                                       workspace['CameraSettings']['imgWidth'],
                                                                       workspace['CameraSettings']['imgHeight']
                                                                       )
                                               }

with open('defaultWorkspace.txt', 'w') as outfile:
    json.dump(workspace, outfile, indent=4)
