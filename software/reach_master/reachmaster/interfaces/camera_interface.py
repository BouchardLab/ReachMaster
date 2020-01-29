"""This module provides a set of functions for interfacing
with Ximea machine vision USB3.0 cameras via the
Ximea API. It establishes camera connections, application
of user-selected settings, image acquisition, debayering,
and encoding to video. Video encoding can be performed as 
either a single or multiprocess. For multiprocess,
fast NVIDIA gpu-accelerated encoding is performed using 
CUDA-enabled ffmpeg. 

Todo:
    * Object orient
    * GPU-accelerated video encoding
    * Automate unit tests

"""

from ximea import xiapi
import cv2
import numpy as np
import subprocess as sp
import multiprocessing as mp
from ctypes import c_bool
from time import time, sleep
import datetime
import numpy as np

#private functions -----------------------------------------------------------------

def _set_camera(cam, config):
        cam.set_imgdataformat(config['CameraSettings']['imgdataformat'])
        cam.set_exposure(config['CameraSettings']['exposure'])
        cam.set_gain(config['CameraSettings']['gain'])
        cam.set_sensor_feature_value(config['CameraSettings']['sensor_feature_value'])
        cam.set_gpi_selector(config['CameraSettings']['gpi_selector'])
        cam.set_gpi_mode(config['CameraSettings']['gpi_mode'])
        cam.set_trigger_source(config['CameraSettings']['trigger_source'])
        cam.set_gpo_selector(config['CameraSettings']['gpo_selector'])
        cam.set_gpo_mode(config['CameraSettings']['gpo_mode'])        
        if config['CameraSettings']['downsampling'] == "XI_DWN_2x2":
            cam.set_downsampling(config['CameraSettings']['downsampling'])
        else:
            widthIncrement = cam.get_width_increment()
            heightIncrement = cam.get_height_increment()
            if (config['CameraSettings']['img_width'] % widthIncrement) != 0:
                raise Exception(
                    "Image width not divisible by " + str(widthIncrement)
                    )
                return
            elif (config['CameraSettings']['img_height'] % heightIncrement) != 0:
                raise Exception(
                    "Image height not divisible by " + str(heightIncrement) 
                    )
                return
            elif (
                config['CameraSettings']['img_width'] + 
                config['CameraSettings']['offset_x']
                ) > 1280:
                raise Exception("Image width + x offset > 1280") 
                return
            elif (
                config['CameraSettings']['img_height'] + 
                config['CameraSettings']['offset_y']
                ) > 1024:
                raise Exception("Image height + y offset > 1024") 
                return
            else:
                cam.set_height(config['CameraSettings']['img_height'])
                cam.set_width(config['CameraSettings']['img_width'])
                cam.set_offsetX(config['CameraSettings']['offset_x'])
                cam.set_offsetY(config['CameraSettings']['offset_y'])
        cam.enable_recent_frame()

def _set_cameras(cams, config):
    for i in range(config['CameraSettings']['num_cams']):
        print(('Setting camera %d ...' %i))
        _set_camera(cams[i], config)

def _open_cameras(config):              
    cams = []
    for i in range(config['CameraSettings']['num_cams']):
        print(('loading camera %s ...' %(i)))
        cams.append(xiapi.Camera(dev_id = i))
        cams[i].open_device()     
    return cams

def _start_cameras(cams):
    for i in range(len(cams)):
        print(('Starting camera %d ...' %i))
        cams[i].start_acquisition()

#public functions -----------------------------------------------------------------

def stop_interface(cams):
    """Stop image acquisition and close all cameras.

    Parameters
    ----------
    cams : list
        A list of ximea api Camera objects.

    """
    for i in range(len(cams)):
        print(('stopping camera %d ...' %i))
        cams[i].stop_acquisition()
        cams[i].close_device()

def start_interface(config):
    """Open all cameras, loads user-selected settings, and
    starts image acquisition.

    Parameters
    ----------
    config : dict
        The currently loaded configuration file.

    Returns
    -------
    cams : list
        A list of ximea api Camera objects.

    """
    cams = _open_cameras(config)
    _set_cameras(cams, config)
    try:
        _start_cameras(cams)     
    except xiapi.Xi_error as err:
        expActive = False
        stop_interface(cams)
        if err.status == 10:
            raise Exception("No image triggers detected.")
            return
        raise Exception(err)   
    return cams

def init_image():
    """Initialize a ximea container object to store images.

    Returns
    -------
    img : ximea.xiapi.Image
        A ximea api Image container object

    """ 
    img = xiapi.Image()
    return img

def get_npimage(cam, img):
    """Get the most recent image from a camera as a numpy 
    array.

    Parameters
    ----------
    cam : ximea.xiapi.Camera
        A ximea api Camera object.
    img : ximea.xiapi.Image
        A ximea api Image container object

    Returns
    -------
    npimg : numpy.ndarray
        The most recently acquired image from the camera 
        as a numpy array.

    """
    cam.get_image(img, timeout = 2000)                  
    npimg = img.get_image_data_numpy()
    return npimg

class CameraInterface:

    def __init__(self, config):
        self.config = config
        self.ffmpeg_command = [
        'ffmpeg', '-y', 
        '-hwaccel', 'cuvid', 
        '-f', 'rawvideo',  
        '-s', str(config['CameraSettings']['img_width']) + 'x' + 
        str(config['CameraSettings']['img_height']), 
        '-pix_fmt', 'bgr24',
        '-r', str(config['CameraSettings']['fps']), 
        '-i', '-',
        '-b:v', '2M', 
        '-maxrate', '2M', 
        '-bufsize', '1M',
        '-c:v', 'h264_nvenc', 
        '-preset', 'llhp', 
        '-profile:v', 'high',
        '-rc', 'cbr', 
        '-pix_fmt', 'yuv420p'
        ]
        self.camera_processes = []
        self.cam_trigger_pipes = []
        self.poi_deviation_pipes = []
        self.cams_started = mp.Value(c_bool, False)
        self.increment_trial = mp.Value(c_bool, False)
        # self.cams_triggered = mp.Value(c_bool, False)
        # self.cams_triggered = mp.Array(
        #     c_bool, 
        #     [False]*self.config['CameraSettings']['num_cams'], 
        #     )    
        # self.poi_deviations = mp.Array(
        #     'i', 
        #     [0]*self.config['CameraSettings']['num_cams'], 
        #     )            

    def start_protocol_interface(self):
        print('starting camera processes... ')
        for cam_id in range(self.config['CameraSettings']['num_cams']):
            trigger_parent, trigger_child = mp.Pipe()
            self.cam_trigger_pipes.append(trigger_parent)
            poi_parent, poi_child = mp.Pipe()
            self.poi_deviation_pipes.append(poi_parent)
            self.camera_processes.append(
                mp.Process(
                    target = self._protocol_process, 
                    args = (
                        cam_id,
                        trigger_child,
                        poi_child
                        )
                    )
                )
        self.cams_started.value = True   
        for process in self.camera_processes:
            process.start()
        sleep(5) #give cameras time to setup       

    def stop_interface(self):
        self.cams_started.value = False
        for proc in self.camera_processes:
            proc.join()
        self.cam_trigger_pipes = []
        self.poi_deviation_pipes = []
        # self.cams_triggered[:] = [False for cam in self.cams_triggered]
        # self.poi_deviations[:] = [0 for cam in self.poi_deviations]

    def triggered(self):
        for pipe in self.cam_trigger_pipes:
            pipe.send('p')

    def all_triggerable(self):
        print('before trig')
        if all([pipe.poll() for pipe in self.cam_trigger_pipes]):
            for pipe in self.cam_trigger_pipes:
                pipe.recv()
            print('after trig success')
            return True
        else: 
            return False

    def get_poi_deviation(self):
        print('before dev')
        deviations = [pipe.recv() for pipe in self.poi_deviation_pipes]
        print('after dev')
        return min(deviations)

    def trial_ended(self):
        self.increment_trial.value = True

    def _acquire_baseline(self, cam, cam_id, img, num_imgs, trigger_pipe):
        poi_indices = self.config['CameraSettings']['saved_pois'][cam_id]
        num_pois = len(poi_indices)
        baseline_pois = np.zeros(shape = (num_pois, num_imgs))
        trigger_pipe.send(1)
        #acquire baseline images 
        for i in range(num_imgs):  
            while not trigger_pipe.poll():
                pass 
            trigger_pipe.recv()
            try:
                cam.get_image(img, timeout = 2000) 
                trigger_pipe.send('c') 
                npimg = img.get_image_data_numpy()   
                for j in range(num_pois): 
                    baseline_pois[j,i] = npimg[
                    poi_indices[j][1],
                    poi_indices[j][0]
                    ]
            except Exception as err:
                print("error: "+str(cam_id))
                print(err)
        #compute summary stats
        poi_means = np.mean(baseline_pois, axis = 1)            
        poi_std = np.std(
            np.sum(
                np.square(
                    baseline_pois - 
                    poi_means.reshape(num_pois, 1)
                    ), 
                axis = 0
                )
            )
        return poi_means, poi_std

    def _estimate_poi_deviation(self, cam_id, npimg, poi_means, poi_std):
        poi_indices = self.config['CameraSettings']['saved_pois'][cam_id]
        num_pois = len(poi_indices)
        poi_obs = np.zeros(num_pois)
        for j in range(num_pois): 
            poi_obs[j] = npimg[poi_indices[j][1], poi_indices[j][0]]
        dev = int(
            np.sum(np.square(poi_obs - poi_means)) / (poi_std + np.finfo(float).eps)
            )
        return dev

    def _protocol_process(self, cam_id, trigger_pipe, poi_deviation_pipe):
        sleep(2*cam_id) #prevents simultaneous calls to the ximea api
        print('finding camera: ', cam_id)
        cam = xiapi.Camera(dev_id = cam_id)
        print('opening camera: ', cam_id)
        cam.open_device()
        print('setting camera: ', cam_id)
        _set_camera(cam, self.config)
        print('starting camera: ', cam_id)
        cam.start_acquisition()
        img = xiapi.Image()
        num_baseline = (
            int(
                np.round(
                    float(
                        self.config['ExperimentSettings']['baseline_dur']
                        ) * 
                    float(
                        self.config['CameraSettings']['fps']
                        ), 
                    decimals = 0
                    )
                )
            )
        if num_baseline > 0:
            poi_means, poi_std = self._acquire_baseline(
                cam,
                cam_id,
                img,
                num_baseline,
                trigger_pipe
                ) 
        if self.config['Protocol']['type'] == 'CONTINUOUS':
            vid_fn = (
                self.config['ReachMaster']['data_dir'] + '/videos/' +
                str(datetime.datetime.now()) + '_cam' + str(cam_id) + '.mp4'
                )
        elif self.config['Protocol']['type'] == 'TRIALS':
            trial_num = 0
            vid_fn = (
                self.config['ReachMaster']['data_dir'] + '/videos/trial' +
                str(trial_num) + '_cam' + str(cam_id) + '.mp4'
                )        
        self.ffmpeg_command.append(vid_fn)
        ffmpeg_process = sp.Popen(
            self.ffmpeg_command, 
            stdin=sp.PIPE, 
            stdout=sp.DEVNULL, 
            stderr=sp.DEVNULL, 
            bufsize=-1
            )
        while self.cams_started.value == True:        
            if trigger_pipe.poll():
                trigger_pipe.recv()
                print('cam_id: '+str(cam_id)+', trigger received')
                try:
                    cam.get_image(img, timeout = 2000) 
                    trigger_pipe.send('c')
                    npimg = img.get_image_data_numpy()  
                    print('cam_id: ' + str(cam_id) + ', pipe sent')    
                    frame = cv2.cvtColor(npimg, cv2.COLOR_BAYER_BG2BGR)
                    ffmpeg_process.stdin.write(frame)
                    print('cam_id: '+str(cam_id)+' getting dev')
                    dev = self._estimate_poi_deviation(cam_id, npimg, poi_means, poi_std)      
                    poi_deviation_pipe.send(dev)
                except Exception as err:
                    print("cam_id: " + str(err))
                    pass
            elif self.config['Protocol']['type'] == 'TRIALS' and self.increment_trial.value:
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()
                ffmpeg_process = None
                trial_num += 1
                vid_fn = (
                self.config['ReachMaster']['data_dir'] + '/videos/trial' +
                str(trial_num) + '_cam' + str(cam_id) + '.mp4'
                )
                self.ffmpeg_command[-1] = vid_fn
                ffmpeg_process = sp.Popen(
                    self.ffmpeg_command, 
                    stdin=sp.PIPE, 
                    stdout=sp.DEVNULL, 
                    stderr=sp.DEVNULL, 
                    bufsize=-1
                    )
        cam.stop_acquisition()
        cam.close_device()
        if ffmpeg_process is not None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            ffmpeg_process = None

# if __name__ == '__main__':
#     #for debugging purposes only 
#     ctx = mp.set_start_method('spawn')
#     config = {
#     'CameraSettings': {
#     'num_cams': 3
#     }
#     }
#     interface = CameraInterface(config)
#     interface.start_recording()
#     sleep(30)
#     interface.stop_recording()