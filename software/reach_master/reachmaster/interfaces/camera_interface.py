"""This module provides a set of functions for interfacing
with multiple Ximea machine vision USB3.0 cameras via the
Ximea API. It provides functions start/stop the cameras 
using the user-selected settings, and to get images from
the cameras. 

Todo:
    * Object orient
    * GPU-accelerated video encoding
    * Automate unit tests
    * PEP 8

"""

from ximea import xiapi

#private functions -----------------------------------------------------------------

def _open_cameras(config):              
    cams = []
    for i in range(config['CameraSettings']['num_cams']):
        print(('loading camera %s ...' %(i)))
        cams.append(xiapi.Camera(dev_id = i))
        cams[i].open_device()     
    return cams

def _set_cameras(cams, config):
    for i in range(config['CameraSettings']['num_cams']):
        print(('Setting camera %d ...' %i))
        cams[i].set_imgdataformat(config['CameraSettings']['imgdataformat'])
        cams[i].set_exposure(config['CameraSettings']['exposure'])
        cams[i].set_gain(config['CameraSettings']['gain'])
        cams[i].set_sensor_feature_value(config['CameraSettings']['sensor_feature_value'])
        cams[i].set_gpi_selector(config['CameraSettings']['gpi_selector'])
        cams[i].set_gpi_mode(config['CameraSettings']['gpi_mode'])
        cams[i].set_trigger_source(config['CameraSettings']['trigger_source'])
        cams[i].set_gpo_selector(config['CameraSettings']['gpo_selector'])
        cams[i].set_gpo_mode(config['CameraSettings']['gpo_mode'])        
        if config['CameraSettings']['downsampling'] == "XI_DWN_2x2":
            cams[i].set_downsampling(config['CameraSettings']['downsampling'])
        else:
            widthIncrement = cams[i].get_width_increment()
            heightIncrement = cams[i].get_height_increment()
            if (config['CameraSettings']['img_width']%widthIncrement)!=0:
                raise Exception("Image width not divisible by "+str(widthIncrement))
                return
            elif (config['CameraSettings']['img_height']%heightIncrement)!=0:
                raise Exception("Image height not divisible by "+str(heightIncrement))
                return
            elif (config['CameraSettings']['img_width']+config['CameraSettings']['offset_x'])>1280:
                raise Exception("Image width + x offset > 1280") 
                return
            elif (config['CameraSettings']['img_height']+config['CameraSettings']['offset_y'])>1024:
                raise Exception("Image height + y offset > 1024") 
                return
            else:
                cams[i].set_height(config['CameraSettings']['img_height'])
                cams[i].set_width(config['CameraSettings']['img_width'])
                cams[i].set_offsetX(config['CameraSettings']['offset_x'])
                cams[i].set_offsetY(config['CameraSettings']['offset_y'])
        cams[i].enable_recent_frame()

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

