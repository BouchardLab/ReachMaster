from ximea import xiapi

def open_cameras(cfg):              
    cams = []
    for i in range(cfg['CameraSettings']['num_cams']):
        print('loading camera %s ...' %(i))
        cams.append(xiapi.Camera(dev_id = i))
        cams[i].open_device()     
    return cams

def set_cameras(cams,cfg):
    for i in range(cfg['CameraSettings']['num_cams']):
        print('Setting camera %d ...' %i)
        cams[i].set_imgdataformat(cfg['CameraSettings']['imgdataformat'])
        cams[i].set_exposure(cfg['CameraSettings']['exposure'])
        cams[i].set_gain(cfg['CameraSettings']['gain'])
        cams[i].set_sensor_feature_value(cfg['CameraSettings']['sensor_feature_value'])
        cams[i].set_gpi_selector(cfg['CameraSettings']['gpi_selector'])
        cams[i].set_gpi_mode(cfg['CameraSettings']['gpi_mode'])
        cams[i].set_trigger_source(cfg['CameraSettings']['trigger_source'])
        cams[i].set_gpo_selector(cfg['CameraSettings']['gpo_selector'])
        cams[i].set_gpo_mode(cfg['CameraSettings']['gpo_mode'])        
        if cfg['CameraSettings']['downsampling'] == "XI_DWN_2x2":
            cams[i].set_downsampling(cfg['CameraSettings']['downsampling'])
        else:
            widthIncrement = cams[i].get_width_increment()
            heightIncrement = cams[i].get_height_increment()
            if (cfg['CameraSettings']['img_width']%widthIncrement)!=0:
                raise Exception("Image width not divisible by "+str(widthIncrement))
                return
            elif (cfg['CameraSettings']['img_height']%heightIncrement)!=0:
                raise Exception("Image height not divisible by "+str(heightIncrement))
                return
            elif (cfg['CameraSettings']['img_width']+cfg['CameraSettings']['offset_x'])>1280:
                raise Exception("Image width + x offset > 1280") 
                return
            elif (cfg['CameraSettings']['img_height']+cfg['CameraSettings']['offset_y'])>1024:
                raise Exception("Image height + y offset > 1024") 
                return
            else:
                cams[i].set_height(cfg['CameraSettings']['img_height'])
                cams[i].set_width(cfg['CameraSettings']['img_width'])
                cams[i].set_offsetX(cfg['CameraSettings']['offset_x'])
                cams[i].set_offsetY(cfg['CameraSettings']['offset_y'])
        cams[i].enable_recent_frame()

def start_cameras(cams):
    for i in range(len(cams)):
        print('Starting camera %d ...' %i)
        cams[i].start_acquisition()

def start_interface(cfg):
    cams = open_cameras(cfg)
    set_cameras(cams,cfg)
    try:
        start_cameras(cams)     
    except xiapi.Xi_error as err:
        expActive = False
        camint.stop_interface()
        if err.status == 10:
            raise Exception("No image triggers detected.")
            return
        raise Exception(err)    
    return cams

def stop_interface(cams):
    for i in range(len(cams)):
        print('stopping camera %d ...' %i)
        cams[i].stop_acquisition()
        cams[i].close_device()

def init_image():
    img = xiapi.Image()
    return img

def get_npimage(cam,img):
    cam.get_image(img, timeout = 2000)                  
    npImg = img.get_image_data_numpy()
    return npImg



