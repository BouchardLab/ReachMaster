"""The camera settings window is opened as a child of the 
ReachMaster root application. It is designed, mainly to 
specify settings required by Ximea USB3.0 cameras via the 
camera interface module and the Ximea API. It provides 
functionality for specifying the number of cameras, setting 
their intrinsic parameters (e.g., exposure, gain, triggers, 
etc.), and data output parameters (e.g., cropping, pixel 
resolution, etc.). It provides functions to capture images 
and record videos which can be used by post-hoc camera 
calibration rotuines. It also provides functions for adding, 
removing and saving pixels-of-interest which are used by 
the experiment protocols for reach detection.

Todo:
    * Automate unit tests

"""

from .. import config
from ..interfaces import experiment_interface as expint
from ..interfaces import camera_interface as camint
import tkinter as tk 
import tkinter.messagebox
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime
import numpy as np
import os 
from vidgear.gears import WriteGear
import subprocess as sp

class CameraSettings(tk.Toplevel):
    """The primary class for the camera settings window.

    Configures and provides window callback methods. Also provides 
    methods to stream/record video and capture images that can be 
    used for post hoc analyses (e.g., camera calibration). The 
    `on_quit()` method is called prior to application destruction.   

    Attributes
    ----------
    config : dict
        The current configuration settings for the application
    exp_controller : instance
        Serial interface to the experiment microcontroller which
        is used to trigger the cameras
    exp_connected : bool 
        True if the experiment interface is active
    num_cams : instance
        Tkinter StringVar that captures the user-selected number 
        of cameras
    fps : instance 
        Tkinter StringVar that captures the user-selected frames 
        per second
    exposure : instance 
        Tkinter StringVar that captures the user-selected camera 
        exposure
    gain : instance 
        Tkinter StringVar that captures the user-selected camera 
        gain.
    gpi_mode : instance 
        Tkinter StringVar that captures the camera input mode. 
        Currently, this is selected by the configuration file and 
        not the user as it is assumed all camera interfaces will 
        trigger images with the same method.
    trigger_source : instance 
        Tkinter StringVar that captures the user-selected camera 
        trigger source. Defers to the configuration file. Do not 
        change unless you know what you are doing.
    gpo_mode : instance 
        Tkinter StringVar that captures the user-selected camera 
        sync output mode.
    img_width : instance 
        Tkinter StringVar that captures the user-selected image 
        width in pixels. Must be less than your camera's maximum 
        allowable image width and divisible by it's width increment 
        value. See your camera's manual for details.
    img_height : instance 
        Tkinter StringVar that captures the user-selected image 
        height in pixels. Must be less than your camera's maximum 
        allowable image width and divisible by it's width increment 
        value. See your camera's manual for details. 
    offset_x : instance 
        Tkinter StringVar that captures the user-selected image 
        horizontal offset in pixels. Must be less than your camera's
        maximum allowable image width minus the selected img_width. 
        See your camera's manual for details.
    offset_y : instance 
        Tkinter StringVar that captures the user-selected image 
        vertical offset in pixels. Must be less than your camera's 
        maximum allowable image width minus the selected img_height. 
        See your camera's manual for details.
    downsampling : instance
        Tkinter StringVar that captures the user-selected image 
        downsampling. Can be 'XI_1x1' (full resolution) or 'XI_2x2' 
        (1/4 resolution). The latter can only be used when no cropping 
        or offsets are applied. 
    poi_threshold : instance 
        Tkinter StringVar that captures the user-selected 
        pixel-of-interest (poi) threshold (standard deviations) used 
        for reach detection.
    streaming : bool
        True if camera interface is acquiring and displaying images.
    cams_connected : bool
        True if camera interface is active.
    draw_saved : bool 
        True if saved poi's should be displayed while streaming.
    add_pois : bool 
        True if clicking on images should add new poi's during 
        streaming.
    remove_pois : bool
        True if clicking on images should remove added or saved poi's 
        during streaming.
    added_pois : list 
        A nested list of added poi coordinates for each camera. 
    saved_pois : list
        A nested list of saved poi coordinates for each camera. 
    capture : bool
        True if image should be saved to png while streaming.
    record : bool
        True while recording video.
    img_num : int 
        Counts captured images 

    """

    def __init__(self, parent):
        #create window and suppress parent
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()
        self.title("Camera Settings")
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit) 
        #initialize tk variables from config
        self.config = config.load_config('./temp/tmp_config.json')
        self.ffmpeg_command = [
        'ffmpeg', '-y', 
        '-hwaccel', 'cuvid', 
        '-f', 'rawvideo',  
        '-s', str(
            self.config['CameraSettings']['img_width'] * 
            self.config['CameraSettings']['num_cams']
            ) + 'x' + str(self.config['CameraSettings']['img_height']), 
        '-pix_fmt', 'bgr24',
        '-r', str(self.config['CameraSettings']['fps']), 
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
        self.ffmpeg_process = None
        self.exp_controller = expint.start_interface(self.config)
        self.exp_connected = True       
        self.num_cams = tk.StringVar()
        self.num_cams.set(str(self.config['CameraSettings']['num_cams']))
        self.fps = tk.StringVar()
        self.fps.set(str(self.config['CameraSettings']['fps']))
        self.exposure = tk.StringVar()
        self.exposure.set(str(self.config['CameraSettings']['exposure']))
        self.gain = tk.StringVar()
        self.gain.set(str(self.config['CameraSettings']['gain']))   
        self.gpi_mode = tk.StringVar()
        self.gpi_mode.set(self.config['CameraSettings']['gpi_mode'])
        self.trigger_source = tk.StringVar()
        self.trigger_source.set(self.config['CameraSettings']['trigger_source'])
        self.gpo_mode = tk.StringVar()
        self.gpo_mode.set(self.config['CameraSettings']['gpo_mode'])
        self.img_width = tk.StringVar()
        self.img_width.set(str(self.config['CameraSettings']['img_width']))
        self.img_height = tk.StringVar()
        self.img_height.set(str(self.config['CameraSettings']['img_height']))
        self.offset_x = tk.StringVar()
        self.offset_x.set(str(self.config['CameraSettings']['offset_x']))
        self.offset_y = tk.StringVar()
        self.offset_y.set(str(self.config['CameraSettings']['offset_y']))
        self.downsampling = tk.StringVar()
        self.downsampling.set(str(self.config['CameraSettings']['downsampling']))
        self.poi_threshold = tk.StringVar()
        self.poi_threshold.set(str(self.config['CameraSettings']['poi_threshold']))   
        #initialize housekeeping variables     
        self.streaming = False
        self.cams_connected = False
        self.draw_saved = False
        self.add_pois = False
        self.remove_pois = False
        self.added_pois = [[] for _ in range(self.config['CameraSettings']['num_cams'])]
        self.saved_pois = [[] for _ in range(self.config['CameraSettings']['num_cams'])] 
        self.capture = False
        self.record = False
        self.img_num = [1]
        #configure window      
        self._configure_window()

    def on_quit(self):
        """Called prior to destruction of the camera settings window.

        Prior to destruction, the configuration file must be updated
        to reflect the change in settings, and any active interfaces 
        must be closed. 

        """
        self.config['CameraSettings']['num_cams'] = int(self.num_cams.get())
        self.config['CameraSettings']['fps'] = int(self.fps.get())
        self.config['CameraSettings']['exposure'] = int(self.exposure.get())
        self.config['CameraSettings']['gain'] = float(self.gain.get()) 
        self.config['CameraSettings']['img_width'] = int(self.img_width.get())
        self.config['CameraSettings']['img_height'] = int(self.img_height.get())
        self.config['CameraSettings']['offset_x'] = int(self.offset_x.get())
        self.config['CameraSettings']['offset_y'] = int(self.offset_y.get())
        self.config['CameraSettings']['downsampling'] = self.downsampling.get()
        self.config['CameraSettings']['trigger_source'] = self.trigger_source.get()
        self.config['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
        self.config['CameraSettings']['poi_threshold'] = float(self.poi_threshold.get())
        config.save_tmp(self.config)
        if self.streaming:
            self._on_stream_quit()
        expint.stop_interface(self.exp_controller)
        self.destroy()

    def _configure_window(self):        
        tk.Label(
            self,
            text = "# Cameras:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 0, sticky = 'W')   
        self.num_cams_menu = tk.OptionMenu(self, self.num_cams,"1","2","3")
        self.num_cams_menu.configure(width = 12, anchor = "w")
        self.num_cams_menu.grid(row = 0, column = 1)
        tk.Label(
            self,
            text = "FPS:", 
            font = 'Arial 10 bold', 
            bg = "white", 
            width = 23,
            anchor = "e"
            ).grid(row = 1, sticky = 'W')   
        tk.Entry(self, textvariable = self.fps, width = 17).grid(row = 1, column = 1)
        tk.Label(
            self,
            text = "Exposure (usec):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 2, sticky = 'W')   
        tk.Entry(self, textvariable = self.exposure, width = 17).grid(row = 2, column = 1)
        tk.Label(
            self,
            text = "Gain:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 3, sticky = 'W')   
        tk.Entry(self, textvariable = self.gain, width = 17).grid(row = 3, column = 1)
        tk.Label(
            self,
            text = "Trigger Source:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 4, sticky = 'W')   
        self.gpi_trig_menu = tk.OptionMenu(
            self,
            self.trigger_source,
            "XI_TRG_OFF",
            "XI_TRG_EDGE_RISING",
            "XI_TRG_EDGE_FALLING",
            "XI_TRG_SOFTWARE",
            "XI_TRG_LEVEL_HIGH",
            "XI_TRG_LEVEL_LOW")
        self.gpi_trig_menu.configure(width = 12, anchor = "w")
        self.gpi_trig_menu.grid(row = 4, column = 1)
        tk.Label(
            self,
            text = "Sync Mode:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 5, sticky = 'W')   
        self.gpo_mode_menu = tk.OptionMenu(
            self,
            self.gpo_mode,
            "XI_GPO_OFF",
            "XI_GPO_ON",
            "XI_GPO_FRAME_ACTIVE",
            "XI_GPO_FRAME_ACTIVE_NEG",
            "XI_GPO_EXPOSURE_ACTIVE",
            "XI_GPO_EXPOSURE_ACTIVE_NEG",
            "XI_GPO_FRAME_TRIGGER_WAIT",
            "XI_GPO_FRAME_TRIGGER_WAIT_NEG",
            "XI_GPO_EXPOSURE_PULSE",
            "XI_GPO_EXPOSURE_PULSE_NEG",
            "XI_GPO_BUSY",
            "XI_GPO_BUSY_NEG",
            "XI_GPO_HIGH_IMPEDANCE",
            "XI_GPO_FRAME_BUFFER_OVERFLOW")
        self.gpo_mode_menu.configure(width = 12, anchor = "w")
        self.gpo_mode_menu.grid(row = 5, column = 1)
        tk.Label(
            self,
            text = "Image Width (pix):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 6, sticky = 'W')   
        tk.Entry(self, textvariable = self.img_width, width = 17).grid(row = 6, column = 1)
        tk.Label(
            self,
            text = "Image Height (pix):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 7, sticky = 'W')   
        tk.Entry(self, textvariable = self.img_height, width = 17).grid(row = 7, column = 1)
        tk.Label(
            self,
            text = "Image X Offest (pix):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 8, sticky = 'W')   
        tk.Entry(self, textvariable = self.offset_x, width = 17).grid(row = 8, column = 1)
        tk.Label(
            self,
            text = "Image Y Offset (pix):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 9, sticky = 'W')   
        tk.Entry(self, textvariable = self.offset_y, width = 17).grid(row = 9, column = 1)
        tk.Label(
            self,
            text = "Downsampling:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e").grid(row = 10, sticky = 'W') 
        self.downsampling_menu = tk.OptionMenu(
            self,
            self.downsampling,
            "XI_DWN_1x1",
            "XI_DWN_2x2")
        self.downsampling_menu.configure(width = 12, anchor = "w")
        self.downsampling_menu.grid(row = 10, column = 1)
        tk.Button(
            self,
            text = "Start Streaming",
            font = 'Arial 10 bold',
            width = 14,
            command = self.start_stream_callback
            ).grid(row = 11, column = 0, sticky = "e")
        tk.Button(
            self,
            text = "Stop Streaming",
            font = 'Arial 10 bold',
            width = 14,
            command = self.stop_stream_callback
            ).grid(row = 12, column = 0, sticky = "e")
        tk.Button(
            self,
            text = "Load POIs",
            font = 'Arial 10 bold',
            width = 14,
            command = self.load_pois_callback
            ).grid(row = 11, column = 1)
        tk.Button(
            self,
            text = "Save POIs",
            font = 'Arial 10 bold',
            width = 14,
            command = self.save_pois_callback
            ).grid(row = 12, column = 1)
        tk.Button(
            self,
            text = "Add POIs",
            font = 'Arial 10 bold',
            width = 14,
            command = self.add_pois_callback
            ).grid(row = 11, column = 2)
        tk.Button(
            self,
            text = "Remove POIs",
            font = 'Arial 10 bold',
            width = 14,
            command = self.remove_pois_callback
            ).grid(row = 12, column = 2)
        tk.Button(
            self,
            text = "Capture Image",
            font = 'Arial 10 bold',
            width = 14,
            command = self.capture_image_callback
            ).grid(row = 13, column = 0,sticky = "e")
        tk.Button(
            self,
            text = "Start Record",
            font = 'Arial 10 bold',
            width = 14,
            command = self.start_rec_callback
            ).grid(row = 13, column = 1)
        tk.Button(
            self,
            text = "Stop Record",
            font = 'Arial 10 bold',
            width = 14,
            command = self.stop_rec_callback
            ).grid(row = 13, column = 2)        
        tk.Label(
            self,
            text = "POI Threshold (stdev):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 14, sticky = 'W')   
        tk.Entry(self, textvariable = self.poi_threshold, width = 17).grid(row = 14, column = 1)
        tk.Button(
            self,
            text = "Toggle Lights", 
            font = 'Arial 10 bold',
            width = 14, 
            command = self.toggle_lights_callback
            ).grid(row = 15, column = 1)

    # Callbacks -----------------------------------------------------------------------------------

    def start_stream_callback(self):
        """Begins triggering and displaying camera images to the screen."""
        if not self.cams_connected:
            self.config['CameraSettings']['num_cams'] = int(self.num_cams.get())
            self.config['CameraSettings']['fps'] = int(self.fps.get())
            self.config['CameraSettings']['exposure'] = int(self.exposure.get())
            self.config['CameraSettings']['gain'] = float(self.gain.get())   
            self.config['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.config['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.config['CameraSettings']['img_width'] = int(self.img_width.get())
            self.config['CameraSettings']['img_height'] = int(self.img_height.get())
            self.config['CameraSettings']['offset_x'] = int(self.offset_x.get())
            self.config['CameraSettings']['offset_y'] = int(self.offset_y.get())  
            self.config['CameraSettings']['downsampling'] = self.downsampling.get()
            self.cams = camint.start_interface(self.config)
            self.cams_connected = True 
            self.img = camint.init_image()                                     
        if not self.streaming:
            self._start_stream()
        else: 
            tkinter.messagebox.showinfo("Warning", "Already streaming.") 

    def stop_stream_callback(self):
        """Stops triggering and displaying new images."""
        self.streaming = False       


    def load_pois_callback(self):
        """Loads previously saved pixels-of-interest and displays them
        over the streaming images in green."""
        if self.streaming:
            if len(self.config['CameraSettings']['saved_pois'])>0:
                self.saved_pois = self.config['CameraSettings']['saved_pois']
                self.draw_saved = True
            else:
                tkinter.messagebox.showinfo("Warning", "No saved POIs.")
        else: 
            tkinter.messagebox.showinfo("Warning", "Must be streaming to load POIs.")

    def add_pois_callback(self):
        """Allows the user to add new pixels-of-interest by clicking 
        the desired pixel using the cursor. Unsaved added pixels are
        displayed in red."""
        if self.streaming:
            self.add_pois = True
            self.remove_pois = False
        else: 
            tkinter.messagebox.showinfo("Warning", "Must be streaming to add POIs.") 

    def remove_pois_callback(self):
        """Allows the user to removed either saved or unsaved 
        pixels-of-interest. However, the user must save all added 
        pixels in order for the changes to be reflected in the
        configuration file.""" 
        if self.streaming:
            if (len(self.added_pois)+len(self.saved_pois))>0:
                self.add_pois = False
                self.remove_pois = True
            else:
                tkinter.messagebox.showinfo("Warning", "No POIs to remove.")
        else: 
            tkinter.messagebox.showinfo("Warning", "Must be streaming to remove POIs.")

    def save_pois_callback(self):
        """Saves all added and/or removed pixel-of-interest to the 
        configuration file."""
        for i in range(self.config['CameraSettings']['num_cams']):
            self.saved_pois[i] += self.added_pois[i] 
        self.config['CameraSettings']['saved_pois'] = self.saved_pois 
        self.added_pois = [[] for _ in range(self.config['CameraSettings']['num_cams'])]

    def capture_image_callback(self):
        """Allows the user to capture an image while streaming. The image
        is saved to the `calibration images` folder in the data output 
        directory."""
        if self.streaming:
            self.capture = True
        else: 
            tkinter.messagebox.showinfo("Warning", "Must be streaming to capture images.")

    def start_rec_callback(self):
        """Allows the user to record a video which is saved to the `calibration
        videos` folder of the data output directory."""
        if not self.streaming:
            # Set up camera settings, initialize cams
            self.config['CameraSettings']['num_cams'] = int(self.num_cams.get())
            self.config['CameraSettings']['fps'] = int(self.fps.get())
            self.config['CameraSettings']['exposure'] = int(self.exposure.get())
            self.config['CameraSettings']['gain'] = float(self.gain.get())   
            self.config['CameraSettings']['trigger_source'] = self.trigger_source.get()
            self.config['CameraSettings']['gpo_mode'] = self.gpo_mode.get()
            self.config['CameraSettings']['img_width'] = int(self.img_width.get())
            self.config['CameraSettings']['img_height'] = int(self.img_height.get())
            self.config['CameraSettings']['offset_x'] = int(self.offset_x.get())
            self.config['CameraSettings']['offset_y'] = int(self.offset_y.get())          
            self.cams = camint.start_interface(self.config)
            self.cams_connected = True
            self.img = camint.init_image()
            calibration_path = self.config['ReachMaster']['data_dir'] + "/calibration_videos/"
            if not os.path.isdir(calibration_path):
                os.makedirs(calibration_path)
            vid_fn = calibration_path + str(datetime.datetime.now()) + '.mp4'
            # Initialize vidgear wheels
            output_params = self.ffmpeg_command
            #output_params_ = {"-vcodec": "libx264", "-crf": 10, "-preset": "fast", "-tune": "zerolatency"}
            self.vidgear_writer_cal = WriteGear(output_filename=vid_fn, compression_mode=True, logging=True, **output_params)
            # ffmpeg unit commands, depreciated
            #ffmpeg_command = self.ffmpeg_command
            #ffmpeg_command.append(vid_fn)
            #ffmpeg_command[ffmpeg_command.index('-s') + 1] = str(
            #    self.config['CameraSettings']['img_width'] *
            #    self.config['CameraSettings']['num_cams']
            #    ) + 'x' + str(self.config['CameraSettings']['img_height'])
            #ffmpeg_command[ffmpeg_command.index('-r') + 1] = str(
            #    self.config['CameraSettings']['fps']
            #    )
            # ffmpeg pipes (depreciated)
            #self.ffmpeg_process = sp.Popen(
            #ffmpeg_command,
            #stdin=sp.PIPE,
            #stdout=sp.DEVNULL,
            #stderr=sp.DEVNULL,
            #bufsize=-1
            #)
            self.delay = int(np.round(1.0/float(self.config['CameraSettings']['fps'])*1000.0))
            self.record = True
            self._rec()
        else: 
            tkinter.messagebox.showinfo("Warning", "Shouldn't record while streaming. Bad framerates!")

    def stop_rec_callback(self):
        """Stops a video recording."""
        self.record = False
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
        camint.stop_interface(self.cams)
        self.cams_connected = False

    def toggle_lights_callback(self):
        """Allows the user to toggle the neopixel lights while streaming or
        recording a video."""
        if self.exp_connected:
            expint.toggle_lights(self.exp_controller)
        else:
            tkinter.messagebox.showinfo("Warning", "Experiment controller not connected.")

    # private methods -------------------------------------------------------------------------------

    def _on_stream_quit(self):
        self.streaming = False          
        self.poi_active = False  
        self.draw_saved = False    
        for i in range(self.config['CameraSettings']['num_cams']):
            self.cam_windows[i].destroy()
        camint.stop_interface(self.cams)
        self.cams_connected = False

    def _start_stream(self):
        self.cam_windows = [0 for _ in range(self.config['CameraSettings']['num_cams'])]
        for i in range(self.config['CameraSettings']['num_cams']):
            self.cam_windows[i] = tk.Toplevel(self)
            self.cam_windows[i].title("Camera"+str(i))
            self.cam_windows[i].protocol("WM_DELETE_WINDOW", self._on_stream_quit)
            self.cam_windows[i].canvas = tk.Canvas(self.cam_windows[i], 
                width = self.config['CameraSettings']['img_width'], 
                height = self.config['CameraSettings']['img_height'])
            self.cam_windows[i].canvas.grid(row=0,column= 0)            
        self.delay = int(np.round(1.0/float(self.config['CameraSettings']['fps'])*1000.0))
        self.photo_img = [0 for _ in range(self.config['CameraSettings']['num_cams'])]
        self.streaming = True
        try:
            self._refresh()
        except Exception as err:
            tkinter.messagebox.showinfo("Warning", err)
            self._on_stream_quit()

    def _refresh(self):
        if self.streaming:
            expint.trigger_image(self.exp_controller)
            now = str(int(round(time.time()*1000)))            
            for i in range(self.config['CameraSettings']['num_cams']):
                #display image
                npimg = camint.get_npimage(self.cams[i],self.img)
                npimg = cv2.cvtColor(npimg,cv2.COLOR_BAYER_BG2BGR)
                self.photo_img[i] = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(npimg))
                self.cam_windows[i].canvas.create_image(
                    0, 0, image = self.photo_img[i], anchor = tk.NW
                    )                
                #draw saved pixels (green)
                if self.draw_saved:
                    for poi in self.saved_pois[i]:                         
                        self.cam_windows[i].canvas.create_line(
                            poi[0], poi[1], poi[0] + 1, poi[1], width = 1, fill = 'green'
                            )
                #draw currently addded pixels (red)
                for poi in self.added_pois[i]:                        
                    self.cam_windows[i].canvas.create_line(
                        poi[0], poi[1], poi[0] + 1, poi[1], width = 1, fill = 'red'
                        )
                #draw cursor for adding/removing pois
                if self.add_pois or self.remove_pois:
                    self._draw_cursor(i)
                    self.cam_windows[i].bind(
                        '<Button-1>', lambda event, camid = i:self._draw_poi(event,camid)
                        )
                #prepare frame for possible capture
                if i == 0:
                    frame = npimg
                else:
                    frame = np.hstack((frame,npimg))
            if self.capture:
                self.calibration_path = self.config['ReachMaster']['data_dir'] + "/calibration_images/"
                if not os.path.isdir(self.calibration_path):
                    os.makedirs(self.calibration_path)
                fn = "image" + str(self.img_num[0])
                cv2.imwrite('%s/%s.png' % (self.calibration_path, fn), frame)
                self.capture = False
                self.img_num[0] += 1
            self.after(self.delay,self._refresh)

    def _draw_cursor(self,i):
        self.cam_windows[i].bind('<Motion>', self.cam_windows[i].config(cursor = "cross"))        

    def _draw_poi(self, event, camid):
        if self.add_pois:
            self.added_pois[camid].append([event.x,event.y])  
        elif self.remove_pois:
            if len(self.saved_pois[camid])>0:
                tmp = []
                for poi in self.saved_pois[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.saved_pois[camid] = tmp
            if len(self.added_pois[camid])>0:
                tmp = []
                for poi in self.added_pois[camid]:
                    if np.sqrt((event.x-poi[0])**2+(event.y-poi[1])**2)>5:
                        tmp.append(poi)
                self.added_pois[camid] = tmp

    def _rec(self):
        if self.record:
            frame = 0
            expint.trigger_image(self.exp_controller)
            try:          
                for i in range(self.config['CameraSettings']['num_cams']):
                    npimg = camint.get_npimage(self.cams[i],self.img)
                    npimg = cv2.cvtColor(npimg, cv2.COLOR_BAYER_BG2BGR)
                    if i == 0:
                        frame = npimg
                    else:
                        frame = np.hstack((frame, npimg))  
            except Exception as err:
                tkinter.messagebox.showinfo("Warning", err)
                self.stop_rec_callback() 
                return
            # Use WriteGear to write video processed
            self.vidgear_writer_cal.write(frame)
            # reset frame back to real value
            self.after(self.delay, self._rec)