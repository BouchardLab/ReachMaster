"""The protocol window is opened as a child of the 
ReachMaster root application whenever a protocol is run. It 
provides basic functionality for interacting with the rig 
while an experiment is running (e.g., for manual reward 
delivery, toggling lights, etc.).

Todo:
    * Keep adding new protocol types
    * Automate unit tests

"""

from . import config
from .interfaces import camera_interface as camint
from .interfaces import robot_interface as robint
from .interfaces import experiment_interface as expint
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import cv2
from ximea import xiapi
from vidgear.gears import WriteGear
from time import time, sleep
import datetime
import os
import numpy as np
import multiprocessing as mp
import threading
import pdb


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
        print(('Setting camera %d ...' % i))
        _set_camera(cams[i], config)


def _open_cameras(config):
    cams = []
    for i in range(config['CameraSettings']['num_cams']):
        print(('loading camera %s ...' % (i)))
        cams.append(xiapi.Camera(dev_id=i))
        cams[i].open_device()
    return cams


def _start_cameras(cams):
    for i in range(len(cams)):
        print(('Starting camera %d ...' % i))
        cams[i].start_acquisition()


# public functions -----------------------------------------------------------------

def stop_interface(cams):
    """Stop image acquisition and close all cameras.

    Parameters
    ----------
    cams : list
        A list of ximea api Camera objects.

    """
    for i in range(len(cams)):
        print(('stopping camera %d ...' % i))
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
    cam.get_image(img, timeout=2000)
    npimg = img.get_image_data_numpy()
    return npimg


def list_protocols():
    """Generate a list of the available protocol types. Currently 
    limited to 'TRIALS' and 'CONTINUOUS'."""
    protocol_list = list(['TRIALS', 'CONTINUOUS'])
    return protocol_list


class Protocols(tk.Toplevel):
    """The primary class for the protocol window.

    Configures and provides window callback methods. Also provides 
    special `run` methods that implement each of the protocol types.
    The run methods work in concert with the interface modules and
    microntroller scripts to implement the event logic required by
    a specific experiment. If an experiment requires a new protocol
    type that is not currently available, implementing a new run 
    method will often be a good place to start.      

    Attributes
    ----------
    ready : bool
        True when it is okay to call the run method.
    config : dict
        The current configuration settings for the application.
    exp_connected : bool
        True when experiment controller interface is activated.
    rob_connected : bool
        True when robot controller interface is activated.
    cams_connected : bool
        True when camera interface is activated.
    lights_on : bool
        True if neopixel lights are on.
    baseline_acquired : bool
        True is camera processes have collected all baseline images.
    reach_detected : bool
        True if the minimum pixel of interest deviaton across all
        cameras in greater than the selected threshold.
    reach_init : int
        Time at which most recent reach was detected.
    control_message : char
        Character message that is sent to the experiment controller
        according to the communication protocol.
    exp_response : char
        Character received in response to the control_message from 
        the experiment controller interpreted according to the 
        communication protocol. 

    """

    def __init__(self, parent):
        self.ready = False
        # create window
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        self.grab_set()
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.config = config.load_config('./temp/tmp_config.json')
        self.title("Protocol: " + self.config['Protocol']['type'])
        # initialize protocol variables
        self.exp_connected = False
        self.rob_connected = False
        self.cams_connected = False
        self.lights_on = False
        self.baseline_acquired = False
        self.reach_detected = False
        self.lick_window = False
        self.reach_init = 0
        # New config for sliding camera's into protocols
        self.ffmpeg_object = {
            '-f': 'rawvideo',
            '-s': str(
                self.config['CameraSettings']['img_width'] *
                self.config['CameraSettings']['num_cams']
            ) + 'x' + str(self.config['CameraSettings']['img_height']),
            '-pix_fmt': 'bgr24',
            '-r': str(self.config['CameraSettings']['fps']),
            '-i': '-',
            '-b:v': '2M',
            '-maxrate': '2M',
            '-bufsize': '1M',
            '-c:v': 'libx264',
            '-preset': 'superfast',
            '-pix_fmt': 'yuv420p'
        }
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
        self.camera_processes = []
        self.cam_trigger_pipes = []
        self.poi_deviation_pipes = []
        self.trial_ended_pipes = []
        self.cams_started = False

        # check config for errors
        if len(self.config['CameraSettings']['saved_pois']) == 0:
            tkinter.messagebox.showinfo("Warning", "No saved POIs")
            self.on_quit()
            return
        # start interfaces, load settings and acquire baseline for reach detection
        print('starting speaker...')
        self.initialize_speaker()
        self.load_auditory_stimuli(self.config)
        print("starting interfaces...")
        self.exp_controller = expint.start_interface(self.config)
        sleep(1)
        self.exp_connected = True
        print("loading experiment settings...")
        expint.set_exp_controller(self.exp_controller, self.config)

        # self.cams = camint.CameraInterface(self.config)
        self.cams_connected = True
        self.start_camera_interface()  # start the camera interface
        # self.cam_thread = threading.Thread(target=self.cam_init())
        # self.cam_thread.start()
        sleep(10)  # give the cameras time to start
        #self._acquire_baseline()

        self._init_data_output()
        self._configure_window()
        self.control_message = 'b'  # Send experimental micro-controller message indicating initiation
        while not self.all_triggerable():  # If camera's not triggerable..
            pass
        # Start up robot processes.
        sleep(1)
        self.rob_controller = robint.start_interface(self.config)
        sleep(2)
        self.rob_connected = True
        print("loading robot settings...")
        self.config = robint.set_rob_controller(self.rob_controller, self.config)
        sleep(2)

        self.exp_response = expint.start_experiment(self.exp_controller)  # start experiment
        self.ready = True
        self.run_auditory_stimuli()  # runs sound at beginning of experiment!

    def on_quit(self):
        """Called prior to destruction protocol window.

        Prior to destruction, all interfaces must be stopped.

        """
        self.ready = False
        if self.exp_connected:
            expint.stop_interface(self.exp_controller)
        if self.rob_connected:
            robint.stop_interface(self.rob_controller)
        if self.cams_connected:
            self.stop_interface()
        self.destroy()

    def _init_data_output(self):
        self.controller_data_path = self.config['ReachMaster']['data_dir'] + "/controller_data/"
        self.video_data_path = self.config['ReachMaster']['data_dir'] + "/videos/"
        if not os.path.isdir(self.controller_data_path):
            os.makedirs(self.controller_data_path)
        if not os.path.isdir(self.video_data_path):
            os.makedirs(self.video_data_path)
        controller_data_file = self.controller_data_path + str(datetime.datetime.now())
        self.outputfile = open(controller_data_file, "w+")
        header = "time trial exp_response rob_moving image_triggered in_reward_window z_poi"
        self.outputfile.write(header + "\n")

    def _configure_window(self):
        tk.Button(
            self,
            text="Move Robot",
            font='Arial 10 bold',
            width=14,
            command=self.move_robot_callback
        ).grid(row=0, sticky='W')
        tk.Button(
            self,
            text="Toggle LED",
            font='Arial 10 bold',
            width=14,
            command=self.toggle_led_callback
        ).grid(row=1, sticky='W')
        tk.Button(
            self,
            text="Toggle Lights",
            font='Arial 10 bold',
            width=14,
            command=self.toggle_lights_callback
        ).grid(row=2, sticky='W')
        tk.Button(
            self,
            text="Deliver Water",
            font='Arial 10 bold',
            width=14,
            command=self.deliver_water_callback
        ).grid(row=3, sticky='W')

    # Callbacks ------------------------------------------------------------------

    def move_robot_callback(self):
        """Commands the robot to move to the next position in
        its currently loaded command sequence."""
        expint.move_robot(self.exp_controller)
        if self.config['Protocol']['type'] == 'TRIALS':
            self.lights_on = 0

    def toggle_led_callback(self):
        """Toggles the LED located on the robot handle."""
        expint.toggle_led(self.exp_controller)

    def toggle_lights_callback(self):
        """Toggles the neopixel lights."""
        expint.toggle_lights(self.exp_controller)
        self.lights_on = not self.lights_on

    def deliver_water_callback(self):
        """Opens the reward solenoid for the reward duration set 
        in experiment settings."""
        expint.deliver_water(self.exp_controller)

    # Auditory stimuli ---------------------------------------------------------------
    def initialize_speaker(self):
        return

    def load_auditory_stimuli(self, config):
        # audio_file = config['Protocol']['audio_file']
        # load auditory file into speaker
        return

    def run_auditory_stimuli(self):
        # command to check if speaker is online
        # command to initiate auditory stimuli (single use in experiment)
        return

    # Protocol types ---------------------------------------------------------------

    def run_continuous(self):
        """Operations performed for a single iteration of protocol type CONTINOUS.

        If an image is triggered by the experiment controller, reach detection is
        performed on the images, the images are concatenated, and encoded to mp4
        continuously, even during intertrial intervals, in real-time. Consequently,
        for this protocol type, the lights remain on during the intertrial interval. 
        Messages are sent to the experiment controller to signal important events 
        such as reach detections, beginnings/ends of trials, and to initiate robot 
        movement. Responses are read from the experiment controller and saved to 
        the controller data output file.   

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement. 
            * Absorb communication codes into experiment interface module and document

        """
        now = str(int(round(time() * 1000)))  # Norm time
        if self.exp_response[3] == '1':  # If a trial is taking place
            self.triggered()
            self.lights_on = 1
            self.poi_deviation = self.get_poi_deviation()  # get deviation from captured frame
            while not self.all_triggerable():  # are camera pipes triggerable?
                pass
        else:  # If trial is re-setting, robot moving etc
            self.lights_on = 0  # turn off lights
            self.poi_deviation = 0  # reset deviation
        expint.write_message(self.exp_controller, self.control_message)
        self.exp_response = expint.read_response(self.exp_controller)
        self.outputfile.write(
            now + " " + self.exp_response[0:-1:1] + " " + str(self.poi_deviation) + "\n"
        )
        self.exp_response = self.exp_response.split()
        if (
                self.exp_response[1] == 's' and
                self.exp_response[2] == '0' and
                self.poi_deviation > self.config['CameraSettings']['poi_threshold']
        ):
            self.reach_init = now
            self.reach_detected = True
            self.control_message = 'r'
        elif self.exp_response[1] == 'e':
            self.poi_deviation = 0
            self.control_message = 's'
            self.reach_detected = False
            print((self.exp_response[0]))
        elif (
                self.reach_detected and
                (int(now) - int(self.reach_init)) >
                self.config['ExperimentSettings']['reach_timeout'] and
                self.exp_response[4] == '0'
        ):
            self.move_robot_callback()
            self.reach_detected = False

    def run_trials(self):
        """Operations performed for a single iteration of protocol type TRIALS.

        If an image is triggered by the experiment controller, reach detection is
        performed on the images. For this protocol type, the lights are turned off 
        during the intertrial interval and a separate video is encoded for each 
        trial. Messages are sent to the experiment controller to signal important 
        events such as reach detections, beginnings/ends of trials, and to initiate 
        robot movement. Responses are read from the experiment controller and saved 
        to the controller data output file.    

        Todo:
            * Functionalize code chunks so logic is clearer and custom protocol types are easier to implement.
            * Absorb communication codes into experiment interface module and document

        """
        now = str(int(round(time() * 1000)))
        if self.exp_response[3] == '1':
            self.triggered()
            self.lights_on = 1
            self.poi_deviation = self.get_poi_deviation()
            while not self.all_triggerable():
                pass
        else:
            self.lights_on = 0
            self.poi_deviation = 0
        expint.write_message(self.exp_controller, self.control_message)
        self.exp_response = expint.read_response(self.exp_controller)
        self.outputfile.write(
            now + " " + self.exp_response[0:-1:1] + " " + str(self.poi_deviation) + "\n"
        )
        self.exp_response = self.exp_response.split()
        if (
                self.exp_response[1] == 's' and
                self.exp_response[2] == '0' and
                self.poi_deviation > self.config['CameraSettings']['poi_threshold']
        ):
            self.reach_init = now
            self.reach_detected = True
            self.reach_detected = True
            self.control_message = 'r'
        elif self.exp_response[1] == 'e':
            self.trial_ended()
            self.poi_deviation = 0
            self.control_message = 's'
            self.reach_detected = False
            print((self.exp_response[0]))
        elif (
                self.reach_detected and
                (int(now) - int(self.reach_init)) >
                self.config['ExperimentSettings']['reach_timeout'] and
                self.exp_response[4] == '0'
        ):
            self.move_robot_callback()
            self.reach_detected = False

    def trial_ended(self):
        """Alert cameras processes that a trial has ended."""
        for pipe in self.trial_ended_pipes:
            pipe.send(1)

    def run(self):
        """Execute a single iteration of the selected protocol type.
        
        The currently supported protocol types are TRIALS and 
        CONTINUOUS. In order to add a custom type, developers must 
        write a `run_newtype()` method and add it to the call 
        search here. The run methods for each protocol type are 
        where all the vital real-time operations are executed via 
        communication with the interface modules.

        """
        if self.config['Protocol']['type'] == 'CONTINUOUS':
            self.run_continuous()
        elif self.config['Protocol']['type'] == 'TRIALS':
            self.run_trials()
        else:
            print("Invalid protocol!")
            self.on_quit()

    # Camera processes and functions -----------------------------------------

    def start_camera_interface(self):
        """Sets up the camera process and pipe system for each
        camera.

        """

        print('starting camera processes... ')
        for cam_id in range(self.config['CameraSettings']['num_cams']):
            trigger_parent, trigger_child = mp.Pipe()
            self.cam_trigger_pipes.append(trigger_parent)
            poi_parent, poi_child = mp.Pipe()
            self.poi_deviation_pipes.append(poi_parent)
            trial_parent, trial_child = mp.Pipe()
            self.trial_ended_pipes.append(trial_parent)
            self.camera_processes.append(
                mp.Process(
                    target=self._camera_processes,
                    args=(
                        cam_id,
                        trigger_child,
                        poi_child,
                        trial_child
                    )
                )
            )
        self.cams_started = True
        for process in self.camera_processes:
            process.start()
        sleep(5)  # give cameras time to setup

    def _camera_processes(
            self,
            cam_id,
            trigger_pipe,
            poi_deviation_pipe,
            trial_ended_pipe
    ):
        """ Process containing recording, reach-detection techniques. """
        # Create Id's for video functions
        vid_fn = (
                self.config['ReachMaster']['data_dir'] + '/videos/trial' + '_cam' + str(cam_id) + '.mp4'
        )
        sleep(2 * cam_id)  # prevents simultaneous calls to the ximea api
        print('finding camera: ', cam_id)
        cam = xiapi.Camera(dev_id=cam_id)
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
                    decimals=0
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
        vidgear_writer_cal = WriteGear(output_filename=vid_fn)
        # self.ffmpeg_command.append(vid_fn)
        # ffmpeg_process = sp.Popen(
        #    self.ffmpeg_command,
        #    stdin=sp.PIPE,
        #    stdout=sp.DEVNULL,
        #    stderr=sp.DEVNULL,
        #    bufsize=-1
        #    )
        while self.cams_started:
            if trigger_pipe.poll():
                trigger_pipe.recv()
                try:
                    cam.get_image(img, timeout=2000)
                    trigger_pipe.send('c')  # MP, triggers camera pipe to clear image on arduino.
                    npimg = img.get_image_data_numpy()  # Numpy matrix for image.
                    frame = cv2.cvtColor(npimg, cv2.COLOR_BAYER_BG2BGR)  # Takes frame from BG to BGR color encoding.
                    vidgear_writer_cal.write(frame)  #
                    # ffmpeg_process.stdin.write(frame)
                    dev = self._estimate_poi_deviation(cam_id, npimg, poi_means, poi_std)
                    poi_deviation_pipe.send(dev)
                except Exception as err:
                    print("cam_id: " + str(err))
                    pass
            elif (
                    self.config['Protocol']['type'] == 'TRIALS' and
                    trial_ended_pipe.poll()
            ):
                trial_ended_pipe.recv()
                vidgear_writer_cal.close()
                # ffmpeg_process.stdin.close()
                # ffmpeg_process.wait()
                # ffmpeg_process = None
                trial_num += 1
        cam.stop_acquisition()
        cam.close_device()

    def stop_interface(self):
        """Tells the camera processes to shut themselves down, waits
        for them to exit, then cleans all the pipes.
        """
        self.cams_started = False
        for proc in self.camera_processes:
            proc.join()
        self.cam_trigger_pipes = []
        self.poi_deviation_pipes = []
        self.trial_ended_pipes = []

    def triggered(self):
        """Alert camera processes that cameras have been triggered"""
        for pipe in self.cam_trigger_pipes:
            pipe.send(1)

    def all_triggerable(self):
        """Check if camera processes are ready for cameras to be
        triggered.

        Returns
        -------
        triggerable : bool
            True if cameras ready.
        """
        if all([pipe.poll() for pipe in self.cam_trigger_pipes]):
            for pipe in self.cam_trigger_pipes:
                pipe.recv()
            triggerable = True
        else:
            triggerable = False
        return triggerable

    # Tools to estimate, report large light deviations in pre-set POI regions. This is how "reaches" are detected in
    # a trial.

    def _acquire_baseline(self, cam, cam_id, img, num_imgs, trigger_pipe):
        poi_indices = self.config['CameraSettings']['saved_pois'][cam_id]
        num_pois = len(poi_indices)
        baseline_pois = np.zeros(shape=(num_pois, num_imgs))
        trigger_pipe.send(1)
        # acquire baseline images
        for i in range(num_imgs):
            while not trigger_pipe.poll():
                pass
            trigger_pipe.recv()
            try:
                cam.get_image(img, timeout=2000)
                trigger_pipe.send('c')
                npimg = img.get_image_data_numpy()
                for j in range(num_pois):
                    baseline_pois[j, i] = npimg[
                        poi_indices[j][1],
                        poi_indices[j][0]
                    ]
            except Exception as err:
                print("error: " + str(cam_id))
                print(err)
        # compute summary stats
        poi_means = np.mean(baseline_pois, axis=1)
        poi_std = np.std(
            np.sum(
                np.square(
                    baseline_pois - poi_means.reshape(num_pois, 1)
                ),
                axis=0
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

    def get_poi_deviation(self):
        """Check the smallest deviation from baseline of pixels of
        interest across all cameras.

        Returns
        -------
        dev : int
            The minimum deviation across all cameras.
        """
        if all([pipe.poll() for pipe in self.poi_deviation_pipes]):
            deviations = [pipe.recv() for pipe in self.poi_deviation_pipes]
            dev = min(deviations)
        else:
            dev = 0
        return dev
