"""The experiment settings window is opened as a child of the 
ReachMaster root application. It allows the user to set any
experiment parameters required by the experiment 
microcontroller (e.g., number/size of rewards, trial delays, 
etc.).
"""

from .. import config
import tkinter as tk 
import tkinter.messagebox

class ExperimentSettings(tk.Toplevel):
    """The primary class for the experiment settings window.  

    Attributes
    ----------
    config : dict
        The current configuration settings for the application
    lights_on_dur : instance
        Tkinter StringVar that captures the user-selected duration
        (ms) lights should remain on during a trial when `protocol 
        type` is 'TRIALS'. If a reach is not detected in under this 
        duration, the lights will turn off until the next trial is
        intitiated.
    lights_off_dur : instance
        Tkinter StringVar that captures the user-selected duration
        (ms) lights should remain off between trials when `prtocol 
        type` is 'TRIALS'.
    reward_win_dur : instance
        Tkinter StringVar that captures the user-selected duration
        (ms) for which rewards rewards are delivered once the 
        handle is successfully pulled into the reward zone.
    max_rewards : instance
        Tkinter StringVar that captures the user-selected maximum 
        number of rewards (int) delivered for a successful reach.
    solenoid_open_dur : instance
        Tkinter StringVar that captures the user-selected solenoid 
        open duration (ms) which sets the individual reward size.
        Solenoid calibration must be performed to get the mapping 
        from duration (ms) to volulme (mL).
    solenoid_bounce_dur : instance
        Tkinter StringVar that captures the user-selected minimum
        duration (ms) the solenoid should remain closed between
        reward deliveries. Note the redundancy with `reward_win_dur`
        and `max_rewards`.
    flush_dur : instance
        Tkinter StringVar that captures the user-selected duration
        (ms) the solenoid should remain open while flushing the 
        lines.
    reach_delay : instance
        Tkinter StringVar that captures the user-selected minimum
        duration animals are required to wait before attempting a
        reach after a trial is initiated when `protocol type` is 
        'TRIALS'. Once the lights turn on, if a reach is attempted 
        before the end of the delay, the trial is haulted and the 
        lights are turned off. This enforces a timeout period to 
        discourage animals from performing reaches between trials.    

    """

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent) 
        self.grab_set()
        self.title("Experiment Settings") 
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_quit) 
        self.config = config.load_config('./temp/tmp_config.json')
        self.lights_on_dur = tk.StringVar()
        self.lights_on_dur.set(str(self.config['ExperimentSettings']['lights_on_dur']))
        self.lights_off_dur = tk.StringVar()
        self.lights_off_dur.set(str(self.config['ExperimentSettings']['lights_off_dur']))
        self.reward_win_dur = tk.StringVar()
        self.reward_win_dur.set(str(self.config['ExperimentSettings']['reward_win_dur']))
        self.max_rewards = tk.StringVar()
        self.max_rewards.set(str(self.config['ExperimentSettings']['max_rewards']))
        self.solenoid_open_dur = tk.StringVar()
        self.solenoid_open_dur.set(str(self.config['ExperimentSettings']['solenoid_open_dur']))
        self.solenoid_bounce_dur = tk.StringVar()
        self.solenoid_bounce_dur.set(str(self.config['ExperimentSettings']['solenoid_bounce_dur']))
        self.flush_dur = tk.StringVar()
        self.flush_dur.set(str(self.config['ExperimentSettings']['flush_dur']))
        self.reach_delay = tk.StringVar()
        self.reach_delay.set(str(self.config['ExperimentSettings']['reach_delay']))
        #self.audio_file = tk.StringVar()
        #self.audio_file.set(str(self.config['ExperimentSettings']['audio_file']))
        self._configure_window()

    def on_quit(self):
        """Called prior to destruction of the experiment settings window.

        Prior to destruction, the configuration file must be updated
        to reflect the change in settings. 

        """
        self.config['ExperimentSettings']['lights_on_dur'] = int(self.lights_on_dur.get())
        self.config['ExperimentSettings']['lights_off_dur'] = int(self.lights_off_dur.get())
        self.config['ExperimentSettings']['reward_win_dur'] = int(self.reward_win_dur.get())
        self.config['ExperimentSettings']['max_rewards'] = int(self.max_rewards.get()) 
        self.config['ExperimentSettings']['solenoid_open_dur'] = int(self.solenoid_open_dur.get())
        self.config['ExperimentSettings']['solenoid_bounce_dur'] = int(self.solenoid_bounce_dur.get())
        self.config['ExperimentSettings']['flush_dur'] = int(self.flush_dur.get())
        self.config['ExperimentSettings']['reach_delay'] = int(self.reach_delay.get())
        config.save_tmp(self.config)
        self.destroy()

    def _configure_window(self):
        tk.Label(
            self,
            text = "Lights On (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row=1, column=0)   
        tk.Entry(self, textvariable = self.lights_on_dur, width = 17).grid(row = 1, column = 1)
        tk.Label(
            self,
            text = "Lights Off (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 2, column = 0)   
        tk.Entry(self, textvariable = self.lights_off_dur, width = 17).grid(row = 2, column = 1)
        tk.Label(
            self,
            text = "Reward Window (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 3, column = 0)   
        tk.Entry(self, textvariable = self.reward_win_dur, width = 17).grid(row = 3, column = 1)
        tk.Label(
            self,
            text = "# Rewards/Trial:", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 4, column = 0)   
        tk.Entry(self, textvariable = self.max_rewards, width = 17).grid(row = 4, column = 1)
        tk.Label(
            self,
            text = "Solenoid Open (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 5, column = 0)   
        tk.Entry(self, textvariable = self.solenoid_open_dur, width = 17).grid(row = 5, column = 1)
        tk.Label(
            self,
            text = "Solenoid Bounce (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 6, column = 0)   
        tk.Entry(
            self,
            textvariable = self.solenoid_bounce_dur,
            width = 17
            ).grid(row = 6, column = 1)
        tk.Label(
            self,
            text = "Flush (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 7, column = 0)   
        tk.Entry(self, textvariable = self.flush_dur, width = 17).grid(row = 7, column = 1)
        tk.Label(
            self,
            text = "Reach Delay (ms):", 
            font = 'Arial 10 bold', 
            bg = "white",
            width = 23,
            anchor = "e"
            ).grid(row = 8, column = 0)   
        tk.Entry(self, textvariable = self.flush_dur, width = 17).grid(row=8, column = 1)
        tk.Entry(self, textvariable=self.audio_file, width=17).grid(row=9, column=1)
        tk.Label(
            self,
            text="Auditory Stimuli",
            font='Arial 10 bold',
            bg="white",
            width=23,
            anchor="e"
        ).grid(row=8, column=0)
        tk.Entry(self, textvariable=self.audio_file, width=17).grid(row=9, column=1)