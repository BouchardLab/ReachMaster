import pynwb
import numpy as np

def init_nwb_file(file_name, source_script, experimenter):

    #session description
    #identifier
    #session_start_time
    #file_create_date
    #timestamps_reference_time
    #experimenter
    #session_id
    #source_script_file_name
    #stimulus_notes
    #devices
    #subject

    nwbfile = pynwb.NWBFile(session_description='demonstrate NWBFile basics',  # required
                      identifier='NWB123',  # required
                      session_start_time=start_time,  # required
                      file_create_date=create_date)  # optional
    #general time series
    test_ts = pynwb.TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    nwbfile.add_acquisition(test_ts)
    reuse_ts = pynwb.TimeSeries('reusing_timeseries', newdata, 'SIunit', timestamps=test_ts)
    #electrophysiology data
    device = nwbfile.create_device(name='trodes_rig123')
    electrode_group = nwbfile.create_electrode_group(electrode_name,
                                                     description=description,
                                                     location=location,
                                                     device=device)
    for idx in [1, 2, 3, 4]:
        nwbfile.add_electrode(id=idx,
                              x=1.0, y=2.0, z=3.0,
                              imp=float(-idx),
                              location='CA1', filtering='none',
                              group=electrode_group)
    electrode_table_region = nwbfile.create_electrode_table_region([0, 2], 'the first and third electrodes')
    ephys_ts = pynwb.ecephys.ElectricalSeries('test_ephys_data',
                                ephys_data,
                                electrode_table_region,
                                timestamps=ephys_timestamps,
                                # Alternatively, could specify starting_time and rate as follows
                                # starting_time=ephys_timestamps[0],
                                # rate=rate,
                                resolution=0.001,
                                comments="This data was randomly generated with numpy, using 1234 as the seed",
                                description="Random numbers generated with numpy.random.rand")
    nwbfile.add_acquisition(ephys_ts)
    return nwbfile

def make_trial_masks(controller_data, experiment_data):
    """

    Parameters
    ----------
    controller_data : list
        list of data from the experimental microcontroller
    experiment_data : dict
        dict of trodes experimental data over sessions
    Returns
    -------
    mask_array : array
        mask containing trial numbers over trodes time
    """
    time = experiment_data['time']['time']
    trials = np.asarray(controller_data['trial'])
    matched_times = match_times(controller_data, experiment_data)
    trial_transitions = np.where(np.roll(trials, 1) != trials)[0]
    num_trials = np.amax(trials)
    mask_array = np.zeros(len(time))
    trial_transitions = matched_times[trial_transitions]
    trial_index = np.searchsorted(time, trial_transitions)
    for i in range(0, num_trials - 1):
        mask_array[trial_index[i]:trial_index[i + 1]] = i + 1
    return mask_array


def get_successful_trials(controller_data, matched_time, experiment_data):
    """

    Parameters
    ----------
    controller_data : list
        list of data from the microcontroller
    matched_time : array
        controller event times converted to trodes time
    experiment_data : dict
        trodes experimental data for each session

    Returns
    -------
    success_rate : list
        list of successful trials
    """
    success_rate = []
    lick_data = experiment_data['DIO']['IR_beam']
    reach_indices = get_reach_indices(controller_data)
    reach_times = get_reach_times(matched_time, reach_indices)
    reach_start = reach_times['start']
    reach_stop = reach_times['stop']
    trial_num = 0
    for xi in range(len(reach_start)):
        i = reach_start[xi]  # these are start and stop times on trodes time
        j = reach_stop[xi]
        if True in np.vectorize(lambda x: i <= x <= j)(lick_data):
            success_rate.append(xi)
    return success_rate

def make_split_trial_videos(video_path, reach_times):
    """
    Function to split an experimental trial video into discrete videos of each trial inside the session
    Parameters
    ----------
    video_path : str
       full path to video of experiment session
    reach_times : dict
        dict of arrays containing reach start and stop times


    Returns
    -------

    """

    start_times = reach_times['start']  # normalize these times
    stop_times = reach_times['stop']
    try:
        if not os.path.exists(video_path + 'trials'):
            os.makedirs(video_path + 'trials')
    except OSError:
        print('Error: Creating directory of data')
    current_clip = 0
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(str(current_clip) + 'reach.avi', -1, 1, (frameWidth, frameHeight))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    for xi in range(len(start_times)):
        start = start_times[xi] + 1000
        stop = stop_times[xi] + 1000
        for i in range(start, stop):
            video.write(buf[i])
    cv2.destroyAllWindows()
    video.release()
    return