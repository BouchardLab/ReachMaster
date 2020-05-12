import pynwb
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from ..trodes_data import experiment_data_parser as trodes_edp
from ..controller_data import experiment_data_parser as controller_edp
# from ..video_data import experiment_data_parser as video_edp
from ..config_data import config_parser


def init_nwb_file(file_name, source_script, experimenter, session_start_time):
    subject_id, date, session_id = file_name.split("_")
    nwb_file = pynwb.NWBFile(
        session_description='reaching without borders',  # required
        identifier=file_name,  # required
        session_start_time=session_start_time,  # required
        experimenter=experimenter,
        session_id=session_id,
        file_create_date=datetime.now(tzlocal()),
        source_script=source_script
    )
    subject = pynwb.file.Subject(subject_id=subject_id)  # consider adding more metadata
    nwb_file.subject = subject
    return nwb_file


def save_nwb_file(nwb_file, save_dir):
    filename = save_dir + '\\' + nwb_file.identifier + '.nwb'
    io = pynwb.NWBHDF5IO(filename, mode='w')
    io.write(nwb_file)
    io.close()


def add_trodes_analog(nwb_file, trodes_data):
    analog_keys = [*trodes_data['analog']]
    pot_keys = [key for key in analog_keys if '_pot' in key]
    analog = pynwb.behavior.Position(name='trodes analog')
    analog.create_spatial_series(
        name=analog_keys[0],
        data=trodes_data['analog'][pot_keys[0]],
        reference_frame='0',
        timestamps=trodes_data['time']['time']
    )
    analog.fields['spatial_series'][analog_keys[0]].fields['unit'] = 'trodes bits'
    for key in pot_keys[1:]:
        spatial_series = pynwb.behavior.SpatialSeries(
            name=key,
            data=trodes_data['analog'][key],
            reference_frame='0',
            timestamps=trodes_data['time']['time']
        )
        analog.add_spatial_series(spatial_series)
        analog.fields['spatial_series'][key].fields['unit'] = 'trodes bits'
    nwb_file.add_acquisition(analog)
    return nwb_file


def add_trodes_dio(nwb_file, trodes_data):
    dio_keys = [*trodes_data['DIO']]
    dio = pynwb.behavior.BehavioralEpochs(name='trodes dio')
    for key in dio_keys:
        start_times = trodes_data['DIO'][key][1::2]
        stop_times = trodes_data['DIO'][key][2::2]
        if len(start_times) > 0:
            interval_series = pynwb.misc.IntervalSeries(
                name=key,
                data=[1, -1],
                timestamps=[start_times[0], stop_times[0]]
            )
            for i in range(len(start_times))[1:]:
                interval_series.add_interval(
                    start=float(start_times[i]),
                    stop=float(stop_times[i])
                )
            dio.add_interval_series(interval_series)
    nwb_file.add_acquisition(dio)
    return nwb_file


def trodes_to_nwb(nwb_file, data_dir, trodes_name):
    trodes_data = trodes_edp.import_trodes_data(data_dir, trodes_name)
    nwb_file = add_trodes_analog(nwb_file, trodes_data)
    nwb_file = add_trode_dio(nwb_file, trodes_data)
    return nwb_file
    # timestamps_reference_time

    # general time series
    test_ts = pynwb.TimeSeries(name='test_timeseries', data=data, unit='m', timestamps=timestamps)
    nwbfile.add_acquisition(test_ts)
    reuse_ts = pynwb.TimeSeries('reusing_timeseries', newdata, 'SIunit', timestamps=test_ts)
    # electrophysiology data
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


def controller_to_nwb(nwb_file, dir):
    return nwb_file


def config_to_nwb(nwb_file, dir):
    # session_id
    # stimulus_notes
    # devices
    return nwb_file


def link_videos(nwb_file, dir):
    return nwb_file


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


def match_times(controller_data, experiment_data):
    """

    Parameters
    ----------
    controller_data : list
        list of experimental controller variables and values
    experiment_data : dict
        dict of trodes experimental data per session
    Returns
    -------
    controller_time_normalized : array
        array of controller times matched to trodes times, syncing controller and trodes signals
    """
    controller_time = np.asarray(controller_data['time'] / 1000)  # convert to s
    exposures = experiment_data['DIO']['topCam']  # exposure data
    exposures = get_exposure_times(exposures)
    controller_time_normalized = controller_time - controller_time[-1] + exposures[-1]
    return controller_time_normalized
