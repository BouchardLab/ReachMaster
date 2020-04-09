"""A module for exploratory visualizations of raw trodes, controller, and
video data sources.

"""

import matplotlib.pyplot as plt
from .. import preprocessing.trodes_data.experiment_data_parser as trodes_edp
from .. import preprocessing.controller_data.experiment_data_parser as controller_edp

def plot_trodes_timeseries(experiment_data, var_name, time_set=False):
    """

    Parameters
    ----------
    experiment_data : dict
        dict containing trodes experiment data
    var_name : str
        variable to be plotted, for example 'topCam' or 'moving'
    time_set : integer
        set a discrete plotting time range , from 0 to time_set, manually (in seconds)

    Returns
    -------

    """
    time = experiment_data['time']['time']
    time_var = 3
    if time_set:
        time = obtain_times(experiment_data, time_set)
    exp_var = experiment_data['DIO'][var_name]
    mask = create_DIO_mask(time, exp_var)
    plt.plot(time, mask)
    plt.xlabel('time (s)')
    plt.ylabel(var_name)
    plt.title(var_name + ' over experiment')
    plt.show()
    return