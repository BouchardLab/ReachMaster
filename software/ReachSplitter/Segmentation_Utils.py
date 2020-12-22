"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 12/8/2020
    This code is intended to provide utility functions for behavioral time-series segmentation algorithmic approaches.


    Use Case:
    Import ML-ready feature vectors from a previously CLASSIFIED trial.
    Import ML-non-ready label vectors.

"""
# Trial Segmentation -> provides indices for NWB
# Trial Classification -> provides raw trial features for NWB
# ToDo:
import numpy as np
import DataStream_Vis_Utils as utils
import Classification_Utils as CU
import matplotlib.pyplot as plt
import ruptures as rpt


def segment_time_series(time_series, bkps=1, gamma=1e-2, display=True):
    """
    Implementation of ruptures, python library https://github.com/deepcharles/ruptures
    Usage: Segmentation of time-series data

    Inputs:
        time_series: time series kinematic data from the ReachMaster experimental system
        bkps : int, number of changepoints (reaches), 1 for simple detection
        display: Boolean, variable to set display functionality of ruptures
    Returns:

        rpt_result: List containing breakpoint indexes and the total length of the time-series array
    """

    params = {"gamma": gamma}
    algo = rpt.Dynp(model="rbf", params=params, jump=1, min_size=2).fit(
        time_series
    )
    rpt_result = algo.predict(n_bkps=bkps)
    if display:
        rpt.display(time_series, bkps, rpt_result)
        plt.show()
    return rpt_result


def get_arm_velocities(postures):
    #
    velocities = np.diff(postures, axis=1)
    return velocities


def segment_by_arm_velocities(postures):
    """

    Parameters
    ----------
    postures

    Returns
    -------

    """
    arm_velocities = get_arm_velocities(postures)
    reach_start = arm_velocities

    return reach_start

def classify_and_segment_simple_reaches(list_of_time_series):
    # Reach or Other (?)
    # Workflow:
    # Anomaly Detection from kinematic time-series to obtain beginning of reach used on pre-classified trial data
    # Segment -5 frames to condition:: increase in handle velocity, X frames, segment boundaries

    # return summary statistics, best reach start indices, best grasp indices, reward indices
    summary_statistics = []
    best_reach_indices = []
    best_behavioral_indices = []


    return summary_statistics, best_reach_indices, best_behavioral_indices


