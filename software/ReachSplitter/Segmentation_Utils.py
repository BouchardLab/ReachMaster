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
    return rpt_result # returns



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


def prob_mask_segmentation(prob_array,feature_array,threshold=0.01,rs1=4,rs2=6,feat_dim=True):
    # get indices of probability values under threshold
    for ris in range(rs1,rs2):
        low_pvalue_indices_c1 = np.where(prob_array[0,ris,:] < threshold)
        low_pvalue_indices_c2 = np.where(prob_array[1,ris,:] < threshold)
        low_pvalue_indices_c3 = np.where(prob_array[2,ris,:] < threshold)
        low_pvalue_indices_ct = np.intersect1d(low_pvalue_indices_c1,low_pvalue_indices_c2,low_pvalue_indices_c3)
        if feat_dim:
            for dx in range(0,3):
                feature_array[dx,:] [low_pvalue_indices_ct] = 0 # camera 3 is underneath and is an excellent way to check if the arm is out
                feature_array[dx,:] [low_pvalue_indices_ct] = 0
                feature_array[dx,:] [low_pvalue_indices_ct] = 0
    return feature_array

def create_arm_feature_arrays_trial(a,e_d,p_d,ii,left=False,hand=False):

    """
    Function to create arm and hand objects from DLC-generated positional data on a per-trial basis.
    Inputs:
    a : Array size [Trials, Dims, BParts, WLen + Pre)
    e_d : (Trials, 12,  WLen+Pre)
    p_d : (Trials, Dims, BParts, WLen+Pre)

    Outputs:
    a_ : unique type of feature vector (Trials, Dims, features, WLen+Pre)
    """
    #pdb.set_trace()
    if left:
        if hand:
            a_ = a[ii, :, 7:15, :]
        else:
            a_ = a[ii, :, 4:7, :] # sum over shoulder, forearm, palm, wrist
    else:
        if hand:
            a_= a[ii, :, 19:27, :]
        else:
            a_ = a[ii, :, 16:19, :] # sum over shoulder, forearm, palm, wrist
    #pdb.set_trace()
    for tc in range(0,3):
        a_[:, tc, :] = prob_mask_segmentation(p_d[ii, :, :, :], a_[:, tc, :])# threshold possible reach values
   # pdb.set_trace()
    return a_


def merge_in_swap(init_arm_array, ip_array):
    for trials in range(0, 5):
        plt.plot(init_arm_array[trials, 0, 7, :])
    plt.show()

    c = init_arm_array[:, :, 14:27, :]

    for trials in range(0, init_arm_array.shape[0] - 40):
        plt.plot(c[trials, 0, :, :])
    plt.show()
    pc = ip_array[:, :, 0:13, :]

    for trials in range(0, init_arm_array.shape[0] - 40):
        plt.plot(pc[trials, 0, :, :])
    plt.show()

    # init_arm_array[:,:,14:27,:] = pc
    # ip_array[:,:,0:13,:] = c

    for trials in range(0, init_arm_array.shape[0] - 40):
        plt.plot(ip_array[trials, 0, 0:13, :])
    plt.show()
    for trials in range(0, 5):
        plt.plot(init_arm_array[trials, 0, 7, :])
    plt.show()
    return init_arm_array, ip_array


