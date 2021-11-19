import pdb
from math import ceil
from scipy import linalg
import pandas as pd
from Analysis_Utils import preprocessing_df as preprocessing
from moviepy.editor import *
import cv2
import numpy as np
from errno import EEXIST, ENOENT
import shutil
from scipy import interpolate, signal
from scipy.signal import butter, sosfiltfilt
from csaps import csaps
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val

def read_from_csv(input_filepath):
    """ Function to read in csv. """
    input_df = pd.read_csv(input_filepath)
    return input_df


def autocorrelate(x, t=1):
    """ Function to compute regular auto correlation using numpy. """
    return np.corrcoef(np.array([x[:-t], x[t:]]))
# Code adapted from Github page of agramfort
def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

def knn_smoother(input_data, neighbors):
    n_neighbors = np.arange(1, neighbors)
    knn = val.SmoothingParameterSearch(
        ks.KNeighborsSmoother(),
        n_neighbors,
    )
    knn.fit(input_data)
    knn_smoothed = knn.transform(input_data)
    return knn_smoothed

def filter_vector_hamming(input_vector, window_length=3.14):
    """ Function to filter input vectors using Hamming-Cosine window. Used exclusively for DLC-based inputs, not 3-D
    trajectories. """
    filtered_vector = np.zeros(input_vector.shape)
    for i in range(0, input_vector.shape[1]):
        win = np.hamming(window_length)
        filtered_vector[:, i] = np.convolve(win / win.sum(), input_vector[:, i], mode='same')
    return filtered_vector


def butterworth_filtfilt(input_vector, nyquist_freq, cutoff, filt_order=4):
    sos = butter(filt_order, cutoff / nyquist_freq, output='sos')
    y = sosfiltfilt(sos, input_vector.reshape(input_vector.shape[1], input_vector.shape[0]))
    return y.reshape(y.shape[1], y.shape[0])


def cubic_spline_smoothing(input_vector, spline_coeff=0.1):
    timepoints = np.linspace(0, input_vector.shape[0], input_vector.shape[0])
    smoothed_vector = np.zeros(input_vector.shape)
    for i in range(0, 3):
        smoothed_vector[:, i] = csaps(timepoints, input_vector[:, i], timepoints, normalizedsmooth=True,
                                      smooth=spline_coeff)
    return smoothed_vector


def filter_vector_median(input_vector, window_length=3):
    filtered_vector = np.zeros(input_vector.shape)
    filtered_vector[2:-3, :] = signal.medfilt(input_vector[2:-3, :], kernel_size=window_length)
    filtered_vector[0:2, :] = input_vector[0:2, :]
    filtered_vector[-3:-1, :] = input_vector[-3:-1, :]
    return filtered_vector


def interpolate_1d_vector(vec, int_kind='cubic'):
    """ Function to interpolate and re-sample over a 1-D vector using a cubic interpolation."""
    idx = np.nonzero(vec)
    vec_int = 0
    if idx[0].any():
        xx = np.arange(0, vec.shape[0], 1)
        fx = interpolate.interp1d(xx[idx], vec[idx], kind=int_kind, assume_sorted=False)
        vec_int = fx(vec)
    return vec_int


def interpolate_3d_vector(xkin_three_vectors, velocity_index, prob_index, gap_num=4):
    """ Function to interpolate and re-sample, using specified indices of outliers, over a full 3-D vector. """
    gap_index = []
    interpolation_number = 0
    idx = np.union1d(velocity_index, prob_index)
    for i in range(0, xkin_three_vectors.shape[1]):
        # This gets your interpolation indices
        if idx.any():  # If there are any interpolation gaps
            # Handle gaps, we have to chunk up the array into each "int" piece
            xx = np.arange(0, xkin_three_vectors.shape[0], 1)
            uvs = xkin_three_vectors[:, i]
            uvs_mask = np.zeros((xkin_three_vectors.shape[0]))
            uvs_mask[idx] = 1
            cz = 0
            for il, id in enumerate(uvs_mask):  # We need to make sure we aren't interpolating over large gaps!!
                if 3 < il < (len(uvs_mask) - 2):  # Need to keep boundaries at beginning to ensure we dont overflow
                    if id == 0:
                        cz += 1
                    if id == 1:  # If we have a non-thresholded value
                        if uvs_mask[il + 1] == 1 and uvs_mask[il + 2] == 1:
                            if 0 < cz < gap_num:
                                interpolation_number += 1
                                gap_index.append([il])
                                ff = interpolate.interp1d(xx[il - cz - 1:il + 1], uvs[il - cz - 1:il + 1],
                                                          kind='linear', assume_sorted=False)
                                xkin_three_vectors[il - cz:il, i] = ff(xx[il - cz:il])  # Take middle values
                                cz = 0
                            if cz == 0:  # no gap
                                continue
    return np.asarray(xkin_three_vectors), interpolation_number, np.squeeze(np.asarray(gap_index))


def norm_coordinates(kin_three_vector, aff_t=False):
    """ Function to import and transform kinematic data using pre-generated affine transformation. For more information on
    generating this transformation, see ReachPredict3D's documentation on handle matching."""
    if aff_t:
        ax = aff_t[0, 0]
        bx = aff_t[0, 1]
        cx = aff_t[0, 2]
        a = aff_t[1, 0]
        b = aff_t[1, 1]
        c = aff_t[1, 2]
    else:
        ax = -2.5
        bx = -0.2
        cx = 1.5
        a = .25
        b = .25
        c = .5
    xkin_three_vector = np.zeros(kin_three_vector.shape)
    xkin_three_vector[:, 0] = kin_three_vector[:, 0] * ax + a  # flip x-axis
    xkin_three_vector[:, 1] = kin_three_vector[:, 1] * bx + b  # flip y-axis
    xkin_three_vector[:, 2] = kin_three_vector[:, 2] * cx + c
    return np.copy(xkin_three_vector)


def rescale_frame(framez, percent=150):
    """ Function to rescale video arrays. """
    width = int(framez.shape[1] * percent / 100)
    height = int(framez.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(framez, dim, interpolation=cv2.INTER_AREA)


def mkdir_p(my_path):
    """Creates a directory. equivalent to using mkdir -p on the command line. 

    Returns
    -------
    object
    """
    try:
        os.makedirs(my_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(my_path):
            pass
        else:
            raise


def rm_dir(my_path):
    """Deletes a directory. equivalent to using rm -rf on the command line"""
    try:
        shutil.rmtree(my_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == ENOENT:
            pass
        else:
            raise


def import_robot_data(df_path):
    """ Imports experimental "robot" data used in analyzing reaching behavioral data. """
    df = pd.read_pickle(df_path)
    df = preprocessing(df)
    return df
