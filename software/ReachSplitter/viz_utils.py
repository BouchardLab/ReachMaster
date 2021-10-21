import pandas as pd
from Analysis_Utils import preprocessing_df as preprocessing
from moviepy.editor import *
import cv2
import numpy as np
from errno import EEXIST, ENOENT
import shutil
from scipy import interpolate


def read_from_csv(input_filepath):
    """ Function to read in csv. """
    input_df = pd.read_csv(input_filepath)
    return input_df


def autocorrelate(x, t=1):
    """ Function to compute regular autocorration using numpy. """
    return np.corrcoef(np.array([x[:-t], x[t:]]))


def filter_vector_hamming(input_vector, window_length=3):
    """ Function to filter input vectors using Hamming-Cosine window. """
    filtered_vector = np.zeros(input_vector.shape)
    for i in range(0, input_vector.shape[1]):
        win = np.hamming(window_length)
        filtered_vector[:, i] = np.convolve(win / win.sum(), input_vector[:, i], mode='same')
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


def interpolate_3d_vector(xkin_three_vectors, velocity_index, prob_index, gap_num=3):
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
                                                          kind='slinear', assume_sorted=False)
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
