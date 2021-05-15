"""
Functions intended to import and transform behavioral reaching data from ReachMaster experiments into 3-D kinematics
Use with DLT co-effecients obtained through easyWand or other procedures + multi-camera predictions from DLC
Author: Brett Nelson, NSDS Lab 2020

"""
from multiprocessing import Pool
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm


def dlt_reconstruct(c, camPts, weights=None):
    """
    Function to reconstruct multi-camera predictions from 2-D camera space into 3-D euclidean space
    Credit: adapted by An Chi Chen from DLTdv5 by Tyson Hedrick, edited by BN 8/3/2020
    Attributes
    ----------
    c: list or array of DLT co-effecients for the camera system in question
    camPts: array of points from the camera system (can be 2, 3 cameras etc)
    weights:
    Returns
    -------
    xyz: array of positions in 3-D space for N bodyparts over T timeframe
    """
    nFrames = len(camPts)
    nCams = len(camPts[0]) / 2
    # setup output variables
    xyz = np.empty((nFrames, 3))
    rmse = np.empty((nFrames, 1))
    # process each frame
    for i in range(nFrames):
        # get a list of cameras with non-NaN [u,v]
        cdx_size = 0
        cdx_temp = np.where(np.isnan(camPts[i - 1, 0:int(nCams * 2) - 1:2]) == False, 1, 0)
        for x in range(len(cdx_temp)):
            if cdx_temp[x - 1] == 1:
                cdx_size = cdx_size + 1
        cdx = np.empty((1, cdx_size))
        for y in range(cdx_size):
            cdx[0][y] = y + 1
        # if we have 2+ cameras, begin reconstructing
        if cdx_size >= 2:
            # initialize least-square solution matrices
            m1 = np.empty((cdx_size * 2, 3))
            m2 = np.empty((cdx_size * 2, 1))
            temp1 = 1
            temp2 = 1
            for z in range(cdx_size * 2):
                if z % 2 == 0:
                    m1[z, 0] = camPts[i - 1, (temp1 * 2) - 2] * c[8, (temp1 - 1)] - c[0, (temp1 - 1)]
                    m1[z, 1] = camPts[i - 1, (temp1 * 2) - 2] * c[9, (temp1 - 1)] - c[1, (temp1 - 1)]
                    m1[z, 2] = camPts[i - 1, (temp1 * 2) - 2] * c[10, (temp1 - 1)] - c[2, (temp1 - 1)]
                    m2[z, 0] = c[3, temp1 - 1] - camPts[i - 1, (temp1 * 2) - 2]
                    temp1 = temp1 + 1
                else:
                    m1[z, 0] = camPts[i - 1, (temp2 * 2) - 1] * c[8, temp2 - 1] - c[4, temp2 - 1]
                    m1[z, 1] = camPts[i - 1, (temp2 * 2) - 1] * c[9, temp2 - 1] - c[5, temp2 - 1]
                    m1[z, 2] = camPts[i - 1, (temp2 * 2) - 1] * c[10, temp2 - 1] - c[6, temp2 - 1]
                    m2[z, 0] = c[7, temp2 - 1] - camPts[i - 1, (temp2 * 2) - 1]
                    temp2 = temp2 + 1
            # get the least squares solution to the reconstruction
            if weights.any():
                w = np.sqrt(np.diag(weights[i - 1, :]))
                m1 = np.matmul(w, m1)
                m2 = np.matmul(w, m2)
            Q, R = np.linalg.qr(m1)  # QR decomposition with qr function
            y = np.dot(Q.T, m2)  # Let y=Q'.B using matrix multiplication
            x = np.linalg.solve(R, y)  # Solve Rx=y
            xyz_pts = x.transpose()
            xyz[i, 0:3] = xyz_pts
            uv1 = np.matmul(m1, x)
            # compute the number of degrees of freedom in the reconstruction
            dof = m2.size - 3
            # estimate the root mean square reconstruction error
            c1 = np.sum((m2 - uv1) ** 2) / dof
            rmse[i] = np.sqrt(np.abs(c1 / nFrames - 1))  # RMSE is a scalar, take abs value before applying sqrt.
    return xyz, rmse


def reconstruct_3d(dlt_coefs_file, dlc_files):
    """Perform 3-D reconstruction using DLT co-effecients and extracted multi-camera predictions
    Attributes
    ----------
    dlt_coefs_file: array containing 3x4 matrix of DLT co-effecients, found using easyWand
    dlc_files: list of paths to .csv files (extracted predictions from DLC for each camera)

    Returns
    -------
    xyz_all: N x T array,where N is the number of parts tracked and T is the length of frames in a given video or trial

    """
    try:
        dlt_coeffs = np.loadtxt(dlt_coefs_file, delimiter=",")
    except:
        print('Failed to get coeffs')
    first_dataset = np.loadtxt(dlc_files[0], dtype=str, delimiter=',')
    names = first_dataset[1, range(1, first_dataset.shape[1], 3)]
    frames = first_dataset.shape[0] - 3
    cameras = len(dlc_files)
    xyz_all = np.empty([frames, len(names), 6])
    pp = Pool()  # generate MP pool object
    for k in tqdm(range(len(names))):  # Loop over each bodypart
        try:
            p_data, cam_datas = loop_pdata(
                [k, frames, cameras, dlc_files, dlt_coeffs])  # get DLC prob_data and position data
        except:
            p_data = 0
            cam_datas = 0
            print('didnt get cam/prob data  ')
            print(p_data)
        _p_d = p_data[:, ::2]
        try:
            xyz__ = pp.map(r_d_i, [[dlt_coeffs, cam_datas, p_data]])  # reconstruct coordinates
            xyz_all[:, k, 0:3] = np.copy(xyz__)
            xyz_all[:, k, 3:6] = _p_d
        except:
            xyz_all = [0, 0, 0, 0, 0, 0, 0]
            print('Failed to reconstruct video : problem with ')
    print('Finished block coordinates!')
    pp.close()
    return xyz_all, names


def loop_pdata(v):
    """Function to create data structures for positions and p-values for a given session
    Attributes
    --------------
    v: list

    Returns
    --------
    p_data: array containing p values size frames by N_cameras X 2
    cam_data: array containing position values size frames by N_Cameras X 2
    """
    k_, frames_, cameras_, dlc_files_, dlt_coeffs_ = v
    cam_data = np.empty([frames_, 2 * cameras_], dtype=float)
    csv_index = int((k_ * 3) + 1)
    p_data = np.empty([frames_, cameras_ * 2], dtype=float)
    for cam in range(cameras_):
        col = cam * 2
        cam_data[:, col] = np.loadtxt(dlc_files_[cam], dtype=float, delimiter=',', skiprows=3, usecols=csv_index)
        cam_data[:, col + 1] = np.loadtxt(dlc_files_[cam], dtype=float, delimiter=',', skiprows=3,
                                          usecols=(csv_index + 1))
        p_data[:, col] = np.loadtxt(dlc_files_[cam], dtype=float, delimiter=',', skiprows=3, usecols=(csv_index + 2))
        p_data[:, col + 1] = np.loadtxt(dlc_files_[cam], dtype=float, delimiter=',', skiprows=3,
                                        usecols=(csv_index + 2))
    return p_data, cam_data


def r_d_i(vv):
    """ Function for multiprocessing, intended to run the reconstruction code

    Attributes
    --------------
    vv: list, containing dlt_coeffecients, p-values, and positions for a given session

    Returns
    ---------------
    xyz_: 3-D reconstructed positions and probabilities for a given session
        """
    dlt_coeffs__ = vv[0]
    cam_data__ = vv[1]
    weights_ = vv[2]
    xyz_, rmse_ = dlt_reconstruct(dlt_coeffs__, cam_data__, weights_)
    # Call for unweighted reconstruction (no reliance on p values)
    # xyz_ , rmse_ = dlt_reconstruct(dlt_coeffs__, cam_data__)
    return xyz_


def reconstruct_points(list_of_csv, coeffs):
    """
    Attributes
    ----------
    list_of_csv : list of file names to use in reconstruction [file1_cam1.csv, file2cam2.csv...]
    coeffs : DLT co-effecient array
    output_path : Boolean, option to set output path
    plotting : Boolean,option to set plot
    Returns
    -------
    xyzAll: array containing 3-D kinematics for given videos
    """
    try:
        xyzAll = reconstruct_3d(coeffs, list_of_csv)
    except:
        xyzAll = [0, 0, 0, 0, 0, 0, 0, 0]
        print('reconstruct_error')
    return xyzAll


def find_cam_files(root_dir, extension, pathList):
    """
    Function to find cam files for 3-D calibration
    Attributess
    ----------
    root_dir: path directory
    extension: string of extension you want found eg .mp4 or .txt

    pathList: empty list
    Returns
    -------
    pathList: list containing lists of file names
    """
    for file in glob.glob(root_dir, extension, recursive=True):
        if file.find('cam1'):  # check and make sure that the files have been split
            if file.find('catscan'):
                cam_path = file.rsplit('_')[0]
                cam2_path = cam_path + 'cam2.mp4'
                cam3_path = cam_path + 'cam3.mp4'
                pathList.append([file, cam2_path, cam3_path])
        else:
            print('Sorry, file not found :: file designation' + file)

    return pathList


def get_file_sets(root_path, resnet_version, dlc_network='shuffle2_500000', filtered=False):
    """ function intended to fetch the proper network file sets for a given session. A session directory
    may have multiple predictions from unique DLC networks saved to a folder. This requires us to dictate which
    network's predictions to use
    Attributes
    ------------
    root_path: str, path of root experimental directory
    resnet_version: str, network version of DLC to use
    dlc_network: str, name of network to use in advanced string filtering
    filtered: bool, flag to filter DLC predictions
    Returns
    -------------
    cam1_list: list containing cam1 DLC predictions for a specified network
    cam2_list: list containing cam2 DLC predictions for a specified network
    cam3_list: list containing cam3 DLC predictions for a specified network

    """
    cam1_list = []
    cam2_list = []
    cam3_list = []
    all_files = glob.glob(root_path, recursive=True)
    all_files.extend(glob.glob(resnet_version))
    all_files.extend(glob.glob(dlc_network))
    if filtered:
        all_files.extend(glob.glob('_filtered'))
    print(all_files)
    for file in all_files:
        if 'cam3' in file:
            files = file.rsplit('/', 1)[1]
            names = str(files.rsplit('_cam3', 1)[0])
            file_list = [file_list for file_list in all_files if names in file_list]
            for s in file_list:
                if "cam1" in s:
                    if "shuffle5" in s:
                        cam1_list.append(s)
                if "cam2" in s:
                    cam2_list.append(s)
                if "cam3" in s:
                    cam3_list.append(s)
    print('Total Number of 3-D reconstructable DLC predictions found for this rat is ' + str(len(cam1_list)))
    print('Total Number of files in the block is ' + str(len(all_files)))
    return cam1_list, cam2_list, cam3_list


def get_kinematic_data(root_path, dlt_path, resnet_version, rat, date, session, dim, save=True):
    """
    Function intended to take in single blocks of reaching experiments and return the calculated 3-D positional
    variables of interest from DeepLabCut" This is for a SINGLE block.
    Attributes
    --------------
    root_path: str, path of root experimental directory
    dlt_path: str, path to dlt co-effecient .csv files
    resnet_version: str, network version of DLC to use
    rat: str,  name of rat
    date: str, date of experimental session
    session: int, session number, integer
    dim: int, session dim
    save: bool, flag to save kinematic data

    Returns
    ------------
    df: dataframe, contains positional predictions in 3-D for a given experimental session

    """
    cam1_list, cam2_list, cam3_list = get_file_sets(root_path, resnet_version)
    cam_list = 0
    try:
        for i, val in enumerate(cam1_list):
            cam_list = [cam1_list[i], cam2_list[i], cam3_list[i]]
            cu = filter_cam_lists(cam_list)
            if cu == 1:
                print('n')
    except:
        cam_list = 0
    if cam_list:
        xyzAll, labels = reconstruct_3d(dlt_path, cam_list)
        print('Reconstructed 3-D coordinates for block' + str(cam1_list))
    else:
        df = [0]
        print('No Video for ' + str(cam_list))
    coords = ['X', 'Y', 'Z', 'prob_c1', 'prob_c2', 'prob_c3']
    header = pd.MultiIndex.from_product([[rat], [session], [date], [dim], labels,
                                         coords],
                                        names=['rat', 'session', 'date', 'dim', 'bodyparts', 'coords'])
    try:
        df = pd.DataFrame(xyzAll.reshape((xyzAll.shape[0], xyzAll.shape[1] * xyzAll.shape[2])), columns=header)
    except:
        print('Error in reshaping and holding DF')
        df = [0]
    if save:
        print('Saving Kinematic DF')
        df.to_hdf('~/Data/kinematics_p_placeholder_df.h5', key='save_df', mode='w')
    return df


def filter_cam_lists(cam_list):
    """ function intended to ensure DLC has been run on each session
    Attributes
    ---------------
    cam_list: list containing each camera file

    Returns
    -----------
    cu: bool , 0 if ok 1 if not ok
    """
    cu = 0
    c1 = cam_list[0]
    c2 = cam_list[1]
    c3 = cam_list[2]
    # compare lengths
    if len(c1) == len(c2) == len(c3):
        cu = 0
    else:
        print('Video File(s) are not available')
        cu = 1
    return cu
