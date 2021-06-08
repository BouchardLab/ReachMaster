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
import pdb
from software.ReachPredict3D.predictions import GetPredictions, run_main

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


def reconstruct_3d(dlt_coefs_file, dlc_files,filter_2D=True):
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
    xyz_all = np.empty([frames, len(names), 3])
    filtered_cam_data = np.empty([frames, len(names),6])
    _p_d = np.empty([frames, len(names), 3])
    pp = Pool()  # generate MP pool object
    for k in tqdm(range(len(names))):  # Loop over each bodypart
        try:
            p_data, cam_datas = loop_pdata([k, frames, cameras, dlc_files, dlt_coeffs])  # get DLC prob_data and position data
        except:
            p_data = 0
            cam_datas = 0
            print('didnt get cam/prob data  ')
            print(p_data)
        if filter_2D:
            cam_datas = filter_predictions_2D(cam_datas,p_data)
            # save filtered and interpolated 2-D predictions to load into NWB
            filtered_cam_data[:,k,:] = cam_datas
        xyz__ = pp.map(r_d_i, [[dlt_coeffs, cam_datas, p_data]])  # reconstruct coordinates
        xyz_all[:, k, :] = np.copy(xyz__)
        _p_d[:, k, :] = np.copy(p_data[:,::2])       
    print('Finished block coordinates!')
    pp.close()
    return xyz_all, names , _p_d, filtered_cam_data


def filter_predictions_2D(cam_preds, cam_weights,threshold_val = 0.9, windowlength = 9):
    """Function to filter our cam predictions before performing 3-D reconstruction. 
    Attributes
    -------------
    cam_preds: array, camera data
    cam_weights: array, p weights

    Returns
    -----------
    new_preds: array, filtered predictions of our camera predictions

    """
    # 2-D interpolation with numpy
    new_preds = np.copy(cam_preds)
    #pdb.set_trace()
    win = np.hamming(windowlength)
    for irx in range(0,cam_preds.shape[1]): # iterate over cams
        # create mask of low probability events
        mask = cam_weights[:,irx] < threshold_val # cam value
        # use mask to interpolate points with low confidence
        new_preds[mask,irx] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam_preds[:,irx][~mask])
        # Smooth all points with Hamming window
        new_preds[:,irx] = np.convolve(win/win.sum(), new_preds[:,irx], mode='same')
    return new_preds


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


def get_file_sets(root_path, resnet_version, dlc_network_shuffle= 'shuffle2',dlc_network_iteration = '500000', filtered=False):
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
    aaa = [k for k in all_files if dlc_network_iteration in k]
    nf = [k for k in aaa if dlc_network_shuffle in k]
    file_list = 0
    for file in nf:
        if 'cam3' in file:
            files = file.rsplit('/', 1)[1]
            names = str(files.rsplit('_cam3', 1)[0])
            file_list = [file_list for file_list in nf if names in file_list]
            for s in file_list:
                if "cam1" in s:
                    cam1_list.append(s)
                if "cam2" in s:
                    cam2_list.append(s)
                if "cam3" in s:
                    cam3_list.append(s)
    print('Total Number of 3-D reconstructable DLC predictions found for this rat is ' + str(len(cam1_list)))
    if file_list:
        print('Total Number of files in the block is ' + str(len(file_list)))
    else:
        cam1_list, cam2_list, cam3_list = 0, 0, 0
        print('No 2-D DeepLabCut prediction files found for this experimental block for your given network. Please re-run the script DLC.py on this rats block.')
    return cam1_list, cam2_list, cam3_list




def get_save_path(path_of_dir, three = True, filterpaths = True):
    return path_of_filter, path_of_three


def get_kinematic_data(root_path, dlt_path, resnet_version, rat, date, session, dim, save=True,predict=False,split=False,RWB=False,kinematics=False):
    """
    Function intended to take in single blocks of reaching experiments and return the calculated 3-D positional
    variables of interest from DeepLabCut. This is for a SINGLE block.
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
        cam_path_save = '/'
        filtered_save_pathr = '/'
        cam_path = cam_list[0].rsplit('/')[1:-1] # take non-named part of block path
        for c in cam_path:
            cam_path_save += c + ('/')
        previous_save_flag = search_root_for_previous_predictions(cam_path_save)
        if previous_save_flag:
            filtered_camw = previous_save_flag.rsplit('/')[:-1] 
            for f in filtered_camw:
                filtered_save_pathr += f + ('/')
            filtered_save_path = filtered_save_pathr + 'filtered_2d_preds.csv'
            filtered_cam_coords = pd.read_csv(filtered_save_path)
            merge_df = pd.read_hdf(previous_save_flag)
        else:
            if cam_list:
                    xyzAll, labels, pxyzAll, filtered_cam_coords = reconstruct_3d(dlt_path, cam_list)
                    filtered_2d_savepath = cam_path_save + 'filtered_2d_preds.csv'
                    pd.DataFrame(filtered_cam_coords.reshape(filtered_cam_coords.shape[0], filtered_cam_coords.shape[1] * filtered_cam_coords.shape[2])).to_csv(filtered_2d_savepath, header=None, index=None) 
                    print('Reconstructed 3-D coordinates for experimental block  ' + str(cam1_list))
                    coords = ['X', 'Y', 'Z']
                    probs = [ 'prob_c1', 'prob_c2', 'prob_c3' ]
                    header = pd.MultiIndex.from_product([[rat], [session], [date], [dim], labels,
                                         coords],
                                        names=['rat', 'session', 'date', 'dim', 'bodyparts', 'coords'])
                    pheader = pd.MultiIndex.from_product([[rat], [session], [date], [dim], labels,
                                         probs],
                                        names=['rat', 'session', 'date', 'dim', 'bodyparts', 'cam_probs'])
                    try:
                        df = pd.DataFrame(xyzAll.reshape((xyzAll.shape[0], xyzAll.shape[1]*xyzAll.shape[2])), columns=header)
                        pdfs = pd.DataFrame(pxyzAll.reshape((pxyzAll.shape[0], pxyzAll.shape[1]*pxyzAll.shape[2])), columns = pheader)
                    except:
                        print('Error in creating position and probability dataframes for ' + str(cam_list))
                        df = [0]
                    try:
                        merge_df = df.join(pdfs)
                    except:
                        print('Problem merging dataframes')
                        merge_df = [0]
                    if save:
                        print('Saving Our Positional DataFrame') 
                        merge_df.to_hdf(cam_path_save +'predictions_3D_savefile.h5',key='save_df',mode='w')
                        merge_df.to_hdf('~/Data/placeholder_df.h5', key='save_df', mode='w')
            else:
                merge_df = [0]
                print('No full video source found for ' + str(cam_list))
    else:
        dlc_path = root_path[0:-4]
        call_DLC_predictions(dlc_path)
        print('Rerun this block')
        merge_df = [0]
    if predict:
        #prediction_array = get_predictions(merge_df,edf)
        pass
    if split:
        # split_array = get_splits(prediction_array,merge_df)
        pass
# experimental data we need: entire experimental block from NWB
    if RWB: # merge a call into RWB
        pass
    if kinematics:
        pass
    return merge_df


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
        print('Not enough video File(s) are available')
        cu = 1
    return cu


def search_root_for_previous_predictions(cam_path):
    previous_save_flag = 0
    previous_save_path = 0
    #pdb.set_trace()
    for files in glob.glob(cam_path +'**3D_savefile.h5'):
        if files:
            previous_save_path = files
        else:
            previous_save_path = previous_save_flag
    return previous_save_path


def call_DLC_predictions(root_path,config_path = '/clusterfs/NSDS_data/brnelson/DLC/LabLabel-Lab-2020-10-27/config.yaml'):
    dlc_root = root_path + '/videos/'
    GetPredictions() # initialize our 
    run_main(root_path,config_path)
    return 