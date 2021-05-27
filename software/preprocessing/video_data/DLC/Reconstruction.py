"""
Functions intended to import and transform behavioral reaching data from ReachMaster experiments into 3-D kinematics
Use with DLT co-effecients obtained through easyWand or other procedures + multi-camera predictions from DLC
Author: Brett Nelson, NSDS Lab 2020

"""
from errno import EEXIST, ENOENT
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import glob
import pandas as pd


# dlt reconstruct adapted by An Chi Chen from DLTdv5 by Tyson Hedrick
def dlt_reconstruct(c, camPts, weights=None):
    """
    Function to reconstruct multi-camera predictions from 2-D camera space into 3-D euclidean space
    Credit: adapted by An Chi Chen from DLTdv5 by Tyson Hedrick, edited by BN 8/3/2020
    Parameters
    ----------
    c : list or array of DLT co-effecients for the camera system in question
    camPts : array of points from the camera system (can be 2, 3 cameras etc)

    Returns
    -------
    xyz : array of positions in 3-D space for N bodyparts over T timeframe
    """
    # number of frames
    nFrames = len(camPts)
    # number of cameras
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
            if isinstance(weights, np.ndarray):
                w = np.sqrt(np.diag(weights[i - 1, :]))
                # print(w.shape,m1.shape,m2.shape)
                m1 = np.matmul(w, m1)
                m2 = np.matmul(w, m2)
            Q, R = np.linalg.qr(m1)  # QR decomposition with qr function
            y = np.dot(Q.T, m2)  # Let y=Q'.B using matrix multiplication
            x = np.linalg.solve(R, y)  # Solve Rx=y
            xyz_pts = x.transpose()

            xyz[i, 0:3] = xyz_pts
            # print(xyz)
            # compute ideal [u,v] for each camera
            # uv=m1*xyz[i-1,0:2].transpose

            # compute the number of degrees of freedom in the reconstruction
            # dof=m2.size-3

            # estimate the root mean square reconstruction error
            # rmse[i,1]=(sum((m2-uv)**2)/dof)^0.5

    return xyz


def reconstruct_3d(dlt_coefs_file, dlc_files, output_format, plotting=False, weighted=False):
    """Perform 3-D reconstruction using DLT co-effecients and extracted multi-camera predictions
    Parameters
    ----------
    dlt_coefs_file : array containing 3x4 matrix of DLT co-effecients, found using easyWand

    dlc_files : list of paths to .csv files (extracted predictions from DLC for each camera)

    output_format : format of data output ex. .csv, .h5

    plotting : Boolean, optional, return plot options
    Returns
    -------
    xyz_all : N x T array,where N is the number of parts tracked and T is the length of frames in a given video or trial

    """

    # Load DLT Coefficient
    dlt_coefs = np.loadtxt(dlt_coefs_file, delimiter=",")

    # Get Names of Labels
    first_dataset = np.loadtxt(dlc_files[0], dtype=str, delimiter=',')
    names = first_dataset[1, range(1, first_dataset.shape[1], 3)]
    frames = first_dataset.shape[0] - 3
    cameras = len(dlc_files)

    xyz_all = np.empty([frames, 4, len(names)])

    if plotting:
        # Color Settings for Figure
        cmap = 'jet'
        colormap = plt.get_cmap(cmap)
        markerSize = 75
        alphaValue = 1
        markerType = '*'

        col = colormap(np.linspace(0, 1, len(names)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for k in range(len(names)):
        # read in data from DLC
        cam_data = np.empty([frames, 2 * cameras], dtype=float)
        weights = np.empty([frames, 2 * cameras], dtype=float)
        csv_index = int((k * 3) + 1)
        for cam in range(cameras):
            col = cam * 2
            cam_data[:, col] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3, usecols=csv_index)
            cam_data[:, col + 1] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3,
                                              usecols=(csv_index + 1))
            weights[:, col] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3, usecols=(csv_index + 2))
            weights[:, col + 1] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3, usecols=(csv_index + 2))

        # combine
        if weighted:
            xyz = dlt_reconstruct(dlt_coefs, cam_data, weights)
        else:
            xyz = dlt_reconstruct(dlt_coefs, cam_data)
        xyz_k = np.append(xyz, np.mean(weights, axis=1)[:, np.newaxis], axis=1)
        xyz_all[:, :, k] = xyz_k
        if plotting:
            xs = np.copy(xyz[:, 0])
            ys = np.copy(xyz[:, 1])
            zs = np.copy(xyz[:, 2])
            ax.scatter(xs, ys, zs)

        # save xyz coords in csv file
        out_filename = (output_format + names[k] + '.csv')
        np.savetxt(out_filename, xyz_k, delimiter=',', fmt='%1.4f')

    if plotting:
        plt.legend(names)
        plt.show()

    return xyz_all, names


def reconstruct_points(list_of_csv, coeffs, output_path = False,plotting=False):
    """
    Parameters
    ----------
    list_of_csv : list of file names to use in reconstruction [file1_cam1.csv, file2cam2.csv...]
    coeffs : DLT co-effecient array
    output_path : Boolean, option to set output path
    plotting : Boolean,option to set plot
    Returns
    -------
    xyzAll : array containing 3-D kinematics for given videos
    """
    if output_path:
        output_path = output_path
    else:
        output_path = ''
    xyzAll = reconstruct_3d(coeffs, list_of_csv, output_path,plotting=plotting)
    return xyzAll


def find_cam_files(root_dir, extension, pathList):
    """
    Function to find cam files for 3-D calibration
    Parameters
    ----------
    root_dir : path directory
    extension : string of extension you want found eg .mp4 or .txt

    pathList : empty list
    Returns
    -------
    pathList : list containing lists of file names
    """
    for file in glob.glob(root_dir,extension,recursive=True):
        if file.find('cam1'): # check and make sure that the files have been split
            cam_path = file.rsplit('_')[0]
            cam2_path = cam_path + 'cam2.mp4'
            cam3_path = cam_path + 'cam3.mp4'
            pathList.append([file,cam2_path,cam3_path])
        else:
            print('Sorry, file not found :: file designation' + file)

    return pathList


def extract_3D_kinematics(filepath, dlt_coeffs, output_path):
    pl = []
    pts_dataframe = pd.DataFrame()
    cam_files = find_cam_files(filepath, '.mp4', pl)
    for i in cam_files:
        pts = reconstruct_points(i, dlt_coeffs, output_path)


    return pts_dataframe


def get_file_sets(glob_path,resnet_version,filtered=False):
    cam1_list=[]
    cam2_list=[]
    cam3_list=[]
    all_files = glob.glob(glob_path,recursive=True)
    all_files.extend(glob.glob(resnet_version))
    if filtered:
        all_files.extend(glob.glob(filtered))
    for file in all_files:
        if 'cam1' in file:
            # find rest of files containing exp names
            files=file.rsplit('/',1)[1]
            names = str(files.rsplit('_cam1',1)[0])
            file_list = [file_list for file_list in all_files if names in file_list]
            for s in file_list:
                if "cam1" in s:
                    cam1_list.append(s)
                if "cam2" in s:
                    cam2_list.append(s)
                if "cam3" in s:
                    cam3_list.append(s)
    return cam1_list,cam2_list,cam3_list


def get_file_sets(glob_path, resnet_version, filtered=False):
    cam1_list = []
    cam2_list = []
    cam3_list = []
    all_files = glob.glob(glob_path, recursive=True)
    all_files.extend(glob.glob(resnet_version))
    if filtered:
        all_files.extend(glob.glob(filtered))
    for file in all_files:
        if 'cam1' in file:
            # find rest of files containing exp names
            files = file.rsplit('/', 1)[1]
            names = str(files.rsplit('_cam1', 1)[0])
            file_list = [file_list for file_list in all_files if names in file_list]
            for s in file_list:
                if "cam1" in s:
                    cam1_list.append(s)
                if "cam2" in s:
                    cam2_list.append(s)
                if "cam3" in s:
                    cam3_list.append(s)
    return cam1_list, cam2_list, cam3_list


def get_kinematic_data(glob_path, dlt_path, resnet_version, save=True):
    cam1_list, cam2_list, cam3_list = get_file_sets(glob_path, resnet_version)
    for i, val in enumerate(cam1_list):
        cam_list = [cam1_list[i], cam2_list[i], cam3_list[i]]
        cu = filter_cam_lists(cam_list)
        if cu == 1:
            df = 0
            print('Error in the video extraction!! Make sure all your files are extracted.')
            break
        xyzAll, labels = reconstruct_3d(dlt_path, cam_list,'.csv')
        coords = ['X', 'Y', 'Z']
        scorer = ['Brett']
        header = pd.MultiIndex.from_product([scorer,labels,
                                             coords],
                                            names=['scorer','bodyparts', 'coords'])
        df = pd.DataFrame(xyzAll.reshape((xyzAll.shape[0], xyzAll.shape[1] * xyzAll.shape[2])), columns=header)
        if save:
            df.to_csv('/home/kallanved/Desktop/EX/savekinematics_df.csv')
    return df


def filter_cam_lists(cam_list):
    cu = 0
    c1 = cam_list[0]
    c2 = cam_list[1]
    c3 = cam_list[2]
    # compare lengths
    if len(c1) == len(c2) == len(c3):
        cu=0
    else:
        print('Video File(s) are not available')
        cu = 1
    return cu
