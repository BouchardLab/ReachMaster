"""
Functions intended to import and transform behavioral reaching data from ReachMaster experiments into 3-D kinematics
Use with DLT co-effecients obtained through easyWand or other procedures + multi-camera predictions from DLC
Author: Brett Nelson, NSDS Lab 2020

"""
import matplotlib.pyplot as plt
# imports
import numpy as np


# dlt reconstruct adapted by An Chi Chen from DLTdv5 by Tyson Hedrick
def dlt_reconstruct(c, camPts):
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


def reconstruct_3d(dlt_coefs_file, dlc_files, output_format, plotting=True):
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

    xyz_all = np.empty([frames, 3, len(names)])

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
        csv_index = int((k * 3) + 1)
        for cam in range(cameras):
            col = cam * 2
            cam_data[:, col] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3, usecols=csv_index)
            cam_data[:, col + 1] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3,
                                              usecols=(csv_index + 1))

        # combine
        xyz = dlt_reconstruct(dlt_coefs, cam_data)
        xyz_all[:, :, k] = np.copy(xyz)
        if plotting:
            xs = np.copy(xyz[:, 0])
            ys = np.copy(xyz[:, 1])
            zs = np.copy(xyz[:, 2])
            ax.scatter(xs, ys, zs)

        # save xyz coords in csv file
        out_filename = (output_format + names[k] + '.csv')
        np.savetxt(out_filename, xyz, delimiter=',', fmt='%1.4f')

    if plotting:
        plt.legend(names)
        plt.show()

    return xyz_all


def reconstruct_points(list_of_csv, coeffs, output_path=False, plotting=False):
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
    xyzAll = reconstruct_3d(coeffs, list_of_csv, output_path, plotting=plotting)
    return xyzAll
