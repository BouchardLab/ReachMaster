import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from software.preprocessing.video_data.DLC.Reconstruction import dlt_reconstruct
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error
from scipy import signal
import shutil, os
from PIL import Image
from errno import EEXIST, ENOENT

def process_csv(prediction_csvs, label_csvs, windowlength=9, thres = 0.05):
    data_dict = {}
    cam1_csv = np.loadtxt(label_csvs[0], dtype=str, delimiter=",")
    names = cam1_csv[1,1::2].copy()

    cam1_dlc_data = np.loadtxt(prediction_csvs[0], skiprows=3, delimiter=",")[:, 1:]
    cam2_dlc_data = np.loadtxt(prediction_csvs[1], skiprows=3, delimiter=",")[:, 1:]
    cam3_dlc_data = np.loadtxt(prediction_csvs[2], skiprows=3, delimiter=",")[:, 1:]

    cam1_csv_data = np.genfromtxt(label_csvs[0], delimiter=",", skip_header=3, usecols=range(1, len(names)*2 + 1))
    cam2_csv_data = np.genfromtxt(label_csvs[1], delimiter=",", skip_header=3, usecols=range(1, len(names)*2 + 1))
    cam3_csv_data = np.genfromtxt(label_csvs[2], delimiter=",", skip_header=3, usecols=range(1, len(names)*2 + 1))
    start_idx = 0

    discarded = set()
    shortest_len = min(cam1_dlc_data.shape[0], cam2_dlc_data.shape[0], cam3_dlc_data.shape[0])
    for i in range(len(names)):
        labelled_idx = i * 2
        predicted_idx = i * 3
        entry_dict = {}

        cam1_labelled_x = cam1_csv_data[start_idx:shortest_len, labelled_idx].copy()
        cam1_predicted_x = cam1_dlc_data[start_idx:shortest_len, predicted_idx].copy()
        cam1_labelled_y = cam1_csv_data[start_idx:shortest_len, labelled_idx + 1].copy()
        cam1_predicted_y = cam1_dlc_data[start_idx:shortest_len, predicted_idx + 1].copy()
        cam1_predicted_prob = cam1_dlc_data[start_idx:shortest_len, predicted_idx + 2].copy()
        
        cam2_labelled_x = cam2_csv_data[start_idx:shortest_len, labelled_idx].copy()
        cam2_predicted_x = cam2_dlc_data[start_idx:shortest_len, predicted_idx].copy()
        cam2_labelled_y = cam2_csv_data[start_idx:shortest_len, labelled_idx + 1].copy()
        cam2_predicted_y = cam2_dlc_data[start_idx:shortest_len, predicted_idx + 1].copy()
        cam2_predicted_prob = cam2_dlc_data[start_idx:shortest_len, predicted_idx + 2].copy()

        cam3_labelled_x = cam3_csv_data[start_idx:shortest_len, labelled_idx].copy()
        cam3_predicted_x = cam3_dlc_data[start_idx:shortest_len, predicted_idx].copy()
        cam3_labelled_y = cam3_csv_data[start_idx:shortest_len, labelled_idx + 1].copy()
        cam3_predicted_y = cam3_dlc_data[start_idx:shortest_len, predicted_idx + 1].copy()
        cam3_predicted_prob = cam3_dlc_data[start_idx:shortest_len, predicted_idx + 2].copy()


        mask = np.isnan(cam1_labelled_x) | np.isnan(cam1_labelled_y) | np.isnan(cam2_labelled_x) | np.isnan(cam2_labelled_y) | np.isnan(cam3_labelled_x) | np.isnan(cam3_labelled_y)
        if np.sum(~mask) < 10:
            print("Not enough entries for {}, discarding".format(names[i]))
            discarded.add(names[i])
            continue
        cam1_labelled_x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam1_labelled_x[~mask])
        cam1_labelled_y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam1_labelled_y[~mask])
        cam2_labelled_x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam2_labelled_x[~mask])
        cam2_labelled_y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam2_labelled_y[~mask])
        cam3_labelled_x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam3_labelled_x[~mask])
        cam3_labelled_y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cam3_labelled_y[~mask])
        
        entry_dict["Camera 1"] = {}
        entry_dict["Camera 1"]["labelled_x"] = cam1_labelled_x
        entry_dict["Camera 1"]["predicted_x"] = cam1_predicted_x
        entry_dict["Camera 1"]["labelled_y"] = cam1_labelled_y
        entry_dict["Camera 1"]["predicted_y"] = cam1_predicted_y
        entry_dict["Camera 1"]["predicted_prob"] = cam1_predicted_prob
        entry_dict["Camera 1"]["max_x"] = max(np.max(entry_dict["Camera 1"]["labelled_x"]), np.max(entry_dict["Camera 1"]["predicted_x"]))
        entry_dict["Camera 1"]["max_y"] = max(np.max(entry_dict["Camera 1"]["labelled_y"]), np.max(entry_dict["Camera 1"]["predicted_y"]))
        entry_dict["Camera 1"]["min_x"] = min(np.min(entry_dict["Camera 1"]["labelled_x"]), np.min(entry_dict["Camera 1"]["predicted_x"]))
        entry_dict["Camera 1"]["min_y"] = min(np.min(entry_dict["Camera 1"]["labelled_y"]), np.min(entry_dict["Camera 1"]["predicted_y"]))

        entry_dict["Camera 2"] = {}
        entry_dict["Camera 2"]["labelled_x"] = cam2_labelled_x
        entry_dict["Camera 2"]["predicted_x"] = cam2_predicted_x
        entry_dict["Camera 2"]["labelled_y"] = cam2_labelled_y
        entry_dict["Camera 2"]["predicted_y"] = cam2_predicted_y
        entry_dict["Camera 2"]["predicted_prob"] = cam2_predicted_prob
        entry_dict["Camera 2"]["max_x"] = max(np.max(entry_dict["Camera 2"]["labelled_x"]), np.max(entry_dict["Camera 2"]["predicted_x"]))
        entry_dict["Camera 2"]["max_y"] = max(np.max(entry_dict["Camera 2"]["labelled_y"]), np.max(entry_dict["Camera 2"]["predicted_y"]))
        entry_dict["Camera 2"]["min_x"] = min(np.min(entry_dict["Camera 2"]["labelled_x"]), np.min(entry_dict["Camera 2"]["predicted_x"]))
        entry_dict["Camera 2"]["min_y"] = min(np.min(entry_dict["Camera 2"]["labelled_y"]), np.min(entry_dict["Camera 2"]["predicted_y"]))

        entry_dict["Camera 3"] = {}
        entry_dict["Camera 3"]["labelled_x"] = cam3_labelled_x
        entry_dict["Camera 3"]["predicted_x"] = cam3_predicted_x
        entry_dict["Camera 3"]["labelled_y"] = cam3_labelled_y
        entry_dict["Camera 3"]["predicted_y"] = cam3_predicted_y
        entry_dict["Camera 3"]["predicted_prob"] = cam3_predicted_prob
        entry_dict["Camera 3"]["max_x"] = max(np.max(entry_dict["Camera 3"]["labelled_x"]), np.max(entry_dict["Camera 3"]["predicted_x"]))
        entry_dict["Camera 3"]["max_y"] = max(np.max(entry_dict["Camera 3"]["labelled_y"]), np.max(entry_dict["Camera 3"]["predicted_y"]))
        entry_dict["Camera 3"]["min_x"] = min(np.min(entry_dict["Camera 3"]["labelled_x"]), np.min(entry_dict["Camera 3"]["predicted_x"]))
        entry_dict["Camera 3"]["min_y"] = min(np.min(entry_dict["Camera 3"]["labelled_y"]), np.min(entry_dict["Camera 3"]["predicted_y"]))
        data_dict[names[i]] = entry_dict
    filtered_names = [name for name in names if name not in discarded]
    data_dict["filtered_names"] = filtered_names
    
    for name in filtered_names:
        for cam in ["Camera 1", "Camera 2", "Camera 3"]:
            # Interpolation
            mask = data_dict[name][cam]["predicted_prob"] < thres
            data_dict[name][cam]["interp_predicted_prob"] = data_dict[name][cam]["predicted_prob"].copy()
            data_dict[name][cam]["interp_predicted_prob"][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_dict[name][cam]["predicted_prob"][~mask])
            data_dict[name][cam]["interp_predicted_x"] = data_dict[name][cam]["predicted_x"].copy()
            data_dict[name][cam]["interp_predicted_x"][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_dict[name][cam]["predicted_x"][~mask])
            data_dict[name][cam]["interp_predicted_y"] = data_dict[name][cam]["predicted_y"].copy()
            data_dict[name][cam]["interp_predicted_y"][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_dict[name][cam]["predicted_y"][~mask])

            # Smoothing
            win = np.hamming(windowlength)
            data_dict[name][cam]["filtered_predicted_prob"] = np.convolve(win/win.sum(), data_dict[name][cam]["interp_predicted_prob"], mode='same')
            data_dict[name][cam]["filtered_predicted_x"] = np.convolve(win/win.sum(), data_dict[name][cam]["interp_predicted_x"], mode='same')
            data_dict[name][cam]["filtered_predicted_y"] = np.convolve(win/win.sum(), data_dict[name][cam]["interp_predicted_y"], mode='same')

            data_dict[name][cam]["filtered_labelled_x"] = np.convolve(win/win.sum(), data_dict[name][cam]["labelled_x"], mode='same')
            data_dict[name][cam]["filtered_labelled_y"] = np.convolve(win/win.sum(), data_dict[name][cam]["labelled_y"], mode='same')
    return data_dict

def get_reconstructions(data_dict, dlt_coefs_file):
    dlt_coefs = np.loadtxt(dlt_coefs_file, delimiter=",")
    reconstructions = {}
    for name in data_dict["filtered_names"]:
        # read in data from DLC
        frames = data_dict[name]["Camera 1"]["labelled_x"].shape[0]
        cam_data = np.empty([frames, 6], dtype=float)
        weights = np.empty([frames, 6], dtype=float)

        cam_data[:, 0] = data_dict[name]["Camera 1"]["labelled_x"]
        cam_data[:, 1] = data_dict[name]["Camera 1"]["labelled_y"]
        cam_data[:, 2] = data_dict[name]["Camera 2"]["labelled_x"]
        cam_data[:, 3] = data_dict[name]["Camera 2"]["labelled_y"]
        cam_data[:, 4] = data_dict[name]["Camera 3"]["labelled_x"]
        cam_data[:, 5] = data_dict[name]["Camera 3"]["labelled_y"]

        xyz_labelled = dlt_reconstruct(dlt_coefs, cam_data)

        cam_data[:, 0] = data_dict[name]["Camera 1"]["filtered_labelled_x"]
        cam_data[:, 1] = data_dict[name]["Camera 1"]["filtered_labelled_y"]
        cam_data[:, 2] = data_dict[name]["Camera 2"]["filtered_labelled_x"]
        cam_data[:, 3] = data_dict[name]["Camera 2"]["filtered_labelled_y"]
        cam_data[:, 4] = data_dict[name]["Camera 3"]["filtered_labelled_x"]
        cam_data[:, 5] = data_dict[name]["Camera 3"]["filtered_labelled_y"]

        xyz_filtered_labelled = dlt_reconstruct(dlt_coefs, cam_data)

        cam_data[:, 0] = data_dict[name]["Camera 1"]["predicted_x"]
        cam_data[:, 1] = data_dict[name]["Camera 1"]["predicted_y"]
        cam_data[:, 2] = data_dict[name]["Camera 2"]["predicted_x"]
        cam_data[:, 3] = data_dict[name]["Camera 2"]["predicted_y"]
        cam_data[:, 4] = data_dict[name]["Camera 3"]["predicted_x"]
        cam_data[:, 5] = data_dict[name]["Camera 3"]["predicted_y"]
        
        weights[:, 0] = data_dict[name]["Camera 1"]["predicted_prob"]
        weights[:, 1] = data_dict[name]["Camera 1"]["predicted_prob"]
        weights[:, 2] = data_dict[name]["Camera 2"]["predicted_prob"]
        weights[:, 3] = data_dict[name]["Camera 2"]["predicted_prob"]
        weights[:, 4] = data_dict[name]["Camera 3"]["predicted_prob"]
        weights[:, 5] = data_dict[name]["Camera 3"]["predicted_prob"]

        xyz_predicted_weighted = dlt_reconstruct(dlt_coefs, cam_data, weights)
        xyz_predicted_unweighted = dlt_reconstruct(dlt_coefs, cam_data)

        cam_data[:, 0] = data_dict[name]["Camera 1"]["interp_predicted_x"]
        cam_data[:, 1] = data_dict[name]["Camera 1"]["interp_predicted_y"]
        cam_data[:, 2] = data_dict[name]["Camera 2"]["interp_predicted_x"]
        cam_data[:, 3] = data_dict[name]["Camera 2"]["interp_predicted_y"]
        cam_data[:, 4] = data_dict[name]["Camera 3"]["interp_predicted_x"]
        cam_data[:, 5] = data_dict[name]["Camera 3"]["interp_predicted_y"]
        
        weights[:, 0] = data_dict[name]["Camera 1"]["interp_predicted_prob"]
        weights[:, 1] = data_dict[name]["Camera 1"]["interp_predicted_prob"]
        weights[:, 2] = data_dict[name]["Camera 2"]["interp_predicted_prob"]
        weights[:, 3] = data_dict[name]["Camera 2"]["interp_predicted_prob"]
        weights[:, 4] = data_dict[name]["Camera 3"]["interp_predicted_prob"]
        weights[:, 5] = data_dict[name]["Camera 3"]["interp_predicted_prob"]

        xyz_interp_predicted_weighted = dlt_reconstruct(dlt_coefs, cam_data, weights)
        xyz_interp_predicted_unweighted = dlt_reconstruct(dlt_coefs, cam_data)

        cam_data[:, 0] = data_dict[name]["Camera 1"]["filtered_predicted_x"]
        cam_data[:, 1] = data_dict[name]["Camera 1"]["filtered_predicted_y"]
        cam_data[:, 2] = data_dict[name]["Camera 2"]["filtered_predicted_x"]
        cam_data[:, 3] = data_dict[name]["Camera 2"]["filtered_predicted_y"]
        cam_data[:, 4] = data_dict[name]["Camera 3"]["filtered_predicted_x"]
        cam_data[:, 5] = data_dict[name]["Camera 3"]["filtered_predicted_y"]
        
        weights[:, 0] = data_dict[name]["Camera 1"]["filtered_predicted_prob"]
        weights[:, 1] = data_dict[name]["Camera 1"]["filtered_predicted_prob"]
        weights[:, 2] = data_dict[name]["Camera 2"]["filtered_predicted_prob"]
        weights[:, 3] = data_dict[name]["Camera 2"]["filtered_predicted_prob"]
        weights[:, 4] = data_dict[name]["Camera 3"]["filtered_predicted_prob"]
        weights[:, 5] = data_dict[name]["Camera 3"]["filtered_predicted_prob"]

        xyz_filtered_predicted_weighted = dlt_reconstruct(dlt_coefs, cam_data, weights)
        xyz_filtered_predicted_unweighted = dlt_reconstruct(dlt_coefs, cam_data)

        reconstructions[name] = {}
        reconstructions[name]["xyz_labelled"] = xyz_labelled
        reconstructions[name]["xyz_filtered_labelled"] = xyz_filtered_labelled
        reconstructions[name]["xyz_predicted_weighted"] = xyz_predicted_weighted
        reconstructions[name]["xyz_predicted_unweighted"] = xyz_predicted_unweighted
        reconstructions[name]["xyz_interp_predicted_weighted"] = xyz_interp_predicted_weighted
        reconstructions[name]["xyz_interp_predicted_unweighted"] = xyz_interp_predicted_unweighted
        reconstructions[name]["xyz_filtered_predicted_weighted"] = xyz_filtered_predicted_weighted
        reconstructions[name]["xyz_filtered_predicted_unweighted"] = xyz_filtered_predicted_unweighted
        reconstructions[name]["xyz_predicted_prob"] = np.mean(np.array([data_dict[name]["Camera 1"]["predicted_prob"], data_dict[name]["Camera 2"]["predicted_prob"], data_dict[name]["Camera 3"]["predicted_prob"]]), axis=0)
        reconstructions[name]["xyz_interp_predicted_prob"] = np.mean(np.array([data_dict[name]["Camera 1"]["interp_predicted_prob"], data_dict[name]["Camera 2"]["interp_predicted_prob"], data_dict[name]["Camera 3"]["interp_predicted_prob"]]), axis=0)
        reconstructions[name]["xyz_filtered_predicted_prob"] = np.mean(np.array([data_dict[name]["Camera 1"]["filtered_predicted_prob"], data_dict[name]["Camera 2"]["filtered_predicted_prob"], data_dict[name]["Camera 3"]["filtered_predicted_prob"]]), axis=0)
    return reconstructions

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        os.makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else: raise

def rm_dir(mypath):
    '''Deletes a directory. equivalent to using rm -rf on the command line'''
    try:
        shutil.rmtree(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == ENOENT:
            pass
        else: raise
