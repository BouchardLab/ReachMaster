"""Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 5/18/2022
    Library intended to perform initial analysis on pre-processed 3-D kinematic predictions and experimental
    sensor data from the ReachMaster system. Data is preprocessed using the /ReachPredict3D library. Recording
    blocks are divided into coarse trials, then properly classified, segmented, and visualized. """
import DataStream_Vis_Utils as utils
import viz_utils as vu
from Trial_Classifier import Trial_Classify as Classifier
from moviepy.editor import *
import skvideo
import cv2
import imageio
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import pdb


# set ffmpeg path manually (if necessary)
ffm_path = 'C:/Users/bassp/OneDrive/Desktop/ffmpeg/bin/'
skvideo.setFFmpegPath(ffm_path)
import skvideo.io
# Global variables
trial_classifier_path = 'models/null_model.joblib'
hand_classifier_path = 'models/numReach_model.joblib'
num_reach_classifier_path = 'models/whichHand_model.joblib'

# Public functions
def find_linear_transformation_between_DLC_and_robot(handle_vector, robot_vector):
    # Find each dimensions transform
    T = np.zeros((2, 3))
    x_int = 0.15
    y_int = 0.15
    z_int = 0.4
    T[1, 0] = x_int
    T[1, 1] = y_int
    T[1, 2] = z_int
    for i in range(0, handle_vector.shape[1]):
        M = np.linalg.lstsq(handle_vector[:, i].reshape(-1, 1), robot_vector[:, i].reshape(-1, 1))[0]
        T[0, i] = M
    return T


def apply_linear_transformation_to_DLC_data(input_data, transformation_vector):
    new_data = np.zeros(input_data.shape)
    for il in range(0, input_data.shape[1]):
        dd = np.array([np.dot(i, transformation_vector[0, il]) for i in input_data[:, il]])
        new_data[:, il] = np.squeeze(dd)
    return new_data


def plot_session_alignment(input_data, robot_data, transformed_data, dim='x'):
    length = np.linspace(0, input_data.shape[0], input_data.shape[0])
    plt.scatter(length, input_data, label='handle_' + dim)
    plt.scatter(length, robot_data, label='robot_' + dim)
    plt.scatter(length, transformed_data, label='Xformed_' + dim)
    plt.legend()
    plt.show()


def get_principle_components(positions, vel=None, acc=None, num_pcs=10):
    pca = PCA(n_components=num_pcs, whiten=True, svd_solver='full')
    if vel:
        if acc:
            pos = np.asarray(positions)
            vel = np.asarray(vel)
            acc = np.asarray(acc)
            pva = np.hstack((pos.reshape(pos.shape[1], pos.shape[0] * pos.shape[2]),
                             vel.reshape(vel.shape[1], vel.shape[0] * vel.shape[2]),
                             acc.reshape(acc.shape[1], acc.shape[0] * acc.shape[2])))
            pc_vector = pca.fit_transform(pva)
        else:
            pos = np.asarray(positions)
            vel = np.asarray(vel)
            pv = np.hstack((pos.reshape(pos.shape[1], pos.shape[0] * pos.shape[2]),
                            vel.reshape(vel.shape[1], vel.shape[0] * vel.shape[2]
                                        )))
            pc_vector = pca.fit_transform(pv)
    else:
        if acc:
            pos = np.asarray(positions)
            acc = np.asarray(acc)
            pa = np.hstack((pos.reshape(pos.shape[1], pos.shape[0] * pos.shape[2]),
                            acc.reshape(acc.shape[1], acc.shape[0] * acc.shape[2])))
            pc_vector = pca.fit_transform(pa)
        else:
            pos = np.asarray(positions)
            pc_vector = pca.fit_transform(pos.reshape(pos.shape[1], pos.shape[0] * pos.shape[2]))
    #explained_variance_ratio = pca.explained_variance_ratio_
    return pc_vector


def gkern(input_vector, sig=1.0):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`. Filters N-D vector.
    """
    resulting_vector = gaussian_filter1d(input_vector, sig, mode='mirror')
    return resulting_vector


class ReachViz:
    def __init__(self, date, session, data_path, block_vid_file, kin_path, rat):
        self.tug_flag = None
        self.endpoint_error, self.x_endpoint_error, self.y_endpoint_error, self.z_endpoint_error = 0, 0, 0, 0
        self.preprocessed_rmse, self.outlier_list, self.transformation_matrix = [], [], []
        self.probabilities, self.bi_reach_vector, self.trial_index, self.first_lick_signal, self.outlier_indexes = \
            [], [], [], [], []
        self.pos_holder, self.acc_holder, self.vel_holder, self.speed_holder = [], [], [], []
        self.reach_hand_type, self.num_reaches_split = [], None
        self.rat = rat
        self.date = date
        self.session = session
        self.kinematic_data_path = kin_path
        self.block_exp_df, self.d, self.kinematic_block, self.velocities, self.speeds, self.dim, self.save_dict = \
            [], [], [], [], [], [], []
        self.data_path = data_path
        self.sensors, self.gen_p_thresh = 0, 0.5
        self.load_data()  # get exp/kin dataframes
        self.reaching_dataframe = pd.DataFrame()
        self.trial_start_vectors = 0
        self.trial_stop_vectors = 0
        self.predicted_trial_type, self.trial_type, self.trial_type_outlier = False, False, False
        self.arm_id_list = []
        self.predicted_num_reaches, self.handle_sensor_speed = None, []
        self.pos_pc, self.pos_v_pc, self.pos_v_a_pc, self.freq_decomp_pos = [], [], [], []
        self.sstr, self.lick_index = 0, []
        self.rat_gaps, self.total_ints, self.interpolation_rmse, self.outlier_rmse, self.valid_rmse = [], [], [], [], []
        self.block_video_path = block_vid_file
        # Obtain Experimental Block of Data
        self.get_block_data()
        # Find "reaching" peaks and tentative start times agnostic of trial time
        # self.get_reaches_from_block()
        # pdb.set_trace()
        # Get Start/Stop of Trials
        self.get_starts_stops()
        # Initialize sensor variables
        self.exp_response_sensor, self.trial_sensors, self.h_moving_sensor, self.reward_zone_sensor, self.lick, \
        self.trial_num = 0, 0, 0, 0, 0, 0
        self.time_vector, self.images, self.bout_vector = [], [], []
        self.trial_rewarded, self.single_hand  = False, None
        self.filename, self.csv_writer = None, None
        self.total_outliers, self.rewarded, self.left_palm_f_x, self.right_palm_f_x = [], [], [], []
        self.total_raw_speeds, self.total_preprocessed_speeds, self.total_probabilities = [], [], []
        self.behavior_start_time, self.prediction_information = [] , {}
        self.reach_peak_time, self.right_palm_maxima = [], []
        self.left_start_times = []
        self.left_peak_times, self.left_palm_maxima = [], []
        self.right_reach_end_time, self.right_reach_end_times = [], []
        self.left_reach_end_time, self.left_reach_end_times = [], []
        self.right_start_times, self.left_start_times = [], []
        self.left_hand_speeds, self.right_hand_speeds, self.total_speeds, self.left_hand_raw_speeds, \
        self.right_hand_raw_speeds = [], [], [], [], []
        self.right_peak_times = []
        self.bimanual_reach_times = []
        self.k_length = 0
        self.speed_holder, self.raw_speeds = [], []
        self.left_arm_pc_pos, self.left_arm_pc_pos_v, self.left_arm_pc_pos_v_a, self.right_arm_pc_pos, \
        self.right_arm_pc_pos_v, self.right_arm_pc_pos_v_a = [], [], [], [], [], []
        self.uninterpolated_left_palm_v, self.uninterpolated_right_palm_v = [], []
        # Initialize kinematic variables
        self.left_palm_velocity, self.right_palm_velocity, self.lag, self.clip_path, self.lick_vector, self.reach_vector, \
        self.prob_index, self.pos_index, self.seg_num, self.prob_nose, self.right_wrist_velocity, self.left_wrist_velocity \
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []
        self.nose_p, self.handle_p, self.left_shoulder_p, self.left_forearm_p, self.left_wrist_p = [], [], [], [], []
        self.left_palm_p, self.left_index_base_p = [], []
        self.left_index_tip_p, self.left_middle_base_p, self.left_middle_tip_p, self.left_third_base_p, \
        self.left_third_tip_p, self.left_end_base_p, self.left_end_tip_p, self.right_shoulder_p, self.right_forearm_p, \
        self.right_wrist_p, self.right_palm_p = [], [], [], [], [], [], [], [], [], [], []
        self.right_index_base_p, self.right_index_tip_p, self.right_middle_base_p, self.right_middle_tip_p = [], [], [], []
        self.right_third_base_p, self.right_third_tip_p, self.right_end_base_p, self.right_end_tip_p = [], [], [], []
        self.fps = 20
        # Kinematic variable initialization
        self.nose_v, self.handle_v, self.left_shoulder_v, self.left_forearm_v, self.left_wrist_v, self.left_palm_v, self.left_index_base_v, \
        self.left_index_tip_v, self.left_middle_base_v, self.left_middle_tip_v, self.left_third_base_v, self.left_third_tip_v, self.left_end_base_v, \
        self.left_end_tip_v, self.right_shoulder_v, self.right_forearm_v, self.right_wrist_v, self.right_palm_v, self.right_index_base_v, \
        self.right_index_tip_v, self.right_middle_base_v, self.right_middle_tip_v, self.right_third_base_v, self.right_third_tip_v, \
        self.right_end_base_v, self.right_end_tip_v = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                                                      [], [], [], [], [], [], [], [], [], [], []
        self.nose_s, self.handle_s, self.left_shoulder_s, self.left_forearm_s, self.left_wrist_s, self.left_palm_s, self.left_index_base_s, \
        self.left_index_tip_s, self.left_middle_base_s, self.left_middle_tip_s, self.left_third_base_s, self.left_third_tip_s, self.left_end_base_s, \
        self.left_end_tip_s, self.right_shoulder_s, self.right_forearm_s, self.right_wrist_s, self.right_palm_s, self.right_index_base_s, \
        self.right_index_tip_s, self.right_middle_base_s, self.right_middle_tip_s, self.right_third_base_s, self.right_third_tip_s, \
        self.right_end_base_s, self.right_end_tip_s = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                                                      [], [], [], [], [], [], [], [], []
        self.nose_a, self.handle_a, self.left_shoulder_a, self.left_forearm_a, self.left_wrist_a, self.left_palm_a, self.left_index_base_a, \
        self.left_index_tip_a, self.left_middle_base_a, self.left_middle_tip_a, self.left_third_base_a, self.left_third_tip_a, self.left_end_base_a, \
        self.left_end_tip_a, self.right_shoulder_a, self.right_forearm_a, self.right_wrist_a, self.right_palm_a, self.right_index_base_a, \
        self.right_index_tip_a, self.right_middle_base_a, self.right_middle_tip_a, self.right_third_base_a, self.right_third_tip_a, \
        self.right_end_base_a, self.right_end_tip_a = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                                                      [], [], [], [], [], [], [], [], []
        self.nose_o, self.handle_o, self.left_shoulder_o, self.left_forearm_o, self.left_wrist_o, self.left_palm_o, self.left_index_base_o, \
        self.left_index_tip_o, self.left_middle_base_o, self.left_middle_tip_o, self.left_third_base_o, self.left_third_tip_o, self.left_end_base_o, \
        self.left_end_tip_o, self.right_shoulder_o, self.right_forearm_o, self.right_wrist_o, self.right_palm_o, self.right_index_base_o, \
        self.right_index_tip_o, self.right_middle_base_o, self.right_middle_tip_o, self.right_third_base_o, self.right_third_tip_o, \
        self.right_end_base_o, self.right_end_tip_o = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
                                                      [], [], [], [], [], [], [], [], []
        # Initialize kinematic, positional and probability-indexed variables
        self.prob_filter_index, self.left_arm_filter_index, self.right_arm_filter_index = [], [], []
        self.right_index_base, self.right_index_tip, self.right_middle_base, self.right_middle_tip, \
        self.right_third_base, self.right_third_tip, self.right_end_base, self.right_end_tip = [], [], [], [], [], [], [], []
        self.left_index_base, self.left_index_tip, self.left_middle_base, self.left_middle_tip, self.left_third_base, \
        self.left_third_tip, self.left_end_base, self.left_end_tip = [], [], [], [], [], [], [], []
        self.start_trial_indice, self.trial_cut_vector, self.block_cut_vector, self.handle_velocity, \
        self.bout_reach = [], [], [], [], []
        self.handle_moved, self.gif_save_path, self.prob_right_index, self.prob_left_index, self.l_pos_index, \
        self.r_pos_index = 0, 0, 0, 0, 0, 0
        self.x_robot, self.y_robot, self.z_robot, self.uninterpolated_right_palm, \
        self.uninterpolated_left_palm = [], [], [], [], []
        self.prob_left_digit, self.prob_right_digit, self.left_digit_filter_index, \
        self.right_digit_filter_index = [], [], [], []
        self.reaching_mask, self.right_arm_speed, self.left_arm_speed, self.reprojections, \
        self.interpolation_gaps, self.total_reach_vector = [], [], [], [], [], []
        self.left_palm_speed, self.right_palm_speed, self.handle_speed, self.right_arm_velocity, \
        self.left_arm_velocity = [], [], [], [], []
        self.bout_flag, self.positions, self.sensor_data_list = False, [], []
        self.nose, self.handle, self.body_prob, self.central_body_mass, self.prob_left_shoulder, self.prob_right_shoulder = [
            0, 0, 0, 0, 0, 0]
        self.left_shoulder, self.right_forearm, self.left_forearm, self.right_wrist, self.left_wrist, self.right_palm, self.left_palm = [
            0, 0, 0, 0, 0, 0, 0]
        self.prob_nose, self.prob_right_arm, self.prob_left_arm, self.right_digits, self.left_digits, self.right_shoulder = [
            0, 0, 0, 0, 0, 0]
        self.robot_handle_speed, self.reprojected_handle, self.reprojected_nose, self.reprojected_bhandle, self.reprojected_left_palm = [
            0, 0, 0, 0, 0]
        self.reprojected_right_palm, self.reprojected_left_wrist, self.reprojected_right_wrist, self.reprojected_left_shoulder, self.reprojected_right_shoulder = [
            0, 0, 0, 0, 0]
        self.prob_right_index, self.prob_left_index, self.bi_pos_index, self.r_reach_vector, self.l_reach_vector, \
        self.left_prob_index, self.right_prob_index = [0, 0, 0, 0, 0, 0, 0]
        self.left_PCS, self.right_PCS, self.total_PCS = None, None, None
        self.prob_list, self.pos_list, self.interpolation_rmse_list, self.valid_rmse_list, \
        self.outlier_rmse_list = [], [], [], [], []
        self.predictions_header = ['Trial Number', 'Null Prediction', 'Num Reaches Prediction', 'Handedness Prediction',
                                   'Null Outlier Flag', 'Num Reaches Outlier Flag', 'Handedness Prediction Outlier Flag',
                                   'Segmented Reaches', 'Segmented Null', 'Segmented Num', 'Segmented Hand']
        #self.prediction_information = dict.fromkeys(self.predictions_header, [0])
        self.total_block_reaches, self.individual_reach_peak_times, self.retract_start, self.retract_end = 0, [], 0, 0
        self.behavior_duration, self.behavior_end_time, self.total_block_reaches, self.sweep_total_reaches = [], [], 0, 0
        self.block_outlier_flag, self.prediction_information_list, self.speed_robot, self.reward_zone = False, [], [], False
        return

    def load_data(self):
        """ Function to load per-rat database into ReachLoader. """
        df = vu.import_robot_data(self.data_path)
        self.sensors = df.reset_index(drop=True)
        with (open(self.kinematic_data_path, "rb")) as openfile:
            self.d = pickle.load(openfile)
        return

    def make_paths(self):
        """ Function to construct a structured directory to save visual results. """
        vu.mkdir_p(self.sstr)
        vu.mkdir_p(self.sstr + '/data')
        vu.mkdir_p(self.sstr + '/videos')
        vu.mkdir_p(self.sstr + '/videos/reaches')
        vu.mkdir_p(self.sstr + '/plots')
        vu.mkdir_p(self.sstr + '/plots/reaches')
        vu.mkdir_p(self.sstr + '/timeseries')
        vu.mkdir_p(self.sstr + '/timeseries_analysis_plots')
        vu.mkdir_p(self.sstr + '/timeseries_analysis_plots/reaches')
        return

    def create_csv_block_predictions(self):
        """ Function to create csv writer object that writes rows of data """
        csv_path = str(self.rat) + str(self.date) + str(self.session) + 'predictions.csv'
        idf = pd.DataFrame(self.prediction_information_list)
        #with open(csv_path, 'a', newline='') as f:
        #    csv_writer = csv.DictWriter(f, fieldnames=self.prediction_information_list[0].keys())
        #    csv_writer.writeheader()
        #    for dicts in self.prediction_information_list:
        #        csv_writer.writerow(dicts)
        #f.close()
        idf.to_csv(csv_path)
        pdb.set_trace()
    def get_block_data(self):
        """ Function to fetch block positional and sensor data from rat database. """
        for kin_items in self.d:
            try:
                sess = kin_items.columns.levels[1]
                date = kin_items.columns.levels[2]
                self.dim = kin_items.columns.levels[3]
            except:  # fetched a null dataframe (0 entry in list), avoid..
                pass
            if sess[0] in self.session:
                if '_' in date[0][-1]:
                    if date[0][-3:-1] in self.date:
                        print('Hooked block positions for date  ' + date[0] + '     and session  ' + sess[0])
                        self.kinematic_block = kin_items
                else:
                    if date[0][-2:] in self.date:
                        print('Hooked block positions for date  ' + date[0] + '     and session  ' + sess[0])
                        self.kinematic_block = kin_items
        self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        return

    def threshold_data_with_probabilities(self, p_vector, p_thresh):
        """ Function to threshold input position vectors by the probability of this position being present. The mean
            over multiple cameras is used to better estimate error.
        """
        low_p_idx = np.where(p_vector < p_thresh)  # Filter positions by ind p values
        return np.asarray(low_p_idx)

    def get_starts_stops(self):
        """ Obtain the start and stop times of coarse behavior from the sensor block. """
        self.trial_start_vectors = self.block_exp_df['r_start'].values[0]
        self.trial_stop_vectors = self.block_exp_df['r_stop'].values[0]
        print('Number of Trials: ' + str(len(self.trial_start_vectors)))
        return

    def extract_sensor_data(self, idxstrt, idxstp, check_lick=True, filter=False):
        """ Function to extract probability thresholded sensor data from ReachMaster. Data has the option for it to be
         coarsely filtered.
        """
        self.k_length = self.kinematic_block[self.kinematic_block.columns[3:6]].values.shape[0]
        self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        self.h_moving_sensor = np.asarray(np.copy(self.block_exp_df['moving'].values[0][idxstrt:idxstp]))
        self.lick_index = np.asarray(np.copy(self.block_exp_df['lick'].values[0]))  # Lick DIO sensor
        self.reward_zone_sensor = np.asarray(np.copy(self.block_exp_df['RW'].values[0][idxstrt:idxstp]))
        self.time_vector = np.asarray(self.block_exp_df['time'].values[0][
                                      idxstrt:idxstp])  # extract trial timestamps from SpikeGadgets
        self.exp_response_sensor = np.asarray(self.block_exp_df['exp_response'].values[0][idxstrt:idxstp])
        # Re-Sample Analog Potentiometer signals to video data sample rate
        robot_length = self.block_exp_df['x_pot'].values[0][::4].shape[0]
        x_pot = self.block_exp_df['x_pot'].values[0][::4]  # sample rate at 4x camera
        y_pot = self.block_exp_df['y_pot'].values[0][::4]
        z_pot = self.block_exp_df['z_pot'].values[0][::4]

        x_pot = x_pot[robot_length - self.k_length:robot_length]
        y_pot = y_pot[robot_length - self.k_length:robot_length]
        z_pot = z_pot[robot_length - self.k_length:robot_length]

        r, theta, phi, self.x_robot, self.y_robot, self.z_robot = utils.forward_xform_coords(
            x_pot[idxstrt:idxstp], y_pot[idxstrt:idxstp], z_pot[idxstrt:idxstp])
        self.x_robot = gkern(self.x_robot, 11)
        self.y_robot = gkern(self.y_robot, 11)
        self.z_robot = gkern(self.z_robot, 11)
        if check_lick:
            self.check_licking_times_and_make_vector()
        self.sensor_data_list = [self.h_moving_sensor, self.lick_vector, self.reward_zone_sensor,
                                 self.exp_response_sensor,
                                 self.x_robot, self.y_robot, self.z_robot]
        return

    def check_licking_times_and_make_vector(self):
        """ Function to check if a lick occurs during the specified experimental time interval. """
        self.lick_vector = np.zeros((len(self.time_vector)))
        self.lick = list(np.around(np.array(self.lick_index), 2))
        self.time_vector = list(np.around(np.array(self.time_vector), 2))
        if self.lick:
            self.rewarded = True
            for lx in self.lick:
                if lx >= self.time_vector[0]:  # Is this lick happening before or at the first moment of reaching?
                    for trisx, t in enumerate(self.time_vector):
                        if t in self.lick:  # If this time index is in the lick vector
                            self.lick_vector[trisx] = 1  # Mask Array for licking
        self.lick = np.asarray(self.lick)
        # Find first lick signal
        try:
            self.first_lick_signal = np.where(self.lick == 1)[0][0]  # Take first signal
        except:
            self.first_lick_signal = False
        return

    def calculate_number_of_speed_peaks_in_block(self):
        left_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:21]].values[0:-1, :])
        right_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[0:-1, :])
        self.time_vector = list(right_palm[0, :])
        left_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values[0:-1, :],
                              axis=1)
        right_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[0:-1, :],
                               axis=1)
        left_palm_f = vu.cubic_spline_smoothing(np.copy(left_palm), spline_coeff=0.1)
        right_palm_f = vu.cubic_spline_smoothing(np.copy(right_palm), spline_coeff=0.1)
        left_palm_v, left_palm_a, left_palm_s = self.calculate_kinematics_from_position(left_palm_f)
        right_palm_v, right_palm_a, right_palm_s = self.calculate_kinematics_from_position(right_palm_f)
        # If palms are < 0.8 p-value, remove chance at "maxima
        left_palm_prob = np.where(left_palm_p < 0.3)[0]
        right_palm_prob = np.where(right_palm_p < 0.3)[0]
        # If palms are > 0.21m in the x-direction towards the handle 0 position.
        left_palm_s[left_palm_prob] = 0
        right_palm_s[right_palm_prob] = 0
        left_palm_pos_f = np.where(left_palm_f[:, 0] < .14)[0]
        right_palm_pos_f = np.where(left_palm_f[:, 0] < .14)[0]
        left_palm_s[left_palm_pos_f] = 0
        right_palm_s[right_palm_pos_f] = 0
        right_palm_maxima = find_peaks(right_palm_s, height=0.25, distance=30)[0]
        left_palm_maxima = find_peaks(left_palm_s, height=0.25, distance=30)[0]
        num_peaks = len(right_palm_maxima) + len(left_palm_maxima)
        print("Number of tentative reaching actions detected:  " + str(num_peaks))
        self.sweep_total_reaches = num_peaks
        return num_peaks

    def segment_and_filter_kinematic_block(self, cl1, cl2, p_thresh=0.5, coarse_threshold=0.3,
                                           preprocess=True):
        """ Function to segment, filter, and interpolate positional data across all bodyparts,
         using start and stop indices across
            whole-trial kinematics. """
        self.k_length = self.kinematic_block[self.kinematic_block.columns[3:6]].values.shape[0]
        self.prob_nose = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values[cl1:cl2, :], axis=1))
        w = 81
        # Threshold data if uncertainties in position > threshold
        self.prob_filter_index = np.where(self.prob_nose < coarse_threshold)[0]
        # Body parts, XYZ, used in ReachViz
        nose = vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[6:9]].values[cl1:cl2, :])
        handle = np.mean(
            [vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[0:3]].values[cl1:cl2, :]),
             vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[3:6]].values[cl1:cl2, :])], axis=0)
        left_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[9:12]].values[cl1:cl2, :])  # 21 end
        right_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])  # 57 end
        left_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        right_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])  # 57 end
        left_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[15:18]].values[cl1:cl2, :], )  # 21 end
        right_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])  # 57 end
        left_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:21]].values[cl1:cl2, :])
        right_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        # Digits, optional for now
        right_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[27:30]].values[cl1:cl2, :])
        right_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[30:33]].values[cl1:cl2, :])
        right_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[36:39]].values[cl1:cl2, :])
        right_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[39:42]].values[cl1:cl2, :])
        right_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[42:45]].values[cl1:cl2, :])
        right_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])
        right_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])
        right_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])
        left_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        left_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[57:60]].values[cl1:cl2, :])
        left_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[60:63]].values[cl1:cl2, :])
        left_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[66:69]].values[cl1:cl2, :])
        left_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[69:72]].values[cl1:cl2, :])
        left_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[72:75]].values[cl1:cl2, :])
        left_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[75:78]].values[cl1:cl2, :])
        left_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[78:81]].values[cl1:cl2, :])
        # Probabilities
        nose_p = np.mean(self.kinematic_block[self.kinematic_block.columns[6 + w:9 + w]].values[cl1:cl2, :], axis=1)
        handle_p = np.mean(self.kinematic_block[self.kinematic_block.columns[3 + w:6 + w]].values[cl1:cl2, :], axis=1)
        left_shoulder_p = np.mean(self.kinematic_block[self.kinematic_block.columns[9 + w:12 + w]].values[cl1:cl2,
                                  :], axis=1)  # 21 end
        right_shoulder_p = np.mean(self.kinematic_block[self.kinematic_block.columns[45 + w:48 + w]].values[cl1:cl2,
                                   :], axis=1)  # 57 end
        left_forearm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[12 + w:15 + w]].values[cl1:cl2,
                                 :], axis=1)  # 21 end
        right_forearm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[48 + w:51 + w]].values[cl1:cl2,
                                  :], axis=1)  # 57 end
        left_wrist_p = np.mean(self.kinematic_block[self.kinematic_block.columns[15 + w:18 + w]].values[cl1:cl2,
                               :], axis=1)  # 21 end
        right_wrist_p = np.mean(self.kinematic_block[self.kinematic_block.columns[51 + w:54 + w]].values[cl1:cl2,
                                :], axis=1)  # 57 end
        left_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[18 + w:21 + w]].values[cl1:cl2, :],
                              axis=1)
        right_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[54 + w:57 + w]].values[cl1:cl2, :],
                               axis=1)
        right_index_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[27 + w:30 + w]].values[cl1:cl2, :], axis=1)
        right_index_tip_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[30 + w:33 + w]].values[cl1:cl2, :], axis=1)
        right_middle_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[36 + w:39 + w]].values[cl1:cl2, :], axis=1)
        right_middle_tip_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[39 + w:42 + w]].values[cl1:cl2, :], axis=1)
        right_third_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[42 + w:45 + w]].values[cl1:cl2, :], axis=1)
        right_third_tip_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[45 + w:48 + w]].values[cl1:cl2, :], axis=1)
        right_end_base_p = np.mean(self.kinematic_block[self.kinematic_block.columns[48 + w:51 + w]].values[cl1:cl2, :],
                                   axis=1)
        right_end_tip_p = np.mean(self.kinematic_block[self.kinematic_block.columns[51 + w:54 + w]].values[cl1:cl2, :],
                                  axis=1)
        left_index_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[54 + w:57 + w]].values[cl1:cl2, :], axis=1)
        left_index_tip_p = np.mean(self.kinematic_block[self.kinematic_block.columns[57 + w:60 + w]].values[cl1:cl2, :],
                                   axis=1)
        left_middle_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[60 + w:63 + w]].values[cl1:cl2, :], axis=1)
        left_middle_tip_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[66 + w:69 + w]].values[cl1:cl2, :], axis=1)
        left_third_base_p = np.mean(
            self.kinematic_block[self.kinematic_block.columns[69 + w:72 + w]].values[cl1:cl2, :], axis=1)
        left_third_tip_p = np.mean(self.kinematic_block[self.kinematic_block.columns[72 + w:75 + w]].values[cl1:cl2, :],
                                   axis=1)
        left_end_base_p = np.mean(self.kinematic_block[self.kinematic_block.columns[75 + w:78 + w]].values[cl1:cl2, :],
                                  axis=1)
        left_end_tip_p = np.mean(self.kinematic_block[self.kinematic_block.columns[78 + w:81 + w]].values[cl1:cl2, :],
                                 axis=1)
        self.extract_sensor_data(cl1, cl2, check_lick=False)  # Get time vectors for calculating kinematics.
        self.positions = [nose, handle, left_shoulder, left_forearm, left_wrist,
                          left_palm, left_index_base,
                          left_index_tip, left_middle_base, left_middle_tip, left_third_base,
                          left_third_tip,
                          left_end_base, left_end_tip, right_shoulder, right_forearm,
                          right_wrist, right_palm,
                          right_index_base, right_index_tip, right_middle_base, right_middle_tip,
                          right_third_base, right_third_tip, right_end_base, right_end_tip]
        self.probabilities = [nose_p, handle_p, left_shoulder_p, left_forearm_p, left_wrist_p,
                              left_palm_p, left_index_base_p,
                              left_index_tip_p, left_middle_base_p, left_middle_tip_p,
                              left_third_base_p,
                              left_third_tip_p,
                              left_end_base_p, left_end_tip_p, right_shoulder_p, right_forearm_p,
                              right_wrist_p, right_palm_p,
                              right_index_base_p, right_index_tip_p, right_middle_base_p,
                              right_middle_tip_p,
                              right_third_base_p, right_third_tip_p, right_end_base_p,
                              right_end_tip_p]
        self.outlier_list = []
        self.uninterpolated_left_palm = left_palm
        self.uninterpolated_right_palm = right_palm
        # For graphing examples
        self.uninterpolated_right_palm_v = self.calculate_kinematics_from_position(right_palm)[0]
        self.uninterpolated_left_palm_v = self.calculate_kinematics_from_position(left_palm)[0]
        # Pre-process (threshold, interpolate if necessary/possible, and apply hamming filter post-interpolation
        self.pos_holder, self.acc_holder, self.speed_holder, self.vel_holder = [], [], [], []
        if preprocess:
            self.preprocess_kinematics(p_thresh=0.3)
        # Zero out any large outliers in the data, as well as any low probability events.
        self.zero_out_outliers()
        self.assign_final_variables()
        self.left_PCS = get_principle_components(self.positions[2:14],vel=self.vel_holder[2:14], acc = self.acc_holder[2:14],
                                            num_pcs=3)
        self.right_PCS = get_principle_components(self.positions[14:], vel=self.vel_holder[14:], acc=self.acc_holder[14:],
                                            num_pcs=3)
        self.total_PCS = get_principle_components(self.positions, vel= self.vel_holder, acc= self.acc_holder, num_pcs=3)
        # method to obtain variables for plotting
        right_speeds = np.mean(self.uninterpolated_right_palm_v, axis=1)
        left_speeds = np.mean(self.uninterpolated_left_palm_v, axis=1)
        self.left_palm_f_x = np.union1d(np.where(left_speeds > 1.2),
                                        self.threshold_data_with_probabilities(self.left_palm_p, p_thresh=p_thresh))
        self.right_palm_f_x = np.union1d(np.where(right_speeds > 1.2),
                                         self.threshold_data_with_probabilities(self.right_palm_p, p_thresh=p_thresh))
        self.endpoint_error, self.x_endpoint_error, self.y_endpoint_error, self.z_endpoint_error = \
            self.calculate_endpoint_error()
        return

    def zero_out_outliers(self):
        for idx, pos in enumerate(self.positions):
            outlier_index = self.outlier_list[idx]
            self.speed_holder[idx][outlier_index] = 0
            self.pos_holder[idx][outlier_index] = 0
            self.vel_holder[idx][outlier_index] = 0
            self.acc_holder[idx][outlier_index] = 0
            spatial_outliers = np.where(pos[0,:] < 0.135)[0] # X-value less than cage length
            self.speed_holder[idx][spatial_outliers] = 0
            self.pos_holder[idx][spatial_outliers] = 0
            self.vel_holder[idx][spatial_outliers] = 0
            self.acc_holder[idx][spatial_outliers] = 0
            prob_outliers = np.where(self.probabilities[idx] < 0.2)[0]
            self.speed_holder[idx][prob_outliers] = 0
            self.pos_holder[idx][prob_outliers] = 0
            self.vel_holder[idx][prob_outliers] = 0
            self.acc_holder[idx][prob_outliers] = 0

    def calculate_endpoint_error(self):
        """Function to calculate the kinematic feature 'endpoint_error' by finding the minimum distance between
        the position of the palm and the center of the reaching target."""
        total_distance_error = min(
            min(np.sqrt(((self.left_palm[0, :] - self.handle[0, :]) ** 2 + (self.left_palm[1, :] -
                                                                            self.handle[1, :]) ** 2 +
                         (self.left_palm[2, :] - self.handle[2, :]) ** 2))),
            min(np.sqrt(((self.right_palm[0, :] - self.handle[0, :]) ** 2 + (self.right_palm[1, :] -
                                                                             self.handle[1, :]) ** 2 +
                         (self.right_palm[2, :] - self.handle[2, :]) ** 2))))

        x_distance_error = min(min(self.left_palm[0, :] - self.handle[0, :]),
                               min(self.right_palm[0, :] - self.handle[0, :]))
        y_distance_error = min(min(self.left_palm[1, :] - self.handle[1, :]),
                               min(self.right_palm[1, :] - self.handle[1, :]))
        z_distance_error = min(min(self.left_palm[2, :] - self.handle[2, :]),
                               min(self.right_palm[2, :] - self.handle[2, :]))
        return total_distance_error, x_distance_error, y_distance_error, z_distance_error

    def preprocess_kinematics(self, p_thresh, spline=0.05):
        for di, pos in enumerate(self.positions):
            o_positions = np.asarray(pos)
            probs = self.probabilities[di]
            prob_outliers = self.threshold_data_with_probabilities(probs, p_thresh=p_thresh)
            svd, acc, speed_c = self.calculate_kinematics_from_position(np.copy(pos))
            self.raw_speeds.append(speed_c)
            v_outlier_index = np.where(svd > 1.2)
            possi, num_int, gap_ind = vu.interpolate_3d_vector(np.copy(o_positions), v_outlier_index, prob_outliers)
            try:
                filtered_pos = vu.cubic_spline_smoothing(np.copy(possi), spline_coeff=spline)
            except:
                print('bad filter')
            try:
                v, a, s = self.calculate_kinematics_from_position(np.copy(filtered_pos))
            except:
                print('bf')
            # Find and save still-present outliers in the data
            velocity_outlier_indexes = np.where(s > 1.2)[0]
            # Find array of total outliers
            outliers = np.squeeze(np.union1d(velocity_outlier_indexes, self.prob_filter_index)).flatten()
            if outliers.any():
                self.outlier_list.append(outliers)
            else:
                self.outlier_list.append(0)
            # Append data into proper data structure
            self.pos_holder.append(np.copy(filtered_pos))
            self.vel_holder.append(v)
            self.acc_holder.append(a)
            self.speed_holder.append(s)
        return

    def calculate_kinematics_from_position(self, pos_v):
        """ Function that calculates velocity, speed, and acceleration on a per-bodypart basis."""
        v_hold = np.zeros(pos_v.shape)
        a_hold = np.zeros(pos_v.shape)
        speed_hold = np.zeros(pos_v.shape[0])
        pos_v = pos_v[0:len(self.time_vector)]  # Make sure we don't have a leading number ie extra "time" frame
        for ddx in range(0, pos_v.shape[0]):
            v_hold[ddx, :] = np.copy((pos_v[ddx, :] - pos_v[ddx - 1, :]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
        try:
            for ddx in range(0,3):
                v_hold[:,ddx] = savgol_filter(v_hold[:,ddx],9, 2)
        except:
            pdb.set_trace()
        # Calculate speed, acceleration from smoothed (no time-dependent jumps) velocities
        for ddx in range(0, pos_v.shape[0]):
            speed_hold[ddx] = np.sqrt(v_hold[ddx, 0] ** 2 + v_hold[ddx, 1] ** 2 + v_hold[ddx, 2] ** 2)
            speed_holder = gkern(speed_hold, 1)
        #speed_holder = savgol_filter(speed_hold,11, 1)
        for ddx in range(0, pos_v.shape[0]):
            a_hold[ddx, :] = np.copy(
                (v_hold[ddx, :] - v_hold[ddx - 1, :]) / (self.time_vector[ddx] - self.time_vector[ddx - 1]))
        for ddx in range(0,3):
            a_hold[:,ddx] = savgol_filter(a_hold[:, ddx],9, 2)
            #speed_holder = vu.cubic_spline_smoothing(speed_holder,spline_coeff=.05)
        return np.asarray(v_hold), np.asarray(a_hold), np.asarray(speed_holder)

    def tug_of_war_flag(self):
        self.tug_flag = 0
        self.lick_tug = 0
        init_handle_speed = np.where(self.handle_s[0:10] > 0.04)
        if init_handle_speed.any():
            self.tug_flag = 1
        if self.first_lick_signal < 30:
            self.lick_tug = 1

    def get_filtered_speed_data(self, p_filter=0.3, pos_filter=0.14):
        # Find peaks in left palm time-series with masked data
        lps = np.copy(self.left_palm_s[:])  # Masked left palm speed
        rps = np.copy(self.right_palm_s[:])  # Masked right palm speed
        lps[self.prob_filter_index] = 0  # if p_values < 0.3
        rps[self.prob_filter_index] = 0
        hidx = np.where(self.handle_s > .03)
        lps[hidx] = 0
        rps[hidx] = 0
        # If palms are < 0.4 p-value, remove chance at "maxima"
        left_palm_prob = np.where(self.left_palm_p < p_filter)[0]
        right_palm_prob = np.where(self.right_palm_p < p_filter)[0]
        # If palms are > 0.13m in the x-direction towards the handle 0 position.
        left_palm_pos_f = np.where(self.left_palm[:, 0] < pos_filter)[0]
        right_palm_pos_f = np.where(self.right_palm[:, 0] < pos_filter)[0]
        lps[left_palm_prob] = 0
        rps[right_palm_prob] = 0
        rps[right_palm_pos_f] = 0
        lps[left_palm_pos_f] = 0
        lps[0:4] = 0  # remove any possible edge effect
        rps[0:4] = 0  # remove any possible edge effect
        return lps,rps

    def calculate_robot_handle_speed(self):
        v_hold_x, v_hold_y, v_hold_z = np.zeros((len(self.time_vector))), np.zeros((len(self.time_vector))), \
                                       np.zeros((len(self.time_vector)))
        for ddx in range(0, len(self.time_vector)):
            v_hold_x[ddx] = np.copy((self.x_robot[ddx] - self.x_robot[ddx - 1]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
            v_hold_y[ddx] = np.copy((self.y_robot[ddx] - self.y_robot[ddx - 1]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
            v_hold_z[ddx] = np.copy((self.z_robot[ddx] - self.z_robot[ddx - 1]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
        self.speed_robot = np.sqrt(v_hold_x ** 2 + v_hold_y ** 2 + v_hold_z ** 2)

    def reward_zone_reach(self):
        if np.where(self.reward_zone_sensor > 0)[0].any():
            self.reward_zone = True
        else:
            self.reward_zone = False

    def segment_reaches_with_speed_peaks(self,p_filter_val=0.4, pos_filter_val = 0.14, n_peaks = None, hand=None):
        """ Function to segment out reaches using a positional and velocity threshold. Thresholds are tunable """
        # find peaks of speed velocities
        # For each peak > 0.4 m/s, find the local minima (s < 0.05) for that time series
        # Take minima - 15 as tentative reach start time
        self.behavior_start_time = []
        self.retract_start, self.retract_end = None, None
        self.reach_peak_time = []
        self.left_start_times = []
        self.right_start_times = []
        self.left_peak_times = []
        self.right_peak_times = []
        self.right_reach_end_times = []
        self.left_reach_end_times = []
        self.behavior_duration = []
        self.total_block_reaches = 0
        self.behavior_end_time = 0
        self.handle_movement_mask = False
        # Get some basic information about what's going on in the trial from micro-controller data
        self.reward_zone_reach()
        self.calculate_robot_handle_speed()
        hidx = np.where(self.handle_s > .08)
        hid = np.zeros(self.handle_s.shape[0])
        hid[hidx] = 1  # Mask for handle speed > 0.1
        self.handle_movement_mask = hid
        if np.count_nonzero(hid) > 1:
            self.handle_moved = True
            print('Handle Movement Detected')
            try:
                self.retract_start = np.where(hid == 1)[0][0]
                self.grasp = self.retract_start
                self.retract_end = np.where(hid == 1)[0][-1] + 5
            # Find where handle stops moving
            except:
                self.retract_start = None
                self.retract_end = None
                self.grasp = None
        else:
            self.handle_moved = False
            self.retract_start, self.retract_end, self.grasp = None, None, None
        if self.reward_zone: # Handle made it to reward zone area.
            self.rewarded = True
        else:
            self.rewarded = False
        lps, rps = self.get_filtered_speed_data(p_filter= p_filter_val, pos_filter=pos_filter_val)
        start_pad = 10
        # Get possible reach peaks, use to segment reaching behavior
        self.left_palm_maxima = find_peaks(lps, height=0.25, distance=40)[0]
        self.right_palm_maxima = find_peaks(rps, height=0.25, distance=40)[0]
        if self.handle_moved:
            #pdb.set_trace()
            if np.where(self.reward_zone_sensor[0:20] > 0)[0].any():# Handle in reward zone too early
                    self.tug_flag = True
                    print('Tug of War')
            if self.left_palm_maxima.any() or self.right_palm_maxima.any():
                print('Reaches Found')
            else:
                # Get lps with less stringent filters
                lps, rps = self.get_filtered_speed_data(p_filter= 0.4, pos_filter=.135)
                self.left_palm_maxima = np.asarray(np.argmax(lps)).reshape(1) # Take max value
                self.right_palm_maxima = np.asarray(np.argmax(rps)).reshape(1) # Take max value
                if self.left_palm_maxima.any():
                    if self.left_palm_s[self.left_palm_maxima[0]] < 0.1:
                        self.left_palm_maxima = np.array([])
                if self.right_palm_maxima.any():
                    if self.right_palm_s[self.right_palm_maxima[0]] < 0.1:
                        self.right_palm_maxima = np.array([])

        if self.left_palm_maxima.any():
            print('Left Palm Reach')
            left_palm_max_list = self.left_palm_maxima.tolist()
            for ir, maxes in enumerate(left_palm_max_list):
                try:
                    if maxes + 35 > left_palm_max_list[ir+1]: # is next entry close?
                        left_palm_max_list.pop(ir+1)
                except:
                    pass
            self.left_palm_maxima = np.asarray(left_palm_max_list)
            # demarcate multiple reaches through looking at steps (<40 between peaks is multiple)
            for ir in range(0, self.left_palm_maxima.shape[0]):
                if self.left_palm_maxima[ir] > 31:
                    try:
                        below_reach_speed = self.left_palm_s[self.left_palm_maxima[ir] - 30: self.left_palm_maxima[ir]]
                        index_below = np.where(np.asarray(below_reach_speed) < 0.2)[-1][
                            -1]  # find last index where speed < 0.1
                        left_palm_below_thresh = self.left_palm_maxima[ir] - index_below - start_pad
                    except:
                        left_palm_below_thresh = self.left_palm_maxima[ir] - \
                                                 np.argmin(self.left_palm_s[self.left_palm_maxima[ir] - 30:
                                                                            self.left_palm_maxima[ir]]) - start_pad
                else:
                    lmax = self.left_palm_maxima[ir]
                    below_reach_speed = self.left_palm_s[0:lmax]
                    try:
                        index_below = np.where(np.asarray(below_reach_speed) < 0.2)[-1][-1]
                        if index_below < start_pad:
                            left_palm_below_thresh = 0
                        else:
                            left_palm_below_thresh = self.left_palm_maxima[ir] - index_below - start_pad
                    except:
                        pdb.set_trace()
                        left_palm_below_thresh = 0

                try:
                    left_palm_below_thresh_after = self.left_palm_maxima[ir] + \
                                               np.where(np.asarray(self.left_palm_s[
                                                                   self.left_palm_maxima[ir]:
                                                                   self.left_palm_maxima[ir] + 50]) < 0.15)[0][0]
                except:
                    left_palm_below_thresh_after = self.left_palm_maxima[ir] + 30
                start_time_l = left_palm_below_thresh
                try:
                    if start_time_l < 0:
                        start_time_l = 1
                except:
                    pass
                self.left_start_times.append(start_time_l)
                self.left_peak_times.append(self.left_palm_maxima[ir])
                self.reach_peak_time.append(self.left_palm_maxima[ir])  # Record Peak
                self.left_reach_end_times.append(left_palm_below_thresh_after)  # Record putative end of motion
            # Find peaks in right palm time-series
        else:
            self.left_peak_times.append(None)
            self.left_reach_end_times.append(None)
            self.left_start_times.append(None)
            self.reach_peak_time.append(None)
        if self.right_palm_maxima.any():
            print('Right Palm Reach')
            right_palm_max_list = self.right_palm_maxima.tolist()
            for ir, maxes in enumerate(right_palm_max_list):
                try:
                    if maxes + 35 > right_palm_max_list[ir+1]: # is next entry close?
                        right_palm_max_list.pop(ir+1)
                except:
                    pass
            self.right_palm_maxima = np.asarray(right_palm_max_list)
            for ir in range(0, self.right_palm_maxima.shape[0]):
                if self.right_palm_maxima[ir] > 31:
                    try:
                        below_reach_speed = self.right_palm_s[
                                            self.right_palm_maxima[ir] - 30: self.right_palm_maxima[ir]]
                        index_below = np.where(np.asarray(below_reach_speed) < 0.2)[-1][-1]  # find last index where speed < 0.1
                        right_palm_below_thresh = self.right_palm_maxima[ir] - index_below - start_pad
                    except:
                        right_palm_below_thresh = self.right_palm_maxima[ir] - np.argmin(
                            self.right_palm_s[self.right_palm_maxima[ir] - 30: self.right_palm_maxima[ir]] - start_pad)
                else:
                    rmax = self.right_palm_maxima[ir]
                    below_reach_speed = self.right_palm_s[0:rmax]
                    try:
                        index_below = np.where(np.asarray(below_reach_speed) < 0.2)[0]
                        if index_below < start_pad:
                            right_palm_below_thresh = 0
                        else:
                            right_palm_below_thresh = self.right_palm_maxima[ir] - index_below - start_pad
                    except:
                        right_palm_below_thresh = 0
                try:
                    right_palm_below_thresh_after = self.right_palm_maxima[ir] + \
                                                    np.where(np.asarray(self.right_palm_s[
                                                                        self.right_palm_maxima[ir]:
                                                                        self.right_palm_maxima[
                                                                            ir] + 50]) < 0.15)[0][0]
                except:
                        right_palm_below_thresh_after = self.right_palm_maxima[ir] + 10
                start_time_r = right_palm_below_thresh
                try:
                    if start_time_r < 0:
                        start_time_r = 1
                except:
                    pass
                self.right_start_times.append(start_time_r)
                self.right_peak_times.append(self.right_palm_maxima[ir])
                self.reach_peak_time.append(self.right_palm_maxima[ir])
                self.right_reach_end_times.append(right_palm_below_thresh_after)
        else:
            self.right_peak_times.append(None)
            self.right_reach_end_times.append(None)
            self.right_start_times.append(None)
            self.reach_peak_time.append(None)
        #pdb.set_trace()
        # Check for unrealistic start and stop values (late in trial), adjust accordingly
        for idx, time in enumerate(self.right_start_times):
            try:
                if any in self.right_start_times is None:
                    continue
                else:
                    try:
                        if self.right_start_times[idx] > 650:
                            self.right_start_times[idx] = 650
                            if time > 700:
                                self.right_reach_end_times[idx] = 700  # Max trial time.
                    except:
                        pass
            except:
                pass
        for idx, time in enumerate(self.left_start_times):
            try:
                if self.left_start_times.any() is None:
                    continue
                else:
                    try:
                        if self.left_start_times[idx] > 650:
                            self.left_start_times[idx] = 650
                            if time > 700:
                                self.left_reach_end_times[idx] = 700  # Max trial time.
                    except:
                        pass
            except:
                pass
        #pdb.set_trace()
        self.find_reach_information() # obtain general information about reaching in trial
        self.trial_cut_vector = []

    def find_reach_information(self):
        self.multiple_reaches_flag = False
        self.individual_reach_times, self.individual_reach_end_times, self.individual_reach_peak_times = [], [], []
        self.reach_hand_type = []
        self.num_peaks = 0
        self.num_reaches_split = 0
        self.individual_grasp_times, self.individual_retract_times, self.individual_reach_duration = [], [], []
        in_flag = True
        # Check Handle Behavior wrt reaching statistics to detect tug of war or pre-reaching
        if self.handle_moved is True and self.right_start_times is None and self.left_start_times is None:
            self.handle_moved = False
        # Take min of right and left start times as "reach times" for start of classification extraction
        self.individual_grasp_times.append(self.grasp)
        self.individual_retract_times.append(self.retract_start)
        if None in self.right_start_times and None in self.left_start_times: # No reaching in trial found
            self.trial_type = False
            self.num_peaks = 0
            self.num_reaches_split = None
            self.reach_hand_type.append(None)
            self.individual_reach_times.append(None)
            self.individual_grasp_times.append(None)
            self.individual_retract_times.append(None)
            self.individual_reach_duration.append(None)
            self.behavior_start_time = 0
            self.behavior_end_time = 200
        # Right start times
        elif None in self.left_start_times:
            self.trial_type = True
            self.num_peaks = len(self.right_peak_times)
            self.num_reaches_split = len(self.right_start_times)
            if self.num_reaches_split > 1:
                self.multiple_reaches_flag = True
                self.individual_reach_times = self.right_start_times
                self.individual_reach_end_times = self.right_reach_end_times
                self.individual_reach_peak_times = self.right_peak_times
            else:
                self.individual_reach_times = [self.right_start_times[-1]]
                self.individual_reach_end_times = [self.right_reach_end_times[-1]]
                self.individual_reach_peak_times = [self.right_peak_times[-1]]

            for items in self.right_start_times:
                self.reach_hand_type.append('Right')
            try:
                self.behavior_start_time = min(self.right_start_times)
                self.behavior_end_time = self.retract_end
            except:
                self.behavior_start_time = self.right_start_times[0]
                self.behavior_end_time = self.right_reach_end_times[-1]
        # Left start times
        elif None in self.right_start_times: # If there are none in right_start_times
            self.trial_type = True
            self.num_peaks = len(self.left_peak_times)
            self.num_reaches_split = len(self.left_start_times)
            if self.num_reaches_split > 1: # Count and append parts of behavior
                self.multiple_reaches_flag = True
                self.individual_reach_times = self.left_start_times
                self.individual_reach_end_times = self.left_reach_end_times
                self.individual_reach_peak_times = self.left_peak_times
                self.individual_grasp_times = self.grasp
                self.individual_retract_times = self.retract_start
            else: # Count and append different parts of behavior
                self.individual_reach_times = [self.left_start_times[-1]]
                self.individual_reach_end_times = [self.left_reach_end_times[-1]]
                self.individual_reach_peak_times = [self.left_peak_times[-1]]
            for items in self.left_start_times: # Append hand type.
                self.reach_hand_type.append('Left')
            try: # Get beginning and end times
                self.behavior_start_time = min(self.left_start_times)
                self.behavior_end_time = self.retract_end
            except:
                self.behavior_start_time = self.left_start_times[0]
                self.behavior_end_time = self.left_reach_end_times[-1]
        # Both hands perform reaching behavior
        elif None not in self.left_start_times and None not in self.right_start_times:
            # Two-hands reaching, check times that reaching occurs in both hands
            # If one hand is ~20 frames within eachother, seperate reaches
            # If hands together, bimanual
            self.trial_type = True
            for idr, r_reach in enumerate(self.right_start_times): # Compare indexes
                r_reach_end_time = self.right_reach_end_times[idr]
                r_reach_peak_time = self.right_peak_times[idr]
                for idl, l_reach in enumerate(self.left_start_times):
                    l_reach_end_time = self.left_start_times[idl]
                    l_reach_peak_time = self.left_peak_times[idl]
                # Check here for each reach
                    if l_reach - 5 < r_reach < l_reach + 5: # is r reach within 30 frames of a l reach
                        print('bimanual')
                        print(l_reach, r_reach)
                        self.reach_hand_type.append('Bimanual')
                        self.num_reaches_split += 1
                        if l_reach < r_reach:
                            self.individual_reach_times.append(l_reach)
                            self.individual_reach_end_times.append(l_reach_end_time)
                            self.individual_reach_peak_times.append(l_reach_peak_time)
                        else:
                            self.individual_reach_times.append(r_reach)
                            self.individual_reach_end_times.append(r_reach_end_time)
                            self.individual_reach_peak_times.append(r_reach_peak_time)
            for l_reach in self.left_start_times:
                        if l_reach not in self.individual_reach_times:
                            for reaches in self.individual_reach_times:
                                if l_reach - 10 < reaches < l_reach + 10:
                                    print('no add')
                                    in_flag=False
                        if in_flag:
                            #pdb.set_trace()
                            self.reach_hand_type.append('Left')
                            self.individual_reach_times.append(l_reach)
                            self.individual_reach_peak_times.append(l_reach_peak_time)
                            self.individual_reach_end_times.append(l_reach_end_time)
                            self.num_reaches_split += 1
                        in_flag = True
            for r_reach in self.right_start_times:
                        if r_reach not in self.individual_reach_times:
                            for reaches in self.individual_reach_times:
                                if r_reach - 20 < reaches < r_reach + 20:
                                    print('no add')
                                    in_flag = False
                        if in_flag:
                            #pdb.set_trace()
                            self.reach_hand_type.append('Right')
                            self.individual_reach_times.append(r_reach)
                            self.individual_reach_peak_times.append(r_reach_peak_time)
                            self.individual_reach_end_times.append(r_reach_end_time)
                            self.num_reaches_split += 1
                        in_flag = True
            self.num_peaks = len(self.individual_reach_times)
            try:
                self.behavior_start_time = np.amin(
                    np.hstack((np.asarray(self.right_start_times), np.asarray(self.left_start_times))))
                if self.retract_end is not None:
                    self.behavior_end_time = self.retract_end
                else:
                    self.behavior_end_time = np.amax(np.hstack((np.asarray(self.right_reach_end_times),
                                                                np.asarray(self.left_reach_end_times))))
            except:
                print('fail b loop')
                pdb.set_trace()
                self.behavior_start_time = np.amin(
                    np.hstack((np.asarray(self.right_start_times), np.asarray(self.left_start_times))))
                self.behavior_end_time = np.amax(np.hstack((np.asarray(self.right_reach_end_times),
                                                            np.asarray(self.left_reach_end_times))))
        else:
            self.trial_type = False
            self.behavior_start_time = 0
            self.behavior_end_time = 200
            self.num_peaks = 0
            self.reach_hand_type.append(None)
            self.individual_reach_times.append(None)
        if self.individual_reach_times is not None:
            for irx, reaches in enumerate(self.individual_reach_times):
                try:
                    self.individual_reach_duration.append(self.individual_reach_end_times[irx] - reaches)
                except:
                    self.individual_reach_duration.append(None)
        print('npeaks = ' + str(self.num_peaks), self.reach_hand_type, self.individual_reach_times)
        if self.behavior_end_time is not None:
            if self.behavior_end_time < 20 and self.handle_moved is False: # unsuccessful reach early into trial
                self.behavior_end_time = self.behavior_end_time + 40 # Pad by 200ms
        if all(ix is not None for ix in self.left_palm_maxima) and all(
                ix is not None for ix in self.right_palm_maxima):  # no reach flag
            pass
        if self.behavior_start_time is not None and self.behavior_end_time is None:
            if self.retract_end:
                self.behavior_end_time=self.retract_end
            elif self.individual_reach_end_times is not None:
                self.behavior_end_time = self.individual_reach_end_times[-1]
            else:
                self.behavior_end_time = self.behavior_start_time + 150
        if self.trial_type:
            try:
                self.behavior_duration = self.behavior_end_time - self.behavior_start_time
            except:
                pdb.set_trace()
        else:
            self.behavior_duration = None

    def assign_final_variables(self):
        [self.nose, self.handle, self.left_shoulder, self.left_forearm, self.left_wrist,
         self.left_palm, self.left_index_base, self.left_index_tip, self.left_middle_base,
         self.left_middle_tip, self.left_third_base, self.left_third_tip, self.left_end_base, self.left_end_tip,
         self.right_shoulder, self.right_forearm, self.right_wrist, self.right_palm, self.right_index_base,
         self.right_index_tip, self.right_middle_base, self.right_middle_tip, self.right_third_base,
         self.right_third_tip, self.right_end_base, self.right_end_tip] = self.pos_holder

        [self.nose_p, self.handle_p, self.left_shoulder_p, self.left_forearm_p, self.left_wrist_p,
         self.left_palm_p, self.left_index_base_p, self.left_index_tip_p, self.left_middle_base_p,
         self.left_middle_tip_p, self.left_third_base_p, self.left_third_tip_p, self.left_end_base_p,
         self.left_end_tip_p,
         self.right_shoulder_p, self.right_forearm_p, self.right_wrist_p, self.right_palm_p, self.right_index_base_p,
         self.right_index_tip_p, self.right_middle_base_p, self.right_middle_tip_p, self.right_third_base_p,
         self.right_third_tip_p, self.right_end_base_p, self.right_end_tip_p] = self.probabilities

        [self.nose_v, self.handle_v, self.left_shoulder_v, self.left_forearm_v, self.left_wrist_v,
         self.left_palm_v, self.left_index_base_v, self.left_index_tip_v, self.left_middle_base_v,
         self.left_middle_tip_v, self.left_third_base_v, self.left_third_tip_v, self.left_end_base_v,
         self.left_end_tip_v,
         self.right_shoulder_v, self.right_forearm_v, self.right_wrist_v, self.right_palm_v, self.right_index_base_v,
         self.right_index_tip_v, self.right_middle_base_v, self.right_middle_tip_v, self.right_third_base_v,
         self.right_third_tip_v, self.right_end_base_v, self.right_end_tip_v] = self.vel_holder

        [self.nose_s, self.handle_s, self.left_shoulder_s, self.left_forearm_s, self.left_wrist_s, self.left_palm_s,
         self.left_index_base_s, self.left_index_tip_s, self.left_middle_base_s, self.left_middle_tip_s,
         self.left_third_base_s, self.left_third_tip_s, self.left_end_base_s, self.left_end_tip_s,
         self.right_shoulder_s,
         self.right_forearm_s, self.right_wrist_s, self.right_palm_s, self.right_index_base_s, self.right_index_tip_s,
         self.right_middle_base_s, self.right_middle_tip_s, self.right_third_base_s, self.right_third_tip_s,
         self.right_end_base_s, self.right_end_tip_s] = self.speed_holder

        [self.nose_a, self.handle_a, self.left_shoulder_a, self.left_forearm_a, self.left_wrist_a, self.left_palm_a,
         self.left_index_base_a, self.left_index_tip_a, self.left_middle_base_a, self.left_middle_tip_a,
         self.left_third_base_a,
         self.left_third_tip_a, self.left_end_base_a, self.left_end_tip_a, self.right_shoulder_a, self.right_forearm_a,
         self.right_wrist_a, self.right_palm_a, self.right_index_base_a, self.right_index_tip_a,
         self.right_middle_base_a,
         self.right_middle_tip_a, self.right_third_base_a, self.right_third_tip_a, self.right_end_base_a,
         self.right_end_tip_a] = self.acc_holder

        [self.nose_o, self.handle_o, self.left_shoulder_o, self.left_forearm_o, self.left_wrist_o, self.left_palm_o,
         self.left_index_base_o, self.left_index_tip_o, self.left_middle_base_o, self.left_middle_tip_o,
         self.left_third_base_o, self.left_third_tip_o, self.left_end_base_o, self.left_end_tip_o,
         self.right_shoulder_o,
         self.right_forearm_o, self.right_wrist_o, self.right_palm_o, self.right_index_base_o, self.right_index_tip_o,
         self.right_middle_base_o, self.right_middle_tip_o, self.right_third_base_o, self.right_third_tip_o,
         self.right_end_base_o, self.right_end_tip_o] = self.outlier_list
        return

    def segment_data_into_reach_dict(self, trial_num, error_flag=False, append_reach_information = False,
                                     append_outliers = False):
        """ Function that iterates over reaching indices,
            saves the segmented data and corresponding class labels into a dataframe.
        """
        # Check segment flags for none-type list
        try:
            if len(self.arm_id_list) == 0:
                self.arm_id_list = 0
        except:
            pass
        try:
            if len(self.right_start_times[0]) == 0:
                self.right_start_times = 0
                self.right_reach_end_times = 0
                self.right_peak_times = 0
        except:
            pass
        try:
            if len(self.left_start_times[0]) == 0:
                self.left_start_times = 0
                self.left_reach_end_times = 0
                self.left_peak_times = 0
        except:
            pass
        try:
            if len(self.behavior_start_time[0]) == 0:
                self.behavior_start_time = 0
                self.behavior_duration = 0
                self.behavior_end_time = 0
        except:
            pass
        self.save_dict = {'nose_vx': np.asarray(self.nose_v[:, 0]), 'nose_vy': self.nose_v[:, 1],
                          'nose_vz': self.nose_v[:, 2],
                          'handle_vx': self.handle_v[:, 0], 'handle_vy': self.handle_v[:, 1],
                          'handle_vz': self.handle_v[:, 2],
                          'left_shoulder_vx': self.left_shoulder_v[:, 0],
                          'left_shoulder_vy': self.left_shoulder_v[:, 1],
                          'left_shoulder_vz': self.left_shoulder_v[:, 2],
                          'left_forearm_vx': self.left_forearm_v[:, 0],
                          'left_forearm_vy': self.left_forearm_v[:, 1],
                          'left_forearm_vz': self.left_forearm_v[:, 2],
                          'left_wrist_vx': self.left_wrist_v[:, 0], 'left_wrist_vy': self.left_wrist_v[:, 1],
                          'left_wrist_vz': self.left_wrist_v[:, 2],
                          'left_palm_vx': self.left_palm_v[:, 0], 'left_palm_vy': self.left_palm_v[:, 1],
                          'left_palm_vz': self.left_palm_v[:, 2],
                          'left_index_base_vx': self.left_index_base_v[:, 0],
                          'left_index_base_vy': self.left_index_base_v[:, 1],
                          'left_index_base_vz': self.left_index_base_v[:, 2],
                          'left_index_tip_vx': self.left_index_tip_v[:, 0],
                          'left_index_tip_vy': self.left_index_tip_v[:, 1],
                          'left_index_tip_vz': self.left_index_tip_v[:, 2],
                          'left_middle_base_vx': self.left_middle_base_v[:, 0],
                          'left_middle_base_vy': self.left_middle_base_v[:, 1],
                          'left_middle_base_vz': self.left_middle_base_v[:, 2],
                          'left_middle_tip_vx': self.left_middle_tip_v[:, 0],
                          'left_middle_tip_vy': self.left_middle_tip_v[:, 1],
                          'left_middle_tip_vz': self.left_middle_tip_v[:, 2],
                          'left_third_base_vx': self.left_third_base_v[:, 0],
                          'left_third_base_vy': self.left_third_base_v[:, 1],
                          'left_third_base_vz': self.left_third_base_v[:, 2],
                          'left_third_tip_vx': self.left_third_tip_v[:, 0],
                          'left_third_tip_vy': self.left_third_tip_v[:, 1],
                          'left_third_tip_vz': self.left_third_tip_v[:, 2],
                          'left_end_base_vx': self.left_end_base_v[:, 0],
                          'left_end_base_vy': self.left_end_base_v[:, 1],
                          'left_end_base_vz': self.left_end_base_v[:, 2],
                          'left_end_tip_vx': self.left_end_tip_v[:, 0],
                          'left_end_tip_vy': self.left_end_tip_v[:, 1],
                          'left_end_tip_vz': self.left_end_tip_v[:, 2],
                          'right_shoulder_vx': self.right_shoulder_v[:, 0],
                          'right_shoulder_vy': self.right_shoulder_v[:, 1],
                          'right_shoulder_vz': self.right_shoulder_v[:, 2],
                          'right_forearm_vx': self.right_forearm_v[:, 0],
                          'right_forearm_vy': self.right_forearm_v[:, 1],
                          'right_forearm_vz': self.right_forearm_v[:, 2],
                          'right_wrist_vx': self.right_wrist_v[:, 0], 'right_wrist_vy': self.right_wrist_v[:, 1],
                          'right_wrist_vz': self.right_wrist_v[:, 2],
                          'right_palm_vx': self.right_palm_v[:, 0], 'right_palm_vy': self.right_palm_v[:, 1],
                          'right_palm_vz': self.right_palm_v[:, 2],
                          'right_index_base_vx': self.right_index_base_v[:, 0],
                          'right_index_base_vy': self.right_index_base_v[:, 1],
                          'right_index_base_vz': self.right_index_base_v[:, 1],
                          'right_index_tip_vx': self.right_index_tip_v[:, 0],
                          'right_index_tip_vy': self.right_index_tip_v[:, 1],
                          'right_index_tip_vz': self.right_index_tip_v[:, 1],
                          'right_middle_base_vx': self.right_middle_base_v[:, 0],
                          'right_middle_base_vy': self.right_middle_base_v[:, 1],
                          'right_middle_base_vz': self.right_middle_base_v[:, 2],
                          'right_middle_tip_vx': self.right_middle_tip_v[:, 0],
                          'right_middle_tip_vy': self.right_middle_tip_v[:, 1],
                          'right_middle_tip_vz': self.right_middle_tip_v[:, 2],
                          'right_third_base_vx': self.right_third_base_v[:, 0],
                          'right_third_base_vy': self.right_third_base_v[:, 1],
                          'right_third_base_vz': self.right_third_base_v[:, 2],
                          'right_third_tip_vx': self.right_third_tip_v[:, 0],
                          'right_third_tip_vy': self.right_third_tip_v[:, 1],
                          'right_third_tip_vz': self.right_third_tip_v[:, 2],
                          'right_end_base_vx': self.right_end_base_v[:, 0],
                          'right_end_base_vy': self.right_end_base_v[:, 1],
                          'right_end_base_vz': self.right_end_base_v[:, 2],
                          'right_end_tip_vx': self.right_end_tip_v[:, 0],
                          'right_end_tip_vy': self.right_end_tip_v[:, 1],
                          'right_end_tip_vz': self.right_end_tip_v[:, 2],
                          # Accelerations
                          'nose_ax': self.nose_a[:, 0], 'nose_ay': self.nose_a[:, 1], 'nose_az': self.nose_a[:, 2],
                          'handle_ax': self.handle_a[:, 0], 'handle_ay': self.handle_a[:, 1],
                          'handle_az': self.handle_a[:, 2],
                          'left_shoulder_ax': self.left_shoulder_a[:, 0],
                          'left_shoulder_ay': self.left_shoulder_a[:, 1],
                          'left_shoulder_az': self.left_shoulder_a[:, 2],
                          'left_forearm_ax': self.left_forearm_a[:, 0],
                          'left_forearm_ay': self.left_forearm_a[:, 1],
                          'left_forearm_az': self.left_forearm_a[:, 2],
                          'left_wrist_ax': self.left_wrist_a[:, 0], 'left_wrist_ay': self.left_wrist_a[:, 1],
                          'left_wrist_az': self.left_wrist_a[:, 2],
                          'left_palm_ax': self.left_palm_a[:, 0], 'left_palm_ay': self.left_palm_a[:, 1],
                          'left_palm_az': self.left_palm_a[:, 2],
                          'left_index_base_ax': self.left_index_base_a[:, 0],
                          'left_index_base_ay': self.left_index_base_a[:, 1],
                          'left_index_base_az': self.left_index_base_a[:, 2],
                          'left_index_tip_ax': self.left_index_tip_a[:, 0],
                          'left_index_tip_ay': self.left_index_tip_a[:, 1],
                          'left_index_tip_az': self.left_index_tip_a[:, 2],
                          'left_middle_base_ax': self.left_middle_base_a[:, 0],
                          'left_middle_base_ay': self.left_middle_base_a[:, 1],
                          'left_middle_base_az': self.left_middle_base_a[:, 2],
                          'left_middle_tip_ax': self.left_middle_tip_a[:, 0],
                          'left_middle_tip_ay': self.left_middle_tip_a[:, 1],
                          'left_middle_tip_az': self.left_middle_tip_a[:, 2],
                          'left_third_base_ax': self.left_third_base_a[:, 0],
                          'left_third_base_ay': self.left_third_base_a[:, 1],
                          'left_third_base_az': self.left_third_base_a[:, 2],
                          'left_third_tip_ax': self.left_third_tip_a[:, 0],
                          'left_third_tip_ay': self.left_third_tip_a[:, 1],
                          'left_third_tip_az': self.left_third_tip_a[:, 2],
                          'left_end_base_ax': self.left_end_base_a[:, 0],
                          'left_end_base_ay': self.left_end_base_a[:, 1],
                          'left_end_base_az': self.left_end_base_a[:, 2],
                          'left_end_tip_ax': self.left_end_tip_a[:, 0],
                          'left_end_tip_ay': self.left_end_tip_a[:, 1],
                          'left_end_tip_az': self.left_end_tip_a[:, 2],
                          'right_shoulder_ax': self.right_shoulder_a[:, 0],
                          'right_shoulder_ay': self.right_shoulder_a[:, 1],
                          'right_shoulder_az': self.right_shoulder_a[:, 2],
                          'right_forearm_ax': self.right_forearm_a[:, 0],
                          'right_forearm_ay': self.right_forearm_a[:, 1],
                          'right_forearm_az': self.right_forearm_a[:, 2],
                          'right_wrist_ax': self.right_wrist_a[:, 0], 'right_wrist_ay': self.right_wrist_a[:, 1],
                          'right_wrist_az': self.right_wrist_a[:, 2],
                          'right_palm_ax': self.right_palm_a[:, 0], 'right_palm_ay': self.right_palm_a[:, 1],
                          'right_palm_az': self.right_palm_a[:, 2],
                          'right_index_base_ax': self.right_index_base_a[:, 0],
                          'right_index_base_ay': self.right_index_base_a[:, 1],
                          'right_index_base_az': self.right_index_base_a[:, 1],
                          'right_index_tip_ax': self.right_index_tip_a[:, 0],
                          'right_index_tip_ay': self.right_index_tip_a[:, 1],
                          'right_index_tip_az': self.right_index_tip_a[:, 1],
                          'right_middle_base_ax': self.right_middle_base_a[:, 0],
                          'right_middle_base_ay': self.right_middle_base_a[:, 1],
                          'right_middle_base_az': self.right_middle_base_a[:, 2],
                          'right_middle_tip_ax': self.right_middle_tip_a[:, 0],
                          'right_middle_tip_ay': self.right_middle_tip_a[:, 1],
                          'right_middle_tip_az': self.right_middle_tip_a[:, 2],
                          'right_third_base_ax': self.right_third_base_a[:, 0],
                          'right_third_base_ay': self.right_third_base_a[:, 1],
                          'right_third_base_az': self.right_third_base_a[:, 2],
                          'right_third_tip_ax': self.right_third_tip_a[:, 0],
                          'right_third_tip_ay': self.right_third_tip_a[:, 1],
                          'right_third_tip_az': self.right_third_tip_a[:, 2],
                          'right_end_base_ax': self.right_end_base_a[:, 0],
                          'right_end_base_ay': self.right_end_base_a[:, 1],
                          'right_end_base_az': self.right_end_base_a[:, 2],
                          'right_end_tip_ax': self.right_end_tip_a[:, 0],
                          'right_end_tip_ay': self.right_end_tip_a[:, 1],
                          'right_end_tip_az': self.right_end_tip_a[:, 2],
                          # Speeds
                          'nose_s': self.nose_s, 'handle_s': self.handle_s, 'left_shoulder_s': self.left_shoulder_s,
                          'left_forearm_s': self.left_forearm_s, 'left_wrist_s': self.left_wrist_s,
                          'left_palm_s': self.left_palm_s, 'left_index_base_s': self.left_index_base_s,
                          'left_index_tip_s': self.left_index_tip_s, 'left_middle_base_s': self.left_middle_base_s,
                          'left_middle_tip_s': self.left_middle_tip_s,
                          'left_third_base_s': self.left_third_base_s, 'left_third_tip_s': self.left_third_tip_s,
                          'left_end_base_s': self.left_end_base_s,
                          'left_end_tip_s': self.left_end_tip_s, 'right_shoulder_s': self.right_shoulder_s,
                          'right_forearm_s': self.right_forearm_s, 'right_wrist_s': self.right_wrist_s,
                          'right_palm_s': self.right_palm_s, 'right_index_base_s': self.right_index_base_s,
                          'right_index_tip_s': self.right_index_tip_s,
                          'right_middle_base_s': self.right_middle_base_s,
                          'right_middle_tip_s': self.right_middle_tip_s,
                          'right_third_base_s': self.right_third_base_s,
                          'right_third_tip_s': self.right_third_tip_s,
                          'right_end_base_s': self.right_end_base_s, 'right_end_tip_s': self.right_end_tip_s,
                          # Positions
                          'nose_px': self.nose[:, 0], 'nose_py': self.nose[:, 1], 'nose_pz': self.nose[:, 2],
                          'handle_px': self.handle[:, 0], 'handle_py': self.handle[:, 1],
                          'handle_pz': self.handle[:, 2],
                          'left_shoulder_px': self.left_shoulder[:, 0],
                          'left_shoulder_py': self.left_shoulder[:, 1],
                          'left_shoulder_pz': self.left_shoulder[:, 2],
                          'left_forearm_px': self.left_forearm[:, 0], 'left_forearm_py': self.left_forearm[:, 1],
                          'left_forearm_pz': self.left_forearm[:, 2],
                          'left_wrist_px': self.left_wrist[:, 0], 'left_wrist_py': self.left_wrist[:, 1],
                          'left_wrist_pz': self.left_wrist[:, 2],
                          'left_palm_px': self.left_palm[:, 0], 'left_palm_py': self.left_palm[:, 1],
                          'left_palm_pz': self.left_palm[:, 2],
                          'left_index_base_px': self.left_index_base[:, 0],
                          'left_index_base_py': self.left_index_base[:, 1],
                          'left_index_base_pz': self.left_index_base[:, 2],
                          'left_index_tip_px': self.left_index_tip[:, 0],
                          'left_index_tip_py': self.left_index_tip[:, 1],
                          'left_index_tip_pz': self.left_index_tip[:, 2],
                          'left_middle_base_px': self.left_middle_base[:, 0],
                          'left_middle_base_py': self.left_middle_base[:, 1],
                          'left_middle_base_pz': self.left_middle_base[:, 2],
                          'left_middle_tip_px': self.left_middle_tip[:, 0],
                          'left_middle_tip_py': self.left_middle_tip[:, 1],
                          'left_middle_tip_pz': self.left_middle_tip[:, 2],
                          'left_third_base_px': self.left_third_base[:, 0],
                          'left_third_base_py': self.left_third_base[:, 1],
                          'left_third_base_pz': self.left_third_base[:, 2],
                          'left_third_tip_px': self.left_third_tip[:, 0],
                          'left_third_tip_py': self.left_third_tip[:, 1],
                          'left_third_tip_pz': self.left_third_tip[:, 2],
                          'left_end_base_px': self.left_end_base[:, 0],
                          'left_end_base_py': self.left_end_base[:, 1],
                          'left_end_base_pz': self.left_end_base[:, 2],
                          'left_end_tip_px': self.left_end_tip[:, 0], 'left_end_tip_py': self.left_end_tip[:, 1],
                          'left_end_tip_pz': self.left_end_tip_a[:, 2],
                          'right_shoulder_px': self.right_shoulder[:, 0],
                          'right_shoulder_py': self.right_shoulder[:, 1],
                          'right_shoulder_pz': self.right_shoulder[:, 2],
                          'right_forearm_px': self.right_forearm[:, 0],
                          'right_forearm_py': self.right_forearm[:, 1],
                          'right_forearm_pz': self.right_forearm[:, 2],
                          'right_wrist_px': self.right_wrist[:, 0], 'right_wrist_py': self.right_wrist[:, 1],
                          'right_wrist_pz': self.right_wrist[:, 2],
                          'right_palm_px': self.right_palm[:, 0], 'right_palm_py': self.right_palm[:, 1],
                          'right_palm_pz': self.right_palm[:, 2],
                          'right_index_base_px': self.right_index_base[:, 0],
                          'right_index_base_py': self.right_index_base[:, 1],
                          'right_index_base_pz': self.right_index_base[:, 1],
                          'right_index_tip_px': self.right_index_tip[:, 0],
                          'right_index_tip_py': self.right_index_tip[:, 1],
                          'right_index_tip_pz': self.right_index_tip[:, 1],
                          'right_middle_base_px': self.right_middle_base[:, 0],
                          'right_middle_base_py': self.right_middle_base[:, 1],
                          'right_middle_base_pz': self.right_middle_base[:, 2],
                          'right_middle_tip_px': self.right_middle_tip[:, 0],
                          'right_middle_tip_py': self.right_middle_tip[:, 1],
                          'right_middle_tip_pz': self.right_middle_tip[:, 2],
                          'right_third_base_px': self.right_third_base[:, 0],
                          'right_third_base_py': self.right_third_base[:, 1],
                          'right_third_base_pz': self.right_third_base[:, 2],
                          'right_third_tip_px': self.right_third_tip[:, 0],
                          'right_third_tip_py': self.right_third_tip[:, 1],
                          'right_third_tip_pz': self.right_third_tip[:, 2],
                          'right_end_base_px': self.right_end_base[:, 0],
                          'right_end_base_py': self.right_end_base[:, 1],
                          'right_end_base_pz': self.right_end_base[:, 2],
                          'right_end_tip_px': self.right_end_tip[:, 0],
                          'right_end_tip_py': self.right_end_tip[:, 1],
                          'right_end_tip_pz': self.right_end_tip[:, 2],
                          # Kinematic Features
                           'reach_start': self.behavior_start_time,
                          'reach_duration': self.behavior_duration,
                          'reach_end': self.behavior_end_time,
                          'endpoint_error': self.endpoint_error, 'endpoint_error_x': self.x_endpoint_error,
                          'endpoint_error_y': self.y_endpoint_error, 'endpoint_error_z': self.z_endpoint_error, 'response_sensor': self.exp_response_sensor,
                          # Sensor Data
                          'handle_moving_sensor': self.h_moving_sensor, 'lick_beam': self.lick_vector,
                          'reward_zone': self.reward_zone_sensor, 'time_vector': self.time_vector,
                           'x_rob': self.x_robot, 'y_rob': self.y_robot,
                          'z_rob': self.z_robot}
        if append_outliers:
            self.save_dict.append({'nose_o': self.nose_o, 'handle_o': self.handle_o, 'left_shoulder_o': self.left_shoulder_o,
                          'left_forearm_o': self.left_forearm_o, 'left_wrist_o': self.left_wrist_o,
                          'left_palm_o': self.left_palm_o, 'left_index_base_o': self.left_index_base_o,
                          'left_index_tip_o': self.left_index_tip_o, 'left_middle_base_o': self.left_middle_base_o,
                          'left_middle_tip_o': self.left_middle_tip_o,
                          'left_third_base_o': self.left_third_base_o, 'left_third_tip_o': self.left_third_tip_o,
                          'left_end_base_o': self.left_end_base_o,
                          'left_end_tip_o': self.left_end_tip_o, 'right_shoulder_o': self.right_shoulder_o,
                          'right_forearm_o': self.right_forearm_o, 'right_wrist_o': self.right_wrist_o,
                          'right_palm_o': self.right_palm_o, 'right_index_base_o': self.right_index_base_o,
                          'right_index_tip_o': self.right_index_tip_o,
                          'right_middle_base_o': self.right_middle_base_o,
                          'right_middle_tip_o': self.right_middle_tip_o,
                          'right_third_base_o': self.right_third_base_o,
                          'right_third_tip_o': self.right_third_tip_o,
                          'right_end_base_o': self.right_end_base_o, 'right_end_tip_o': self.right_end_tip_o})
        if append_reach_information:
            self.save_dict.append({'right_start_time': self.right_start_times, 'left_start_time': self.left_start_times,
                          'left_reach_peak': self.left_peak_times, 'right_reach_peak': self.right_peak_times,
                          'left_end_time': self.left_reach_end_times, 'right_end_time': self.right_reach_end_times,
                              'reach_hand': self.arm_id_list, 'error_flag': error_flag})
        df = pd.DataFrame({key: pd.Series(np.asarray(value)) for key, value in self.save_dict.items()})
        df['Trial'] = trial_num
        df.set_index('Trial', append=True, inplace=True)
        df['Date'] = self.date
        df.set_index('Date', append=True, inplace=True)
        df['Session'] = self.session
        df.set_index('Session', append=True, inplace=True)
        df['Rat'] = self.rat
        df.set_index('Rat', append=True, inplace=True)
        # Create CSV file for block
        return df

    def get_preprocessed_trial_blocks(self):
        for ix, sts in enumerate(self.trial_start_vectors):
            self.trial_index = sts
            try:
                stp = self.trial_stop_vectors[ix]
                if stp < sts:
                    stp = sts + 350
            except:
                stp = sts + 600  # bad trial stop information from arduino..use first 3 1/2 seconds
            self.trial_num = int(ix)
            print('Making dataframe for trial:' + str(ix))
            # Obtain values from experimental data
            self.segment_and_filter_kinematic_block(sts, stp)
            self.extract_sensor_data(sts, stp)
            df = self.segment_data_into_reach_dict(ix)
            self.seg_num = 0
            self.start_trial_indice = []
            self.images = []
            if ix == 0:
                self.reaching_dataframe = df
            else:
                self.reaching_dataframe = pd.concat([df, self.reaching_dataframe])
        return self.reaching_dataframe

    def perform_trial_classification(self, df):
        Classifier_var = Classifier(df)
        pdb.set_trace()
        if Classifier_var.null_class_result == 0:
            print('Null Trial')
            self.predicted_trial_type = 0
        else:
            print('Reaching Detected')
            self.predicted_trial_type = 1
            if Classifier_var.num_reaches == 1:
                self.predicted_num_reaches = 1
                if Classifier_var.hand_result == 1:
                    self.single_hand = 'L'
                    print('Left Single Reach')
                else:
                    self.single_hand = 'R'
                    print('Right Single Reach')
            else:
                self.predicted_num_reaches = 2

    def populate_block_predictions(self):
        """ Function to add trialized predictions about our data to a dictionary. This dictionary is saved as a
            csv file, in order to manually annotate possible outliers in the segregation and classification of
            our reaching data. """

        self.prediction_information = {}
        for key in self.predictions_header:
            if 'Trial' in key:
                self.prediction_information[key] = self.trial_num
            if 'Null Prediction' in key:
                self.prediction_information[key] = self.predicted_trial_type
            if 'Num Reaches Prediction' in key:
                self.prediction_information[key] = self.predicted_num_reaches
            if 'Handedness Prediction' in key:
                self.prediction_information[key] = self.single_hand
            if 'Null Outlier Flag' in key:
               self.prediction_information[key].append()
            if 'Num Reaches Outlier Flag' in key:
                self.prediction_information[key].append()
            if 'Handedness Prediction Outlier Flag' in key:
                self.prediction_information[key].append()
            if 'Segmented Reaches' in key:
                self.prediction_information[key] = self.individual_reach_times
            if 'Segmented Null' in key:
                self.prediction_information[key] = len(self.individual_reach_times)
            if 'Segmented Num' in key:
                self.prediction_information[key] = len(self.individual_reach_times)
            if 'Segmented Hand' in key:
                self.prediction_information[key] = self.reach_hand_type
        self.prediction_information_list.append(self.prediction_information)
        #pdb.set_trace()

    def get_reach_dataframe_from_block(self):
        """ Function that obtains a trialized (based on reaching start times)
            pandas dataframe for a provided experimental session. """
        self.total_speeds = []
        self.left_hand_speeds = []
        self.right_hand_speeds = []
        self.left_hand_raw_speeds = []
        self.right_hand_raw_speeds = []
        self.total_outliers = []
        self.calculate_number_of_speed_peaks_in_block()
        for ix, sts in enumerate(self.trial_start_vectors):
            self.trial_index = sts
            try:
                stp = self.trial_stop_vectors[ix]
                if stp < sts:
                    stp = sts + 350
            except:
                stp = sts + 350  # bad trial stop information from arduino..use first 3 1/2 seconds
            self.trial_num = int(ix)
            print('Making dataframe for trial:' + str(ix))
            self.segment_and_filter_kinematic_block(sts, stp)
            self.extract_sensor_data(sts, stp)
            self.segment_reaches_with_speed_peaks()
            df = self.segment_data_into_reach_dict(ix)
            self.perform_trial_classification(df)
            win_length = 90
            error_flag = False
            try:
                if self.behavior_start_time:  # If reach detected
                    end_pad = 1  # length of behavior
                    reach_end_time = self.behavior_end_time + end_pad  # For equally-spaced arrays
                    print(self.behavior_start_time, self.behavior_end_time)
                    if self.lick_vector.any() == 1:  # If trial is rewarded
                        self.first_lick_signal = np.where(self.lick_vector == 1)[0][0]
                        if 5 < self.first_lick_signal < 20:  # If reward is delivered after initial time-out
                            self.segment_and_filter_kinematic_block(sts, sts + 180)
                            self.extract_sensor_data(sts, sts + 180)
                            print('Possible Tug of War Behavior!')
                        else:  # If a reach is detected (successful reach)
                            try:
                                self.segment_and_filter_kinematic_block(
                                    sts + self.behavior_start_time - win_length,
                                    sts + reach_end_time + win_length)
                                self.extract_sensor_data(sts + self.behavior_start_time - win_length,
                                                         sts + reach_end_time + win_length)
                                print('Successful Reach Detected')
                            except:
                                continue
                    else:  # unrewarded reach
                        self.segment_and_filter_kinematic_block(sts + self.behavior_start_time - win_length,
                                                                sts + reach_end_time + win_length)
                        self.extract_sensor_data(sts + self.behavior_start_time - win_length, sts +
                                                 reach_end_time + win_length)
                        print('Un-Rewarded Reach Detected')
                else:  # If reach not detected in trial
                    self.segment_and_filter_kinematic_block(sts - win_length, sts + 200)
                    self.extract_sensor_data(sts - win_length, sts + 200)
                    print('No Reaching Found in Trial Block' + str(ix))
            except:
                self.segment_and_filter_kinematic_block(sts - win_length, sts + 200)
                self.extract_sensor_data(sts - win_length, sts + 200)
                print('Error extracting')
                error_flag = True
            df = self.segment_data_into_reach_dict(ix, error_flag, append_reach_information=True, append_outliers=True)
            self.seg_num = 0
            self.start_trial_indice = []
            self.images = []
            #self.total_block_reaches += self.num_reaches_split
            if ix == 0:
                self.reaching_dataframe = df
            else:
                self.reaching_dataframe = pd.concat([df, self.reaching_dataframe])
            self.populate_block_predictions()
        if self.total_block_reaches - 10 < self.sweep_total_reaches < self.total_block_reaches + 10:
            self.block_outlier_flag = False
        else:
            self.block_outlier_flag = True
        self.create_csv_block_predictions()
        return self.reaching_dataframe

    def save_reaching_dataframe(self, filename):
        self.reaching_dataframe.to_csv(filename)
        return

    def vid_splitter_and_grapher(self, trial_num=0, plot=True, timeseries_plot=True, plot_reach=True, save_data=False):
        """ Function to split and visualize reaching behavior from a given experimental session. """
        for ix, sts in enumerate(self.trial_start_vectors):
            self.trial_num = ix
            self.trial_index = sts
            if ix > trial_num:
                self.fps = 30
                self.trial_num = int(ix)
                bi = self.block_video_path.rsplit('.')[0]
                self.sstr = bi + '/trial' + str(ix)
                self.make_paths()
                self.clip_path = self.sstr + '/videos/trial_video.mp4'
                try:
                    stp = self.trial_stop_vectors[ix]
                except:
                    stp = sts + 500
                self.split_trial_video(sts, stp)
                print('Split Trial' + str(ix) + ' Video')
                self.segment_and_filter_kinematic_block(sts, stp)
                self.extract_sensor_data(sts, stp)
                if plot:
                    self.plot_predictions_videos(plot_digits=False)
                    self.make_gif_reaching()
                    self.images = []
                    self.make_combined_gif()
                    print('GIF MADE for  ' + str(ix))
                print('Finished Plotting!   ' + str(ix))
                self.segment_reaches_with_speed_peaks()
                win_length = 200  # for non-reaching trials
                cut_reach_start_time = self.behavior_start_time
                # Create dataframe to plot for "reach" segmentation
                print(self.behavior_start_time, self.behavior_end_time, self.retract_end, self.num_peaks, self.behavior_duration)
                if self.behavior_start_time is not None and self.trial_type is not False:
                    if self.retract_end is not None and self.handle_moved is not False and self.trial_type is not False and self.tug_flag is False:
                        print('Successful Reach')
                        if cut_reach_start_time - 30 < 0:  # are we close to the start of the trial?
                            cut_reach_start_time = 0  # spacer to keep array values
                        self.split_trial_video(cut_reach_start_time + self.trial_index, self.trial_index +
                                               self.retract_end, segment=True, num_reach=0, low_fps=True)
                        self.segment_and_filter_kinematic_block(
                            cut_reach_start_time + self.trial_index,
                            self.trial_index + self.retract_end)
                        self.extract_sensor_data(self.trial_index + cut_reach_start_time, self.trial_index +
                                                 self.retract_end)
                    elif self.tug_flag is True:  # No reaching but handle movement during trial
                        print ('Tug of War Detected')
                        self.split_trial_video(self.trial_index, self.trial_index + 100,
                                               segment=True, num_reach=0, low_fps=True)
                        self.segment_and_filter_kinematic_block(self.trial_index,
                                                                self.trial_index+100)
                        self.extract_sensor_data(self.trial_index, self.trial_index + 100)
                    else:
                        print('Unsuccessful Reaching')
                        if self.behavior_start_time - 30 < 0:  # are we close to the start of the trial?
                            cut_reach_start_time = 0  # spacer to keep array values
                        self.split_trial_video(cut_reach_start_time + self.trial_index,
                                               self.trial_index + self.behavior_end_time + 40,
                                               segment=True, num_reach=0, low_fps=True)
                        self.segment_and_filter_kinematic_block(
                            cut_reach_start_time + self.trial_index,
                            self.trial_index + self.behavior_end_time + 40)
                        self.extract_sensor_data(cut_reach_start_time + self.trial_index,
                                                 self.trial_index + self.behavior_end_time + 40)
                else:
                        print('No Reach Detected.')
                        self.split_trial_video(self.trial_index, self.trial_index + win_length, segment=True,
                                               num_reach=0, low_fps=True)
                        self.segment_and_filter_kinematic_block(self.trial_index, self.trial_index + win_length)
                        self.extract_sensor_data(self.trial_index, self.trial_index + win_length)
                if timeseries_plot:
                    #self.plot_interpolation_variables_palm('reach')
                    self.plot_timeseries_features()
                if plot_reach:
                    self.plot_predictions_videos(segment=True)
                    self.make_gif_reaching(segment=True, nr=0)
                    print('Reaching GIF made for reach ' + str(0) + 'in trial ' + str(ix))
                    self.images = []
                    self.make_combined_gif(segment=True, nr=0, )
                try:
                    self.total_block_reaches += self.num_reaches_split
                except:
                    pass
                self.populate_block_predictions()
            self.seg_num = 0
            self.start_trial_indice = []
            self.images = []
            self.tug_flag = False
        self.write_reach_information_to_csv()
        if save_data:
            reaching_trial_indices_df = pd.DataFrame(self.block_cut_vector)
            reaching_trial_indices_df.to_csv(
                'reaching_vector_' + str(self.rat) + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                index=False, header=False)
            print('Saved reaching indices and bouts for block..' + str(self.date) + str(self.session))
        return self.block_cut_vector

    def plot_interpolation_variables_palm(self, filtype):
        """ Plots displaying feature variables for the left and right palms. """
        filename_pos = self.sstr + '/timeseries/' + str(filtype) + 'interpolation_timeseries.png'
        times = np.around((self.time_vector - self.time_vector[0]), 2)
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, figsize=(15, 28))
        times_mask_left = np.around((np.asarray(self.time_vector)[self.left_palm_f_x] - self.time_vector[0]), 2)
        times_mask_right = np.around((np.asarray(self.time_vector)[self.right_palm_f_x] - self.time_vector[0]), 2)
        left_outliers = np.zeros(self.left_palm_p.shape[0])
        left_outliers[self.left_palm_o] = 1
        right_outliers = np.zeros(self.right_palm_p.shape[0])
        right_outliers[self.right_palm_o] = 1
        ax1.set_title('Left Palm')
        ax2.set_title('Right Palm')
        ax3.set_title('Probabilities and Experimental Features')
        ax4.set_title('Speeds')
        ax1.plot(times, self.uninterpolated_left_palm[:, 0], color='r', linestyle='solid',
                 label='Left Palm Pre-Outlier Removal: X')
        ax1.plot(times, self.uninterpolated_left_palm[:, 1], color='darkred', linestyle='solid',
                 label='Left Palm Pre-Outlier Removal: Y')
        ax1.plot(times, self.uninterpolated_left_palm[:, 2], color='salmon', linestyle='solid',
                 label='Left Palm Pre-Outlier Removal: Z')
        ax2.plot(times, self.uninterpolated_right_palm[:, 0], color='navy', linestyle='solid',
                 label='Right Palm Pre-Outlier Removal: x')
        ax2.plot(times, self.uninterpolated_right_palm[:, 1], color='b', linestyle='solid',
                 label='Right Palm Pre-Outlier Removal: Y')
        ax2.plot(times, self.uninterpolated_right_palm[:, 2], color='c', linestyle='solid',
                 label='Right Palm Pre-Outlier Removal: Z')
        ax1.set_xlabel('Time (s) ')
        ax1.set_ylabel('Distance (M) ')
        ax2.set_ylim(0, 0.5)
        ax1.set_ylim(0, 0.5)
        ax3.set_ylim(0, 0.5)
        ax1.plot(times[2:-1], self.left_palm[2:-1, 0], color='r', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Left Palm X')
        ax1.plot(times[2:-1], self.left_palm[2:-1, 1], color='darkred', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Left Palm Y')
        ax1.plot(times[2:-1], self.left_palm[2:-1, 2], color='salmon', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Left Palm Z')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 0], color='navy', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Right Palm X')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 1], color='b', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Right Palm Y')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 2], color='c', linestyle='dashed',
                 label='Post-Interpolation/Smoothing: Right Palm Z')
        # Scatter Plot Interpolation Points
        ax1.scatter(times_mask_left, self.left_palm[:, 0][self.left_palm_f_x], color='m', alpha=0.3, linestyle='dashed',
                    label='Interpolation Intervals')
        ax1.scatter(times_mask_left, self.left_palm[:, 1][self.left_palm_f_x], color='m', alpha=0.3, linestyle='dashed')
        ax1.scatter(times_mask_left, self.left_palm[:, 2][self.left_palm_f_x], color='m', alpha=0.3, linestyle='dashed')
        ax2.scatter(times_mask_right, self.right_palm[:, 0][self.right_palm_f_x], color='m', alpha=0.3,
                    linestyle='dashed',
                    label='Interpolation Intervals')
        ax2.scatter(times_mask_right, self.right_palm[:, 1][self.right_palm_f_x], color='m', alpha=0.3,
                    linestyle='dashed')
        ax2.scatter(times_mask_right, self.right_palm[:, 2][self.right_palm_f_x], color='m', alpha=0.3,
                    linestyle='dashed')
        try:
            if self.right_start_times:
                for tsi, segment_trials in enumerate(self.right_start_times):
                    ax2.plot(times[segment_trials], self.right_palm_s[segment_trials], marker='*', color='black',
                             markersize=20, label='Reach Start')
                    ax4.plot(times[segment_trials], self.right_palm_s[segment_trials], marker='*', color='black',
                             markersize=20, label='Reach Start')
            if self.left_start_times:
                for tsi, segment_trials in enumerate(self.left_start_times):
                    ax2.plot(times[segment_trials], self.left_palm_s[segment_trials], marker='*', color='black',
                             markersize=20,
                             label='Reach Start')
                    ax4.plot(times[segment_trials], self.left_palm_s[segment_trials], marker='*', color='black',
                             markersize=20,
                             label='Reach Start')
            if self.right_peak_times:
                for ti, seg_tr in enumerate(self.right_peak_times):
                    ax4.plot(times[seg_tr], self.right_palm_s[seg_tr], color='orange', marker='*', markersize=30,
                             label='Peak Reach Right')
            if self.left_peak_times:
                for ti, seg_tr in enumerate(self.left_peak_times):
                    ax4.plot(times[seg_tr], self.left_palm_s[seg_tr], color='orange', marker='*', markersize=30,
                             label='Peak Reach Left')
            if self.left_reach_end_times:
                for ti, seg_tr in enumerate(self.left_reach_end_times):
                    ax4.plot(times[seg_tr], self.left_palm_s[seg_tr], color='m', marker='*', markersize=20,
                             label='End Reach Left Palm')
            if self.right_reach_end_times:
                for ti, seg_tr in enumerate(self.right_reach_end_times):
                    ax4.plot(times[seg_tr], self.right_palm_s[seg_tr], color='m', marker='*', markersize=20,
                             label='End Reach Right Palm')
        except:
            pass
        ax2.set_xlabel('Time (s) ')
        ax2.set_ylabel('Distance (M) ')
        ax1.legend(loc=1, fontsize='small')
        ax2.legend(loc=1, fontsize='small')
        ax3.plot(times, self.left_palm_p, color='r', label='Left Palm Mean Probability')
        ax3.plot(times, self.right_palm_p, color='b', label='Right Palm Mean Probability')
        ax3.plot(times, self.nose_p, color='m', label='Location Probability')
        ax3.plot(times, self.lick_vector / 10, color='y', label='Licks Occurring')
        ax4.plot(times, self.handle_s, color='b', label='Handle Speed')
        ax4.plot(times, self.left_palm_s, color='r', label='Left Palm Speed')
        ax4.plot(times, self.right_palm_s, color='g', label='Right Palm Speed')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Probabilities')
        ax3.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (M)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylim(0, 2)
        ax3.legend(loc=5, fontsize='small')
        ax4.legend(loc=5, fontsize='small')
        plt.savefig(filename_pos)
        plt.close('all')
        return

    def plot_timeseries_features(self):
        """ Function to plot diagnostic time-series plots from single-trial data. """
        filename_pc = self.sstr + '/timeseries/' + 'pc_timeseries.png'
        filename_pos = self.sstr + '/timeseries/' + 'full_pos_timeseries.png'
        filename_vel = self.sstr + '/timeseries/' + 'full_vel_timeseries.png'
        filename_speed = self.sstr + '/timeseries/' + 'full_speed_timeseries.png'
        xyzpalms = self.sstr + '/timeseries/' + 'xyzpalms_timeseries.png'
        rightvsleftxy = self.sstr + '/timeseries/' + 'rlxypalms_timeseries.png'
        plt.figure(figsize=(10,10))
        axel0 = plt.axes(projection="3d")
        frames_n = np.around(self.time_vector, 2)
        frames = frames_n - frames_n[0]  # normalize frame values by first entry.
        plt.title('Principal Components of Kinematics for Each Arm')
        plt.plot(self.left_PCS[:, 0], self.left_PCS[:, 1],self.left_PCS[:,2], c='r', label='Left PCS')
        plt.plot(self.right_PCS[:, 0], self.right_PCS[:, 1],self.right_PCS[:, 2], c='b', label='Right PCS')
        axel0.legend()
        plt.savefig(filename_pc)
        axel = plt.figure(figsize=(10, 4))
        plt.title('Palm Positions During Trial')
        plt.plot(frames, self.right_palm[:, 0], c='g', label='X Right Palm')
        plt.plot(frames, self.right_palm[:, 1], c='g', linestyle='dashed', label='Y Right Palm')
        plt.plot(frames, self.right_palm[:, 2], c='g', linestyle='dotted', label='Z Right Palm')
        plt.plot(frames, self.left_palm[:, 0], c='r', label='X Left Palm')
        plt.plot(frames, self.left_palm[:, 1], c='r', linestyle='dashed', label='Y Left Palm')
        plt.plot(frames, self.left_palm[:, 2], c='r', linestyle='dotted', label='Z Left Palm')
        plt.plot(frames, self.lick_vector / 10, color='y', label='Licks Occurring')
        if self.behavior_start_time is not None:
            try:
                for idx, reaches in enumerate(self.individual_reach_times):
                        end_time = self.individual_reach_end_times[idx]
                        peak_time = self.individual_reach_peak_times[idx]
                        plt.axvline(frames[reaches - self.behavior_start_time], color='b', label='Reach ' + str(idx) + ' Start')
                        plt.axvline(frames[peak_time - self.behavior_start_time], color='m', label ='Peak' + str(idx))
                        plt.axvline(frames[end_time - self.behavior_start_time], color='c', label='Reach' + str(idx) + 'end')
            except:
                pass
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        #axel.tight_layout(pad=0.005)
        axel.legend()
        axel.savefig(filename_pos, bbox_inches='tight', dpi=800)
        plt.close()
        axel1 = plt.figure(figsize=(10, 4))
        plt.title('Palm Velocities During Trial')
        plt.plot(frames, self.right_palm_v[:, 0], c='g', label='X Right Palm')
        plt.plot(frames, self.right_palm_v[:, 1], c='g', linestyle='dashed', label='Y Right Palm')
        plt.plot(frames, self.right_palm_v[:, 2], c='g', linestyle='dotted', label='Z Right Palm')
        plt.plot(frames, self.left_palm_v[:, 0], c='r', label='X Left Palm')
        plt.plot(frames, self.left_palm_v[:, 1], c='r', linestyle='dashed', label='Y Left Palm')
        plt.plot(frames, self.left_palm_v[:, 2], c='r', linestyle='dotted', label='Z Left Palm')
        plt.ylim(-1, 1.6)
        if self.behavior_start_time is not None:
            try:
                for idx, reaches in enumerate(self.individual_reach_times):
                        end_time = self.individual_reach_end_times[idx]
                        peak_time = self.individual_reach_peak_times[idx]
                        plt.axvline(frames[reaches - self.behavior_start_time], color='b', label='Reach ' + str(idx) + ' Start')
                        plt.axvline(frames[peak_time - self.behavior_start_time], color='m', label ='Peak' + str(idx))
                        plt.axvline(frames[end_time - self.behavior_start_time], color='c', label ='Reach' + str(idx) + 'end')
            except:
                pass
        plt.xlabel('Time (s)')
        plt.ylabel(' Palm Velocity (m/s)')
        axel1.legend()
        #axel1.tight_layout(pad=0.005)
        # pdb.set_trace()
        axel1.savefig(filename_vel, bbox_inches='tight', dpi=800)
        axel2 = plt.figure(figsize=(10, 4))
        plt.title('Speeds (Arm, Palm) During Trial')
        plt.plot(frames, self.right_palm_s, c='b', label='Right palm speed')
        plt.plot(frames, self.left_palm_s, c='r', label='Left palm speed')
        plt.plot(frames, self.right_forearm_s, c='c', linestyle='dashed', label='Right arm speed')
        plt.plot(frames, self.left_forearm_s, c='pink', linestyle='dashed', label='Left arm speed')
        plt.ylim(0, 1.6)
        plt.xlabel('Time (s) ')
        plt.ylabel('Speed (m/s)')
        if self.behavior_start_time is not None:
            try:
                for idx, reaches in enumerate(self.individual_reach_times):
                        end_time = self.individual_reach_end_times[idx]
                        peak_time = self.individual_reach_peak_times[idx]
                        plt.axvline(frames[reaches - self.behavior_start_time], color='b', label='Reach ' + str(idx) + ' Start')
                        plt.axvline(frames[peak_time - self.behavior_start_time], color='m', label ='Peak' + str(idx))
                        plt.axvline(frames[end_time - self.behavior_start_time], color='c', label='Reach' + str(idx) +
                                                                                               'End')
            except:
                pass
        axel2.legend()
        #axel2.tight_layout(pad=0.0005)
        axel2.savefig(filename_speed, bbox_inches='tight', dpi=800)

        # ap = np.mean([np.mean(self.prob_right_arm, axis=1), np.mean(self.prob_left_arm, axis=1)], axis=0)
        # print(ap.shape)
        axel3 = plt.figure(figsize=(8, 8))
        plt.title(' R vs L X Y Z Plots')
        plt.plot(self.right_palm[:, 0], self.left_palm[:, 0], c='r', label='Palm Vs Palm X')
        plt.plot(self.right_palm[:, 1], self.left_palm[:, 1], c='g', label='Palm Vs Palm Y')
        plt.plot(self.right_palm[:, 2], self.left_palm[:, 2], c='b', label='Palm Vs Palm Z')
        plt.xlabel('M')
        plt.ylabel('M')
        axel3.legend()
        #axel3.tight_layout(pad=0.0005)
        axel3.savefig(xyzpalms, bbox_inches='tight', dpi=800)
        axel4 = plt.figure(figsize=(8, 8))
        plt.title(' R vs L X Y Z Plots')
        plt.plot(self.right_palm[:, 0], self.right_palm[:, 1], c='r', label='Right Palm')
        plt.plot(self.left_palm[:, 0], self.left_palm[:, 1], c='g', label='Left Palm')
        plt.xlabel('M')
        plt.ylabel('M')
        axel4.legend()
        #axel4.tight_layout(pad=0.0005)
        axel4.savefig(rightvsleftxy, bbox_inches='tight', dpi=800)
        plt.close('all')
        self.plot_palm_spectrograms()
        return

    def plot_palm_spectrograms(self):
        filename_specgram = self.sstr + 'specgram.png'
        plt.subplot(211)
        plt.title('Right Palm Speed Spectrogram')
        plt.specgram(self.right_palm_s, Fs=180/2)
        if self.behavior_start_time is not None:
            try:
                for idx, reaches in enumerate(self.individual_reach_times):
                        end_time = self.individual_reach_end_times[idx]
                        peak_time = self.individual_reach_peak_times[idx]
                        plt.axvline(frames[reaches - self.behavior_start_time], color='b', label='Reach ' + str(idx) + ' Start')
                        plt.axvline(frames[peak_time - self.behavior_start_time], color='m', label ='Peak' + str(idx))
                        plt.axvline(frames[end_time - self.behavior_start_time], color ='c')
            except:
                pass
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.subplot(212)
        plt.title('Left Palm Speed Spectrogram')
        plt.specgram(self.left_palm_s, Fs=180/2)
        if self.behavior_start_time is not None:
            try:
                for idx, reaches in enumerate(self.individual_reach_times):
                        end_time = self.individual_reach_end_times[idx]
                        peak_time = self.individual_reach_peak_times[idx]
                        plt.axvline(frames[reaches - self.behavior_start_time], color='b', label='Reach ' + str(idx) + ' Start')
                        plt.axvline(frames[peak_time - self.behavior_start_time], color='m', label ='Peak' + str(idx))
                        plt.axvline(frames[end_time - self.behavior_start_time], color='o')
            except:
                pass
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(filename_specgram, dpi=900)
        plt.clf()

    def set_lag_plot_term(self, isx):
        """ Function to set "lag", or additional frames to plot. """
        if isx == 1:
            self.lag = 1
        if isx == 2:
            self.lag = 2
        if isx == 3:
            self.lag = 3
        if isx == 4:
            self.lag = 4
        return

    def plot_predictions_videos(self, segment=False, clean=False):
        """ Function to plot, frame by frame, the full 3-D reaching behavior over time. """
        tldr = self.left_palm.shape[0]  # set range in single trial
        tlde = 0
        frames_n = np.around(self.time_vector, 2)
        frames = frames_n - frames_n[0]  # normalize frame values by first entry.
        # Cut off long values for smaller reaching times
        for isx in range(tlde, tldr):
            self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            self.set_lag_plot_term(isx)
            fixit1 = plt.figure(figsize=(28,28/2))
            axel = plt.subplot(311)
            axel1 = plt.subplot(312, sharex=axel)
            axel2 = plt.subplot(313, sharex=axel)
            plt.subplots_adjust(top=0.95)
            plt.tick_params('x', labelbottom=False) # first plot don't plot x-axis
            # Plot sensor information (binary)
            axel.plot(frames[0:isx], self.reward_zone_sensor[0:isx], label='Reward Zone')
            axel.plot(frames[0:isx], self.lick_vector[0:isx], label='Licking Activity')
            if self.behavior_start_time is not None:
                for idx, reaches in enumerate(self.individual_reach_times):
                    if reaches-self.behavior_start_time <= isx:
                        axel1.axvline(frames[reaches - self.behavior_start_time], color='k',
                                    label='Reach ' + str(idx) + ' Start')
                        axel2.axvline(frames[reaches - self.behavior_start_time], color='k',
                                      label='Reach ' + str(idx) + ' Start')
                    end_time = self.individual_reach_end_times[idx]
                    if end_time - self.behavior_start_time <= isx:
                        axel1.axvline(frames[end_time - self.behavior_start_time], color='k',
                                    label='Reach' + str(idx) + 'end')
                        axel2.axvline(frames[end_time - self.behavior_start_time], color='k',
                                      label='Reach' + str(idx) + 'end')
                    peak_time = self.individual_reach_peak_times[idx]
                    if peak_time - self.behavior_start_time <= isx:
                        axel1.axvline(frames[peak_time - self.behavior_start_time], color='gold', label='Peak' + str(idx))
                        axel2.axvline(frames[peak_time - self.behavior_start_time], color='gold',
                                      label='Peak' + str(idx))
            axel.set_ylabel('On/Off')
            # Plot kinematic data
            axel2.plot(frames[0:isx], self.handle_s[0:isx], c='y', label='Handle Speed')
            axel2.plot(frames[0:isx], self.left_palm_s[0:isx], marker='.', c='red', label='Left Palm Speed')
            axel2.plot(frames[0:isx], self.right_palm_s[0:isx], marker='.', c='g', label='Right Palm Speed')
            axel1.plot(frames[0:isx], self.left_wrist_s[0:isx], marker='.', c='salmon', label='Left Wrist Speed')
            axel1.plot(frames[0:isx], self.right_wrist_s[0:isx], marker='.', c='springgreen', label='Right Wrist Speed')
            axel1.plot(frames[0:isx], self.left_index_tip_s[0:isx], marker='.', c='r', label='Left Index Speed')
            axel1.plot(frames[0:isx], self.right_index_tip_s[0:isx], marker='.', c='g', label='Right Index Speed')
            axel2.set_xlabel('Time from trial onset (s)')
            axel2.set_ylabel('m/ s')
            axel2.set_ylim(0, 1.2)
            axel1.set_xlabel('Time from trial onset (s)')
            axel1.set_ylabel('m/ s')
            axel1.set_ylim(0, 1.2)
            axel.set_ylim(0,1)
            axel2.set_xlim(0, frames[-1])
            # axel2.set_zlim3d(0.35,0.5)
            fixit1.tight_layout(pad=0.005)
            plt.margins(0.0005)
            axel2.legend(loc="upper right", fontsize="x-large")
            axel.legend(loc="upper right", fontsize="x-large")
            axel1.legend(loc="upper right", fontsize="x-large")
            if segment:
                self.filename = self.sstr + '/plots/reaches/' + str(isx) + 'palm_prob_timeseries.png'
            else:
                self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            plt.savefig(self.filename, bbox_inches='tight', pad_inches=0.00001)
            plt.close('all')
            self.images.append(imageio.imread(self.filename))
            if clean:
                os.remove(self.filename)
        return

    def make_gif_reaching(self, segment=False, nr=0):
        """ Function to make a per-trial GIF from the compilation of diagnostic plots made using ReachLoader. """
        if segment:
            imageio.mimsave(self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4', self.images,
                            fps=10)
            self.gif_save_path = self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4'
        else:
            imageio.mimsave(self.sstr + '/videos/total_3d_movie.mp4', self.images, fps=10)
            self.gif_save_path = self.sstr + '/videos/total_3d_movie.mp4'
        self.images = []  # Clear memory
        return

    def make_combined_gif(self, segment=False, nr=0):
        """ Function to create a combined video and diagnostic graph GIF per-trial. """
        vid_gif = skvideo.io.vread(self.clip_path)
        plot_gif = skvideo.io.vread(self.gif_save_path)
        number_of_frames = int(min(vid_gif.shape[0], plot_gif.shape[0]))
        if segment:
            writer = imageio.get_writer(self.sstr + '/videos/reaches/reach_' + str(nr) + 'split_movie.mp4',
                                        fps=10)
        else:
            writer = imageio.get_writer(self.sstr + '/videos/total_split_movie.mp4', fps=10)
        for frame_number in range(number_of_frames):
            img1 = np.squeeze(vid_gif[frame_number, :, :, :])
            img2 = np.squeeze(plot_gif[frame_number, :, :, :])
            if img2.shape[1] >= img1.shape[1]:
                img2 = np.copy(img2[0:img1.shape[0],0:img1.shape[1], :])
            else:
                img1 = np.copy(img1[0:img2.shape[0], 0:img2.shape[1], :])
            try:
                new_image = np.vstack((img1, img2))
            except:
                pdb.set_trace()
            writer.append_data(new_image)
        writer.close()
        return

    def split_trial_video(self, start_frame, stop_frame, num_reach=0, segment=False, low_fps = False):
        """ Function to split and save the trial video per trial."""
        vc = cv2.VideoCapture(self.block_video_path)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        nfr = vu.rescale_frame(frame)
        (h, w) = nfr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if segment:
            self.clip_path = self.sstr + '/videos/reaches/reach_' + str(num_reach) + '_trial.mp4'
        if low_fps:
            out = cv2.VideoWriter(self.clip_path, fourcc, 10, (w, h))
        else:
            out = cv2.VideoWriter(self.clip_path, fourcc, self.fps, (w, h))
        c = 0
        word_add = 'No Reaching'
        while rval: # Write text here for loop over reaches, peaks etc.
            rval, frame = vc.read()
            if segment:
                if c >= start_frame:
                    if self.individual_reach_times:
                        for irx, reach in enumerate(self.individual_reach_times):
                            # subtract the reach start (reach) from individual, peak, end time values
                            if c >= 0:
                                try:
                                    if reach-reach <= c < reach - reach + start_frame + 2:
                                        word_add = 'Reach' + str(irx) + str(self.reach_hand_type[irx] + ' Start')
                                    elif reach-reach + start_frame + 2 < c <= self.individual_reach_peak_times[irx]- reach + start_frame-1:
                                        word_add = 'Reach ' + str(irx) + str(self.reach_hand_type[irx])
                                    elif self.individual_reach_peak_times[irx]- reach + start_frame - 1 <= c <= self.individual_reach_peak_times[irx] - reach + start_frame + 3:
                                        word_add = 'Reach' + str(irx) + str(self.reach_hand_type[irx] + ' Peak' )
                                    elif self.individual_reach_peak_times[irx] - reach + start_frame + 3 <= c < self.individual_reach_end_times[irx] + start_frame - reach - 5:
                                        word_add = 'Reach' + str(irx) + str(self.reach_hand_type[irx])
                                    elif self.individual_reach_end_times[irx] + start_frame - reach - 2 <= c <= self.individual_reach_end_times[irx] - reach + start_frame:
                                        word_add = 'Reach' + str(irx) + str(self.reach_hand_type[irx] + ' End' )
                                    elif self.individual_reach_end_times[irx] + start_frame - reach - 1  <= c <= self.individual_grasp_times[irx] + start_frame - reach + 3:
                                        word_add = 'Reach' + str(irx) + str(self.reach_hand_type[irx] + ' Grasp' )
                                    elif self.individual_retract_times + start_frame - reach + 4 <= c <= self.retract_end + start_frame - reach +2:
                                        word_add = 'Reach ' + str(irx)  + str(self.reach_hand_type[irx] + ' Retract')
                                    elif self.tug_flag:
                                        word_add = 'Tug of War'
                                    else:
                                        word_add = 'No Reaching'
                                except:
                                    pass

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, word_add, (50, 50), font, 2, (255, 255, 0), 2)
            nfr = vu.rescale_frame(frame)
            if start_frame <= c < stop_frame:
                out.write(nfr)
            if c == stop_frame:
                break
            c += 1
        vc.release()
        out.release()
        return

    def split_videos_into_reaches(self):
        bi = self.block_video_path.rsplit('.')[0]
        self.sstr = bi + '/reaching_split_trials'
        for ir, r in enumerate(self.right_start_times):
            print('Splitting video at' + str(r))
            self.split_trial_video(r - 5, self.right_reach_end_time + 5, segment=True, num_reach=ir)
        for ir, r in enumerate(self.left_start_times):
            print('Splitting video at' + str(r))
            self.split_trial_video(r - 5, self.left_reach_end_time + 5, segment=True, num_reach=ir + 3000)
        return
