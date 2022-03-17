from sklearn.decomposition import PCA
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import DataStream_Vis_Utils as utils
# import DataStream_Vis_Utils as utils
from moviepy.editor import *
import skvideo
import cv2
import imageio
import numpy as np
import viz_utils as vu
import scipy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
# from software.ReachAnalysis.DataLoader import DataLoader as DL
import pdb

# Import ffmpeg to write videos here..

ffm_path = 'C:/Users/bassp/OneDrive/Desktop/ffmpeg/bin/'
skvideo.setFFmpegPath(ffm_path)
import skvideo.io


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


def get_principle_components(positions, vel=0, acc=0, num_pcs=10):
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
    explained_variance_ratio = pca.explained_variance_ratio_
    return pc_vector, explained_variance_ratio


def gkern(input_vector, sig=1.0):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`. Filters N-D vector.
    """
    resulting_vector = gaussian_filter1d(input_vector, sig, mode='mirror')
    return resulting_vector


# noinspection PyTypeChecker,PyBroadException
class ReachViz:
    # noinspection SpellCheckingInspection
    def __init__(self, date, session, data_path, block_vid_file, kin_path, rat):
        self.endpoint_error, self.x_endpoint_error, self.y_endpoint_error, self.z_endpoint_error = 0, 0, 0, 0
        self.preprocessed_rmse, self.outlier_list, self.transformation_matrix = [], [], []
        self.probabilities, self.bi_reach_vector, self.trial_index, self.first_lick_signal, self.outlier_indexes = \
            [], [], [], [], []
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
        self.arm_id_list = []
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
        self.trial_rewarded = False
        self.filename = None
        self.total_outliers, self.rewarded, self.left_palm_f_x, self.right_palm_f_x = [], [], [], []
        self.total_raw_speeds, self.total_preprocessed_speeds, self.total_probabilities = [], [], []
        self.reach_start_time = []
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
        self.prob_list, self.pos_list, self.interpolation_rmse_list, self.valid_rmse_list, \
        self.outlier_rmse_list = [], [], [], [], []
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

    def align_workspace_coordinates_session(self, plot_handle_transformation=False):
        """ Function to align our potentiometer-based coordinate system with our DeepLabCut predictions
            coordinate system"""
        x_int = 0.15
        y_int = 0.15
        z_int = 0.4
        self.handle = np.mean(
            [self.kinematic_block[self.kinematic_block.columns[0:3]].values[100:500, :],
             self.kinematic_block[self.kinematic_block.columns[3:6]].values[100:500, :]], axis=0)
        # Take robot data from entire block
        self.extract_sensor_data(100, 500, check_lick=False)
        df_matrix = np.vstack([self.x_robot, self.y_robot, self.z_robot]).T
        # Use input data to generate linear transformation matrix between DLC handle coordinates and pot coordinates
        self.handle[:, 0] = self.handle[:, 0] + x_int
        self.handle[:, 1] = self.handle[:, 1] + y_int
        self.handle[:, 2] = self.handle[:, 2] + z_int
        self.transformation_matrix = find_linear_transformation_between_DLC_and_robot(self.handle, df_matrix)
        if plot_handle_transformation:
            handle_transformed = apply_linear_transformation_to_DLC_data(self.handle, self.transformation_matrix)
            plot_session_alignment(self.handle[:, 0], df_matrix[:, 0], handle_transformed[:, 0])
            plot_session_alignment(self.handle[:, 1], df_matrix[:, 1], handle_transformed[:, 1], dim='y')
            plot_session_alignment(self.handle[:, 2], df_matrix[:, 2], handle_transformed[:, 2], dim='z')
        return self.transformation_matrix

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
        # pdb.set_trace()
        if check_lick:
            self.check_licking_times_and_make_vector()
        if filter:
            try:
                self.h_moving_sensor[self.prob_filter_index] = 0
                self.reward_zone_sensor[self.prob_filter_index] = 0
                self.lick[self.prob_filter_index] = 0
                self.exp_response_sensor[self.prob_filter_index] = 0
                self.x_robot[self.prob_filter_index] = 0
                self.y_robot[self.prob_filter_index] = 0
                self.z_robot[self.prob_filter_index] = 0
            except:  # all the values are filtered by nose_probability (ie rat isn't there..)
                pass
        self.sensor_data_list = [self.h_moving_sensor, self.lick_vector, self.reward_zone_sensor,
                                 self.exp_response_sensor,
                                 self.x_robot, self.y_robot, self.z_robot]
        return

    def calculate_number_of_speed_peaks_in_block(self):
        left_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:21]].values[0:-1, :], self.transformation_matrix)
        right_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[0:-1, :], self.transformation_matrix)
        self.time_vector = list(right_palm[0, :])
        left_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values[0:-1, :],
                              axis=1)
        right_palm_p = np.mean(self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[0:-1, :],
                               axis=1)
        left_palm_f = vu.cubic_spline_smoothing(np.copy(left_palm), spline_coeff=0.1)
        right_palm_f = vu.cubic_spline_smoothing(np.copy(right_palm), spline_coeff=0.1)
        # If palms are < 0.8 p-value, remove chance at "maxima"
        left_palm_prob = np.where(left_palm_p < 0.5)[0]
        right_palm_prob = np.where(right_palm_p < 0.5)[0]
        # If palms are > 0.21m in the x-direction towards the handle 0 position.
        left_palm_f[left_palm_prob, 0] = 0
        right_palm_f[right_palm_prob, 0] = 0
        right_palm_maxima = find_peaks(right_palm_f[:, 0], height=0.265, distance=8)[0]
        left_palm_maxima = find_peaks(left_palm_f[:, 0], height=0.265, distance=8)[0]
        num_peaks = len(right_palm_maxima) + len(left_palm_maxima)
        print("Number of tentative reaching actions detected:  " + str(num_peaks))
        return num_peaks

    def segment_and_filter_kinematic_block(self, cl1, cl2, p_thresh=0.4, coarse_threshold=0.4,
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
        nose = vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[6:9]].values[cl1:cl2, :],
                                   aff_t=self.transformation_matrix)
        handle = np.mean(
            [vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[0:3]].values[cl1:cl2, :],
                                 aff_t=self.transformation_matrix),
             vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[3:6]].values[cl1:cl2, :],
                                 aff_t=self.transformation_matrix
                                 )], axis=0)
        left_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[9:12]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 21 end
        right_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 57 end
        left_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 21 end
        right_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 57 end
        left_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[15:18]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 21 end
        right_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)  # 57 end
        left_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:21]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        # Digits, optional for now
        right_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[27:30]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[30:33]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[36:39]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[39:42]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[42:45]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        right_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[57:60]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[60:63]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[66:69]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[69:72]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[72:75]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[75:78]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
        left_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[78:81]].values[cl1:cl2, :],
            aff_t=self.transformation_matrix)
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
        self.pos_holder = []
        self.vel_holder = []
        self.speed_holder = []
        self.acc_holder = []
        self.raw_speeds = []
        # Pre-process (threshold, interpolate if necessary/possible, and apply hamming filter post-interpolation
        if preprocess:
            self.preprocess_kinematics(p_thresh=0.1)
        self.assign_final_variables()
        # Calculate principal components
        self.pos_v_a_pc, self.pos_v_a_pc_variance = get_principle_components(self.pos_holder, vel=self.vel_holder,
                                                                             acc=self.acc_holder)
        self.left_arm_pc_pos_v_a, self.left_arm_pos_v_a_pc_variance = get_principle_components(self.pos_holder[2:14],
                                                                                               vel=self.vel_holder[
                                                                                                   2:14],
                                                                                               acc=self.acc_holder[
                                                                                                   2:14])
        self.right_arm_pc_pos_v_a, self.right_arm_pos_v_a_pc_variance = get_principle_components(self.pos_holder[14:-1],
                                                                                                 vel=self.vel_holder[
                                                                                                     14:-1],
                                                                                                 acc=self.acc_holder[
                                                                                                     14:-1])
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
            svd, acc, speed_c = self.calculate_kinematics_from_position(np.copy(pos), spline=False)
            self.raw_speeds.append(speed_c)
            v_outlier_index = np.where(svd > 2)
            possi, num_int, gap_ind = vu.interpolate_3d_vector(np.copy(o_positions), v_outlier_index, prob_outliers)
            filtered_pos = vu.cubic_spline_smoothing(np.copy(possi), spline_coeff=spline)
            v, a, s = self.calculate_kinematics_from_position(np.copy(filtered_pos), spline=True)
            # Find and save still-present outliers in the data
            velocity_outlier_indexes = np.where(s > 2)[0]
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

    def calculate_kinematics_from_position(self, pos_v, spline=False):
        """ Function that calculates velocity, speed, and acceleration on a per-bodypart basis."""
        v_holder = np.zeros(pos_v.shape)
        a_holder = np.zeros(pos_v.shape)
        speed_holder = np.zeros(pos_v.shape[0])
        pos_v = pos_v[0:len(self.time_vector)]  # Make sure we don't have a leading number ie extra "time" frame
        for ddx in range(0, pos_v.shape[0]):
            v_holder[ddx, :] = np.copy((pos_v[ddx, :] - pos_v[ddx - 1, :]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
            # v_holder[ddx, :] = scipy.ndimage.gaussian_filter1d(v_holder[ddx, :], 3)
        # Get cubic spline representation of speeds after smoothing
        if spline:
            v_holder = vu.cubic_spline_smoothing(v_holder, 0.1)
            # v_holder = scipy.signal.medfilt2d(v_holder, 1)
        # Calculate speed, acceleration from smoothed (no time-dependent jumps) velocities
        try:
            for ddx in range(0, pos_v.shape[0]):
                speed_holder[ddx] = np.sqrt(v_holder[ddx, 0] ** 2 + v_holder[ddx, 1] ** 2 + v_holder[ddx, 2] ** 2)
            for ddx in range(0, pos_v.shape[0]):
                a_holder[ddx, :] = np.copy(
                    (v_holder[ddx, :] - v_holder[ddx - 1, :]) / (self.time_vector[ddx] - self.time_vector[ddx - 1]))
        except:
            pass
        if spline:
            speed_holder = scipy.ndimage.gaussian_filter1d(speed_holder, sigma=1, mode='mirror')
        return np.asarray(v_holder), np.asarray(a_holder), np.asarray(speed_holder)

    def segment_reaches_with_speed_peaks(self, block=False):
        """ Function to segment out reaches using a positional and velocity threshold. """
        # find peaks of speed velocities
        # For each peak > 0.4 m/s, find the local minima (s < 0.05) for that time series
        # Take minima - 15 as tentative reach start time
        self.reach_start_time = []
        self.reach_peak_time = []
        self.left_start_times = []
        self.right_start_times = []
        self.left_peak_times = []
        self.right_peak_times = []
        self.right_reach_end_times = []
        self.left_reach_end_times = []
        self.reach_duration = []
        self.total_block_reaches = 0
        # Get some basic information about what's going on in the trial from micro-controller data
        hidx = np.where(self.handle_s > .1)
        hid = np.zeros(self.handle_s.shape[0])
        hid[hidx] = 1
        if np.nonzero(hid):
            self.handle_moved = True
        else:
            self.handle_moved = False
        if np.nonzero(self.lick):
            self.rewarded = True
        else:
            self.rewarded = False
        # Find peaks in left palm time-series
        pad_length = 5  # padding for ensuring we get the full reach
        lps = np.copy(self.left_palm_s[:])
        rps = np.copy(self.right_palm_s[:])
        lps[self.prob_filter_index] = 0
        rps[self.prob_filter_index] = 0
        # If palms are < 0.6 p-value, remove chance at "maxima"
        left_palm_prob = np.where(self.left_palm_p < 0.6)[0]
        right_palm_prob = np.where(self.right_palm_p < 0.6)[0]
        # If palms are > 0.23m in the x-direction towards the handle 0 position.
        left_palm_pos_f = np.where(self.left_palm[:, 0] < 0.15)[0]
        right_palm_pos_f = np.where(self.right_palm[:, 0] < 0.15)[0]
        lps[left_palm_prob] = 0
        rps[right_palm_prob] = 0
        rps[right_palm_pos_f] = 0
        lps[left_palm_pos_f] = 0
        lps[hidx] = 0
        rps[hidx] = 0
        lps[0:4] = 0  # remove any possible edge effect
        rps[0:4] = 0  # remove any possible edge effect
        self.left_palm_maxima = find_peaks(lps, height=0.3, distance=8)[0]
        if self.left_palm_maxima.any():
            print('Left Palm Reach')
            for ir in range(0, self.left_palm_maxima.shape[0]):
                if self.left_palm_maxima[ir] > 20:
                    left_palm_below_thresh = np.argmin(
                        self.left_palm_s[self.left_palm_maxima[ir] - 20: self.left_palm_maxima[ir]])
                else:
                    left_palm_below_thresh = np.argmin(self.left_palm_s[0:self.left_palm_maxima[ir]])
                left_palm_below_thresh_after = self.left_palm_maxima[ir] + \
                                               np.argmin(self.left_palm_s[
                                                         self.left_palm_maxima[ir]: self.left_palm_maxima[ir] + 15])
                start_time_l = self.left_palm_maxima[ir] - left_palm_below_thresh
                if start_time_l < 0:
                    start_time_l = 1
                self.left_start_times.append(start_time_l)
                self.left_peak_times.append(self.left_palm_maxima[ir])
                self.reach_peak_time.append(self.left_palm_maxima[ir])  # Record Peak
                self.left_reach_end_times.append(left_palm_below_thresh_after)  # Record putative end of motion
                self.reach_duration.append(
                    self.time_vector[left_palm_below_thresh_after] - self.time_vector[start_time_l])
        # Find peaks in right palm time-series
        self.right_palm_maxima = find_peaks(rps, height=0.3, distance=8)[0]
        if self.right_palm_maxima.any():
            print('Right Palm Reach')
            for ir in range(0, self.right_palm_maxima.shape[0]):
                if self.right_palm_maxima[ir] > 20:
                    right_palm_below_thresh = np.argmin(
                        self.right_palm_s[self.right_palm_maxima[ir] - 20:self.right_palm_maxima[ir]])
                else:
                    right_palm_below_thresh = np.argmin(self.right_palm_s[0:self.right_palm_maxima[ir]])
                right_palm_below_thresh_after = self.right_palm_maxima[ir] + np.argmin(
                    self.right_palm_s[self.right_palm_maxima[ir]:self.right_palm_maxima[ir] + 15])
                start_time_r = self.right_palm_maxima[ir] - right_palm_below_thresh
                if start_time_r < 0:
                    start_time_r = 1
                self.right_start_times.append(start_time_r)
                self.right_peak_times.append(self.right_palm_maxima[ir])
                self.reach_peak_time.append(self.right_palm_maxima[ir])
                self.right_reach_end_times.append(right_palm_below_thresh_after)
                self.reach_duration.append(
                    self.time_vector[right_palm_below_thresh_after] - self.time_vector[start_time_r])
        if block:
            self.total_block_reaches = 0
            pdb.set_trace()
        # Check for unrealistic values (late in trial)
        # Take min of right and left start times as "reach times" for start of classification extraction
        if self.right_start_times and self.left_start_times:
            self.reach_start_time = min(list(self.right_start_times) + list(self.left_start_times)) + 1
            self.reach_end_time = max(list(self.right_reach_end_times) + list(self.left_reach_end_times)) + 1
            print('LR')
        elif self.right_start_times:
            self.reach_start_time = min(list(self.right_start_times)) + 1
            self.reach_end_time = max(list(self.right_reach_end_times)) + 1
            print('R')
        elif self.left_start_times:
            self.reach_start_time = min(list(self.left_start_times)) + 1
            self.reach_end_time = max(list(self.left_reach_end_times)) + 1
            print('L')
        else:
            self.reach_start_time = 0
            self.reach_end_time = 100
            print('No LR')
        self.handle_moved = 0
        self.trial_cut_vector = []
        return

    def plot_velocities_against_probabilities(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        bins_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
        plt.hist(np.asarray(self.raw_speeds).flatten(), bins=bins_list, alpha=0.7, color='r', log=True,
                 label='Raw Speeds')
        plt.hist(np.asarray(self.speeds).flatten(), bins=bins_list, alpha=0.9, color='b', log=True,
                 label='Cubic Filtered Speeds')
        plt.title('Speeds Vs Probabilities')
        ax.set_xlabel('Speed Values')
        ax.set_ylabel('Log Counts')
        plt.legend()
        plt.savefig(self.sstr + '/timeseries/p_v_hist.png', dpi=1200)
        plt.close()
        return

    def segment_data_into_reach_dict(self, trial_num):
        """ Function that iterates over reaching indices,
            saves the segmented data and corresponding class labels into a dataframe.
        """
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
                          #            # Outliers
                          'nose_o': self.nose_o, 'handle_o': self.handle_o, 'left_shoulder_o': self.left_shoulder_o,
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
                          'right_end_base_o': self.right_end_base_o, 'right_end_tip_o': self.right_end_tip_o,
                          # Kinematic Features
                          'reach_hand': self.arm_id_list, 'right_start_time': self.right_start_times,
                          'left_start_time': self.left_start_times, 'reach_time': self.reach_start_time,
                          'reach_duration': self.reach_end_time,
                          'left_end_time': self.left_reach_end_times, 'right_end_time': self.right_reach_end_times,
                          'left_reach_peak': self.left_peak_times, 'right_reach_peak': self.right_peak_times,
                          'endpoint_error': self.endpoint_error,
                          # Sensor Data
                          'handle_moving_sensor': self.h_moving_sensor, 'lick_beam': self.lick_vector,
                          'reward_zone': self.reward_zone_sensor, 'time_vector': self.time_vector,
                          'lick_vector': self.lick_vector,
                          'response_sensor': self.exp_response_sensor, 'x_rob': self.x_robot, 'y_rob': self.y_robot,
                          'z_rob': self.z_robot,
                          # Principle components
                          'PC1': self.pos_v_a_pc[:, 0], 'PC2': self.pos_v_a_pc[:, 1], 'PC3': self.pos_v_a_pc[:, 2],
                          'PCvar': self.pos_v_a_pc_variance,
                          'Left_PC1': self.left_arm_pc_pos_v_a[:, 0], 'Left_PC2': self.left_arm_pc_pos_v_a[:, 0],
                          'Left_PC3': self.left_arm_pc_pos_v_a[:, 0], 'LeftPCVar': self.left_arm_pos_v_a_pc_variance,
                          'Right_PC1': self.right_arm_pc_pos_v_a[:, 0], 'Right_PC2': self.right_arm_pc_pos_v_a[:, 1],
                          'Right_PC3': self.right_arm_pc_pos_v_a[:, 2], 'RightPCVar': self.right_arm_pos_v_a_pc_variance
                          }
        # Create dataframe object from df, containing
        df = pd.DataFrame({key: pd.Series(np.asarray(value)) for key, value in self.save_dict.items()})
        df['Trial'] = trial_num
        df.set_index('Trial', append=True, inplace=True)
        df['Date'] = self.date
        df.set_index('Date', append=True, inplace=True)
        df['Session'] = self.session
        df.set_index('Session', append=True, inplace=True)
        df['Rat'] = self.rat
        df.set_index('Rat', append=True, inplace=True)
        return df

    def get_reach_dataframe_from_block(self, outlier_data=False):
        """ Function that obtains a trialized (based on reaching start times)
            pandas dataframe for a provided experimental session. """
        self.total_speeds = []
        self.left_hand_speeds = []
        self.right_hand_speeds = []
        self.left_hand_raw_speeds = []
        self.right_hand_raw_speeds = []
        self.total_outliers = []
        # Obtain workspace coordinate alignment
        self.align_workspace_coordinates_session()
        # Code here to count # of "peaks" in total data
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
            # Obtain values from experimental data
            self.segment_and_filter_kinematic_block(sts, stp)
            self.extract_sensor_data(sts, stp)
            # Collect data about the trial
            self.left_hand_raw_speeds.append(np.asarray(self.raw_speeds[2:14]))
            self.right_hand_raw_speeds.append(np.asarray(self.raw_speeds[2:14]))
            # Find and obtain basic classifications for each reach in the data
            self.segment_reaches_with_speed_peaks()
            # Collect data about outliers, robustness of kinematics
            if outlier_data:
                if ix == 0:
                    self.total_raw_speeds = np.asarray(self.raw_speeds).flatten()
                    self.total_preprocessed_speeds = np.asarray(self.speeds).flatten()
                    self.total_outliers = np.asarray(self.outlier_list).flatten()
                else:
                    self.total_raw_speeds = np.hstack((self.total_raw_speeds, np.asarray(self.raw_speeds).flatten()))
                    self.total_preprocessed_speeds = np.hstack((self.total_preprocessed_speeds,
                                                                np.asarray(self.speeds).flatten()))
                    self.total_outliers = np.hstack((np.asarray(self.outlier_list).flatten(), self.total_outliers))
            win_length = 15
            # Segment reach block
            if self.reach_start_time:  # If reach detected
                # tt = self.reach_end_time - self.reach_start_time
                end_pad = 1  # length of behavior
                reach_end_time = self.reach_end_time + end_pad  # For equally-spaced arrays
                print(self.reach_start_time, self.reach_end_time)
                if self.lick_vector.any() == 1:  # If trial is rewarded
                    self.first_lick_signal = np.where(self.lick_vector == 1)[0][0]
                    if 5 < self.first_lick_signal < 20:  # If reward is delivered after initial time-out
                        self.segment_and_filter_kinematic_block(sts, sts + 100)
                        self.extract_sensor_data(sts, sts + 100)
                        print('Possible Tug of War Behavior!')
                    else:  # If a reach is detected (successful reach)
                        self.segment_and_filter_kinematic_block(
                            sts + self.reach_start_time - win_length,
                            sts + reach_end_time + win_length)
                        self.extract_sensor_data(sts + self.reach_start_time - win_length,
                                                 sts + reach_end_time + win_length)
                        print('Successful Reach Detected')
                else:  # unrewarded reach
                    self.segment_and_filter_kinematic_block(sts + self.reach_start_time - win_length,
                                                            sts + reach_end_time + win_length)
                    self.extract_sensor_data(sts + self.reach_start_time - win_length, sts +
                                             reach_end_time + win_length)
                    print('Un-Rewarded Reach Detected')
            else:  # If reach not detected in trial
                self.segment_and_filter_kinematic_block(sts - win_length, sts + 100)
                self.extract_sensor_data(sts - win_length, sts + 100)
                print('No Reaching Found in Trial Block' + str(ix))
            df = self.segment_data_into_reach_dict(ix)
            self.seg_num = 0
            self.start_trial_indice = []
            self.images = []
            if ix == 0:
                self.reaching_dataframe = df
            else:
                self.reaching_dataframe = pd.concat([df, self.reaching_dataframe])
        # savefile = self.sstr + str(self.rat) + str(self.date) + str(self.session) + 'final_save_data.csv'
        # self.save_reaching_dataframe(savefile)
        # self.plot_verification_variables()
        return self.reaching_dataframe

    def save_reaching_dataframe(self, filename):
        self.reaching_dataframe.to_csv(filename)
        return

    def plot_verification_variables(self):
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
        plt.hist(np.asarray(self.total_raw_speeds).flatten(), bins=bins, color='r', log=True, label='Raw')
        plt.hist(np.asarray(self.speed_holder).flatten(), bins=bins, color='b', log=True, label='Final')
        plt.xlabel('Speeds')
        plt.ylabel('Log Counts')
        plt.title('Speeds Over Block')
        plt.legend()
        plt.savefig(self.sstr + 'total_speed_comparison.png', dpi=1200)
        plt.close()
        fig, [ax, ax1] = plt.subplots(nrows=1, ncols=2)
        ax.hist(np.asarray(self.total_raw_speeds[14:-1]).flatten(), bins=bins, color='r', log=True, label='Raw')
        ax.hist(np.asarray(self.speed_holder[14:-1]).flatten(), bins=bins, color='b', log=True, label='Final')
        ax1.hist(np.asarray(self.total_raw_speeds[2:14]).flatten(), bins=bins, color='g', log=True, label='Raw')
        ax1.hist(np.asarray(self.speed_holder[2:14]).flatten(), bins=bins, color='b', log=True, label='Raw')
        ax1.set_xlabel('Speeds')
        ax1.set_ylabel('Log Counts')
        plt.title('Speeds Over Block: Left and Right Hands')
        plt.legend()
        plt.savefig(self.sstr + 'arm_speed_boxplot.png', dpi=1200)
        plt.close()
        return

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
        ax2.set_ylim(0, 0.8)
        ax1.set_ylim(0, 0.8)
        ax3.set_ylim(0, 1)
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
        # Plot reachsplitter variables
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
        # Book-keeping for visuals (metrics, etc)
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
        plt.close()
        return

    def vid_splitter_and_grapher(self, trial_num=0, plot=True, timeseries_plot=True, plot_reach=True, save_data=False):
        """ Function to split and visualize reaching behavior from a given experimental session. """
        tm = self.align_workspace_coordinates_session()
        for ix, sts in enumerate(self.trial_start_vectors):
            self.trial_index = sts
            if ix > trial_num:
                self.fps = 30
                self.trial_num = int(ix)
                bi = self.block_video_path.rsplit('.')[0]
                self.sstr = bi + '/trial' + str(ix)
                self.make_paths()
                self.clip_path = self.sstr + '/videos/trial_video.mp4'
                stp = self.trial_stop_vectors[ix]
                self.split_trial_video(sts, sts+300)
                print('Split Trial' + str(ix) + ' Video')
                self.segment_and_filter_kinematic_block(sts, sts+300)
                self.plot_verification_variables()
                self.extract_sensor_data(sts, sts+300)
                self.segment_reaches_with_speed_peaks()
                self.plot_velocities_against_probabilities()
                print('Finished Plotting!   ' + str(ix))
                if timeseries_plot:
                    self.plot_interpolation_variables_palm('total')
                if plot:
                    self.plot_predictions_videos(plot_digits=False)
                    self.make_gif_reaching()
                    self.images = []
                    self.make_combined_gif()
                    print('GIF MADE for  ' + str(ix))
                win_length = 200
                # Check if we can cut based on lick
                if self.reach_start_time:
                    if self.first_lick_signal:
                        if self.reach_start_time - sts - 30 < 0:  # are we close to the start of the trial?
                            self.reach_start_time += 15  # spacer to keep array values
                        self.split_trial_video(self.reach_start_time + self.trial_index - 30, self.trial_index +
                                               self.first_lick_signal + 20, segment=True, num_reach=0)
                        self.segment_and_filter_kinematic_block(
                            self.reach_start_time + self.trial_index - 30,
                            self.trial_index + self.first_lick_signal + 20)
                        self.extract_sensor_data(self.trial_index + self.reach_start_time - 30, self.trial_index +
                                                 self.first_lick_signal + 20)
                    else:
                        if self.reach_start_time - sts - 30 < 0:  # are we close to the start of the trial?
                            self.reach_start_time += 15  # spacer to keep array values
                        self.split_trial_video(self.reach_start_time + self.trial_index - 30,
                                               self.trial_index + win_length,
                                               segment=True, num_reach=0)
                        self.segment_and_filter_kinematic_block(
                            self.reach_start_time + self.trial_index - 30,
                            self.trial_index + win_length)
                        self.extract_sensor_data(self.reach_start_time + self.trial_index - 30,
                                                 self.trial_index + win_length)
                else:
                    if self.first_lick_signal:
                        self.split_trial_video(self.trial_index, self.trial_index + self.first_lick_signal + 10,
                                               segment=True, num_reach=0)
                        self.segment_and_filter_kinematic_block(self.trial_index,
                                                                self.first_lick_signal + self.trial_index + 10)
                        self.extract_sensor_data(self.trial_index, self.trial_index + self.first_lick_signal + 10)
                    else:
                        print('No Reach Detected.')
                        self.split_trial_video(self.trial_index, self.trial_index + 200, segment=True, num_reach=0)
                        self.segment_and_filter_kinematic_block(self.trial_index, self.trial_index + 200)
                        self.extract_sensor_data(self.trial_index, self.trial_index + 200)
                if timeseries_plot:
                    self.plot_interpolation_variables_palm('reach')
                    self.plot_timeseries_features()
                if plot_reach:
                    self.plot_predictions_videos(segment=True, plot_digits=True)
                    self.make_gif_reaching(segment=True, nr=0)
                    print('Reaching GIF made for reach ' + str(0) + 'in trial ' + str(ix))
                    self.images = []
                    self.make_combined_gif(segment=True, nr=0)
                self.seg_num = 0
                self.start_trial_indice = []
                self.images = []
        if save_data:
            reaching_trial_indices_df = pd.DataFrame(self.block_cut_vector)
            reaching_trial_indices_df.to_csv(
                'reaching_vector_' + str(self.rat) + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                index=False, header=False)
            print('Saved reaching indices and bouts for block..' + str(self.date) + str(self.session))
        return self.block_cut_vector

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

    def plot_timeseries_features(self):
        """ Function to plot diagnostic time-series plots from single-trial data. """
        filename_pc = self.sstr + '/timeseries/' + 'pc_timeseries.png'
        filename_pos = self.sstr + '/timeseries/' + 'full_pos_timeseries.png'
        filename_vel = self.sstr + '/timeseries/' + 'full_vel_timeseries.png'
        filename_speed = self.sstr + '/timeseries/' + 'full_speed_timeseries.png'
        xyzpalms = self.sstr + '/timeseries/' + 'xyzpalms_timeseries.png'
        rightvsleftxy = self.sstr + '/timeseries/' + 'rlxypalms_timeseries.png'
        axel0 = plt.figure(figsize=(8, 4))
        frames_n = np.around(self.time_vector, 2)
        frames = frames_n - frames_n[0]  # normalize frame values by first entry.
        plt.title('Principal Components of Position, P+V, P+V+A')
        # plt.plot(self.pos_v_a_pc_variance[:, 0], self.pos_v_a_pc_variance[:, 1], c='r', label='PCs for total kinematics')
        # plt.plot(self.right_arm_pc_pos[:, 0], self.right_arm_pc_pos[:, 1], c='b', label='Right Arm PC')
        axel0.tight_layout(pad=0.005)
        axel0.legend()
        axel0.savefig(filename_pc, bbox_inches='tight')
        plt.close()
        axel = plt.figure(figsize=(10, 4))
        try:
            plt.axvline(frames[self.reach_start_time], color='black', label='Reach Start ')
        except:
            pass
        plt.title('Palm Positions During Trial')
        plt.plot(frames, self.right_palm[:, 0], c='g', label='X Right Palm')
        plt.plot(frames, self.right_palm[:, 1], c='g', linestyle='dashed', label='Y Right Palm')
        plt.plot(frames, self.right_palm[:, 2], c='g', linestyle='dotted', label='Z Right Palm')
        plt.plot(frames, self.left_palm[:, 0], c='r', label='X Left Palm')
        plt.plot(frames, self.left_palm[:, 1], c='r', linestyle='dashed', label='Y Left Palm')
        plt.plot(frames, self.left_palm[:, 2], c='r', linestyle='dotted', label='Z Left Palm')
        plt.plot(frames, self.lick_vector / 10, color='y', label='Licks Occurring')
        try:
            for tsi, segment_trials in enumerate(self.start_trial_indice):
                plt.axvline(frames[segment_trials], color='black', label='Trial ' + str(tsi))
        except:
            pass
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        axel.tight_layout(pad=0.005)
        axel.legend()
        axel.savefig(filename_pos, bbox_inches='tight')
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
        plt.xlabel('Time (s)')
        plt.ylabel(' Palm Velocity (m/s)')
        axel1.legend()
        axel1.tight_layout(pad=0.005)
        # pdb.set_trace()
        axel1.savefig(filename_vel, bbox_inches='tight')
        plt.close()
        axel2 = plt.figure(figsize=(10, 4))
        plt.title('Speeds (Arm, Palm) During Trial')
        plt.plot(frames, self.right_palm_s, c='b', label='Right palm speed')
        plt.plot(frames, self.left_palm_s, c='r', label='Left palm speed')
        plt.plot(frames, self.right_forearm_s, c='c', linestyle='dashed', label='Right arm speed')
        plt.plot(frames, self.left_forearm_s, c='pink', linestyle='dashed', label='Left arm speed')
        plt.ylim(0, 1.6)
        plt.xlabel('Time (s) ')
        plt.ylabel('Speed (m/s)')
        axel2.legend()
        axel2.tight_layout(pad=0.0005)
        axel2.savefig(filename_speed, bbox_inches='tight')
        plt.close()
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
        axel3.tight_layout(pad=0.0005)
        axel3.savefig(xyzpalms, bbox_inches='tight')
        plt.close()
        axel4 = plt.figure(figsize=(8, 8))
        plt.title(' R vs L X Y Z Plots')
        plt.plot(self.right_palm[:, 0], self.right_palm[:, 1], c='r', label='Right Palm')
        plt.plot(self.left_palm[:, 0], self.left_palm[:, 1], c='g', label='Left Palm')
        plt.xlabel('M')
        plt.ylabel('M')
        axel4.legend()
        axel4.tight_layout(pad=0.0005)
        axel4.savefig(rightvsleftxy, bbox_inches='tight')
        plt.close()
        return

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

    def plot_verification_histograms(self):

        return

    def plot_predictions_videos(self, segment=False, multi_segment_plot=True, plot_digits=False, draw_skeleton=True,
                                clean=False, legend_on=False):
        """ Function to plot, frame by frame, the full 3-D reaching behavior over time. """
        tldr = self.left_palm.shape[0]  # set range in single trial
        tlde = 0
        frames_n = np.around(self.time_vector, 2)
        frames = frames_n - frames_n[0]  # normalize frame values by first entry.
        # Cut off long values for smaller reaching times
        for isx in range(tlde, tldr):
            self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            self.set_lag_plot_term(isx)
            fixit1 = plt.figure(figsize=(44 / 4, 52 / 4))
            axel = plt.subplot2grid((56, 44), (0, 0), colspan=44, rowspan=44, projection='3d')
            axel2 = plt.subplot2grid((56, 44), (46, 8), colspan=30, rowspan=11)
            plt.subplots_adjust(top=0.95)
            axel.set_xlim(0, 0.4)
            axel.set_ylim(0, 0.5)
            axel.set_zlim(0.4, 0.60)
            axel.plot([0.205, 0.205], [0, 0.5], [0.5, 0.5], c='g', linestyle='dashed', linewidth=5,
                      markersize=5, label='Trial Reward Zone')  # make a 3-D line for our "reward zone"
            X, Y = np.meshgrid(np.linspace(0.2, 0, 10), np.linspace(0, 0.5, 10))
            Z = np.reshape(np.linspace(0.4, 0.4, 100), X.shape)
            axel.plot_surface(X, Y, Z, alpha=0.3, zorder=0)
            axel.scatter(self.handle[isx - self.lag: isx, 0], self.handle[isx - self.lag, 1],
                         self.handle[isx - self.lag:isx, 2], marker='x',
                         s=400, c='gold', label='Handle')
            axel.scatter(self.nose[isx - self.lag: isx, 0], self.nose[isx - self.lag, 1],
                         self.nose[isx - self.lag:isx, 2], c='m',
                         s=50 + 100 * (self.prob_nose[isx]), alpha=(self.prob_nose[isx]),
                         label='Nose')
            axel.scatter(self.right_palm[isx - self.lag: isx, 0], self.right_palm[isx - self.lag: isx, 1],
                         self.right_palm[isx - self.lag: isx, 2], marker='.',
                         s=150 + 300 * (self.right_palm_p[isx]), c='skyblue',
                         alpha=(self.right_palm_p[isx]), label='Right Palm')
            axel.scatter(self.left_palm[isx - self.lag: isx, 0], self.left_palm[isx - self.lag: isx, 1],
                         self.left_palm[isx - self.lag: isx, 2], marker='.',
                         s=150 + 300 * (self.left_palm_p[isx]), c='salmon',
                         alpha=(self.left_palm_p[isx]), label='Left Palm')
            if multi_segment_plot:
                axel.scatter(self.right_forearm[isx - self.lag: isx, 0], self.right_forearm[isx - self.lag: isx, 1],
                             self.right_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.right_forearm_p[isx]), c='royalblue',
                             alpha=(self.right_forearm_p[isx]), label='Right Forearm')
                axel.scatter(self.right_wrist[isx - self.lag: isx, 0], self.right_wrist[isx - self.lag: isx, 1],
                             self.right_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.right_wrist_p[isx]), c='b',
                             alpha=(self.right_wrist_p[isx]), label='Right Wrist')
                axel.scatter(self.left_wrist[isx - self.lag: isx, 0], self.left_wrist[isx - self.lag: isx, 1],
                             self.left_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.left_wrist_p[isx]), c='salmon',
                             alpha=(self.left_wrist_p[isx]), label='Left Wrist ')
                axel.scatter(self.left_forearm[isx - self.lag: isx, 0], self.left_forearm[isx - self.lag: isx, 1],
                             self.left_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.left_forearm_p[isx]), c='r',
                             alpha=(self.left_forearm_p[isx]), label='Left Forearm')
                axel.scatter(self.left_shoulder[isx - self.lag: isx, 0], self.left_shoulder[isx - self.lag: isx, 1],
                             self.left_shoulder[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.left_shoulder_p[isx]), c='darkred',
                             alpha=(self.left_shoulder_p[isx]), label='Left Shoulder ')
                axel.scatter(self.right_shoulder[isx - self.lag: isx, 0], self.right_shoulder[isx - self.lag: isx, 1],
                             self.right_shoulder[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.right_shoulder_p[isx]), c='navy',
                             alpha=(self.right_shoulder_p[isx]), label='Right Shoulder')

            if draw_skeleton:
                axel.plot([self.right_wrist[isx, 0], self.right_forearm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_forearm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_forearm[isx, 2]],
                          alpha=(self.right_wrist_p[isx]),
                          markersize=55 + 50 * np.mean(self.right_wrist_p[isx]),
                          c='r', linestyle='dashed')
                axel.plot([self.right_forearm[isx, 0], self.right_shoulder[isx, 0]],
                          [self.right_forearm[isx, 1], self.right_shoulder[isx, 1]],
                          [self.right_forearm[isx, 2], self.right_shoulder[isx, 2]],
                          alpha=(self.right_shoulder_p[isx]),
                          markersize=55 + 50 * np.mean(self.right_shoulder_p[isx]),
                          c='b', linestyle='dashed')
                axel.plot([self.left_forearm[isx, 0], self.left_shoulder[isx, 0]],
                          [self.left_forearm[isx, 1], self.left_shoulder[isx, 1]],
                          [self.left_forearm[isx, 2], self.left_shoulder[isx, 2]],
                          alpha=(self.left_forearm_p[isx]),
                          markersize=55 + 50 * np.mean(self.left_forearm_p[isx]),
                          c='r', linestyle='dashed')
                axel.plot([self.right_wrist[isx, 0], self.right_palm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_palm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_palm[isx, 2]],
                          alpha=(self.right_wrist_p[isx]),
                          markersize=55 + 35 * np.mean(self.right_wrist_p[isx]),
                          c='r', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_forearm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_forearm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_forearm[isx, 2]],
                          alpha=(self.left_forearm_p[isx]),
                          markersize=55 + 50 * np.mean(self.left_forearm_p[isx]),
                          c='c', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_palm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_palm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_palm[isx, 2]],
                          alpha=(self.left_palm_p[isx]),
                          markersize=55 + 40 * np.mean(self.left_palm_p[isx]),
                          c='c', linestyle='dashed')
            if plot_digits:
                axel.scatter(self.left_index_base[isx - self.lag: isx, 0], self.left_index_base[isx - self.lag: isx, 1],
                             self.left_index_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.left_index_base_p[isx], c='dodgerblue',
                             alpha=self.left_index_base_p[isx], label='Left Index Base ')
                axel.scatter(self.right_index_base[isx - self.lag: isx, 0],
                             self.right_index_base[isx - self.lag: isx, 1],
                             self.right_index_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.right_index_base_p[isx], c='pink',
                             alpha=self.right_index_base_p[isx], label='Right Index Base ')
                axel.scatter(self.left_middle_base[isx - self.lag: isx, 0],
                             self.left_middle_base[isx - self.lag: isx, 1],
                             self.left_middle_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.left_middle_base_p[isx], c='dodgerblue',
                             alpha=self.left_middle_base_p[isx], label='Left Middle Base ')
                axel.scatter(self.right_middle_base[isx - self.lag: isx, 0],
                             self.right_middle_base[isx - self.lag: isx, 1],
                             self.right_middle_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.right_middle_base_p[isx], c='pink',
                             alpha=self.right_middle_base_p[isx], label='Right Middle Base ')
                axel.scatter(self.left_third_base[isx - self.lag: isx, 0], self.left_third_base[isx - self.lag: isx, 1],
                             self.left_third_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.left_third_base_p[isx], c='skyblue',
                             alpha=self.left_third_base_p[isx], label='Left Third Base ')
                axel.scatter(self.right_third_base[isx - self.lag: isx, 0],
                             self.right_third_base[isx - self.lag: isx, 1],
                             self.right_third_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.right_third_base_p[isx], c='salmon',
                             alpha=self.left_third_base_p[isx], label='Right Third Base ')
                axel.scatter(self.left_end_base[isx - self.lag: isx, 0], self.left_end_base[isx - self.lag: isx, 1],
                             self.left_end_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.left_end_base_p[isx], c='azure',
                             alpha=self.left_end_base_p[isx], label='Left End Base ')
                axel.scatter(self.right_end_base[isx - self.lag: isx, 0],
                             self.right_end_base[isx - self.lag: isx, 1],
                             self.right_end_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 30 * self.right_end_base_p[isx], c='mistyrose',
                             alpha=self.right_end_base_p[isx], label='Right End Base ')
                axel.scatter(self.left_index_tip[isx - self.lag: isx, 0], self.left_index_tip[isx - self.lag: isx, 1],
                             self.left_index_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.left_index_tip_p[isx], c='azure',
                             alpha=self.left_index_tip_p[isx], label='Left Index Tip ')
                axel.scatter(self.right_index_tip[isx - self.lag: isx, 0],
                             self.right_index_tip[isx - self.lag: isx, 1],
                             self.right_index_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.right_index_tip_p[isx], c='mistyrose',
                             alpha=self.right_index_tip_p[isx], label='Right Index Tip ')
                axel.scatter(self.left_middle_tip[isx - self.lag: isx, 0],
                             self.left_middle_tip[isx - self.lag: isx, 1],
                             self.left_middle_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.left_middle_tip_p[isx], c='azure',
                             alpha=self.left_middle_tip_p[isx], label='Left Middle Tip ')
                axel.scatter(self.right_middle_tip[isx - self.lag: isx, 0],
                             self.right_middle_tip[isx - self.lag: isx, 1],
                             self.right_middle_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.right_middle_tip_p[isx], c='mistyrose',
                             alpha=self.right_middle_tip_p[isx], label='Right Middle Tip ')
                axel.scatter(self.left_third_tip[isx - self.lag: isx, 0], self.left_third_tip[isx - self.lag: isx, 1],
                             self.left_third_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.left_third_tip_p[isx], c='azure',
                             alpha=self.left_third_tip_p[isx], label='Left Third Tip ')
                axel.scatter(self.right_third_tip[isx - self.lag: isx, 0],
                             self.right_third_tip[isx - self.lag: isx, 1],
                             self.right_third_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.right_third_tip_p[isx], c='mistyrose',
                             alpha=self.right_third_tip_p[isx], label='Right Third Tip ')
                axel.scatter(self.left_end_tip[isx - self.lag: isx, 0], self.left_end_tip[isx - self.lag: isx, 1],
                             self.left_end_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.left_end_tip_p[isx], c='dodgerblue',
                             alpha=self.left_end_tip_p[isx], label='Left End Tip ')
                axel.scatter(self.right_end_tip[isx - self.lag: isx, 0],
                             self.right_end_tip[isx - self.lag: isx, 1],
                             self.right_end_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 30 * self.right_end_tip_p[isx], c='pink',
                             alpha=self.right_end_tip_p[isx], label='Right End Tip ')
            axel.set_xlabel('M (X)')
            axel.set_ylabel('M(Y)')
            axel.set_zlabel('M (Z)')
            axel.view_init(10, -60)
            axel2.plot(frames[0:isx], self.left_palm_s[0:isx], marker='.', c='red', label='Left Palm Speed')
            axel2.plot(frames[0:isx], self.right_palm_s[0:isx], marker='.', c='skyblue', label='Right Palm Speed')
            if self.start_trial_indice:
                for ere, tre in enumerate(self.start_trial_indice):
                    if isx - 10 > tre > isx + 10:
                        axel2.axvline(tre, 0, 1, c='black', markersize=30, linestyle='dotted',
                                      label='REACH ' + str(ere))
                        print('Plotted a reach..')
            axel2.set_xlabel('Time from trial onset (s)')
            axel2.set_ylabel('m/ s')
            axel2.set_ylim(0, 2)
            axel2.set_xlim(0, frames[-1])
            fixit1.tight_layout(pad=0.005)
            plt.margins(0.0005)
            if legend_on:
                axel2.legend(loc="upper right", fontsize='x-small')
                axel.legend(fontsize='large')
            if segment:
                self.filename = self.sstr + '/plots/reaches/' + str(isx) + 'palm_prob_timeseries.png'
            else:
                self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            plt.savefig(self.filename, bbox_inches='tight', pad_inches=0.00001)
            plt.close()
            self.images.append(imageio.imread(self.filename))
            if clean:
                os.remove(self.filename)
        return

    def make_gif_reaching(self, segment=False, nr=0):
        """ Function to make a per-trial GIF from the compilation of diagnostic plots made using ReachLoader. """
        if segment:
            imageio.mimsave(self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4', self.images,
                            fps=self.fps)
            self.gif_save_path = self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4'
        else:
            imageio.mimsave(self.sstr + '/videos/total_3d_movie.mp4', self.images, fps=self.fps)
            self.gif_save_path = self.sstr + '/videos/total_3d_movie.mp4'
        return

    def make_combined_gif(self, segment=False, nr=0):
        """ Function to create a combined video and diagnostic graph GIF per-trial. """
        vid_gif = skvideo.io.vread(self.clip_path)
        plot_gif = skvideo.io.vread(self.gif_save_path)
        number_of_frames = int(min(vid_gif.shape[0], plot_gif.shape[0]))
        if segment:
            writer = imageio.get_writer(self.sstr + '/videos/reaches/reach_' + str(nr) + 'split_movie.mp4',
                                        fps=self.fps)
        else:
            writer = imageio.get_writer(self.sstr + '/videos/total_split_movie.mp4', fps=self.fps)
        for frame_number in range(number_of_frames):
            img1 = np.squeeze(vid_gif[frame_number, :, :, :])
            img2 = np.squeeze(plot_gif[frame_number, :, :, :])
            img1 = np.copy(img1[0:img2.shape[0], 0:img2.shape[1]])
            new_image = np.vstack((img1, img2))
            writer.append_data(new_image)
        writer.close()
        return

    def split_trial_video(self, start_frame, stop_frame, segment=False, num_reach=0):
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
        out = cv2.VideoWriter(self.clip_path, fourcc, self.fps, (w, h))
        c = 0
        while rval:
            rval, frame = vc.read()
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
