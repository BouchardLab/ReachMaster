
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import DataStream_Vis_Utils as utils
from moviepy.editor import *
import skvideo
import cv2
import imageio
import numpy as np
import viz_utils as vu
import scipy
# Import ffmpeg to write videos here..
ffm_path = 'C:/Users/bassp/OneDrive/Desktop/ffmpeg/bin/'
skvideo.setFFmpegPath(ffm_path)
import skvideo.io


class ReachViz:
    def __init__(self, date, session, data_path, block_vid_file, kin_path, rat):
        self.probabilities, self.bi_reach_vector, self.trial_index, self.first_lick_signal, self.outlier_indexes = [], [], [], [], []
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
        self.sstr, self.lick_index = 0, []
        self.rat_gaps, self.total_ints, self.interpolation_rmse, self.outlier_rmse, self.valid_rmse = [], [], [], [], []
        self.block_video_path = block_vid_file
        # Obtain Experimental Block of Data
        self.get_block_data()
        # Get Start/Stop of Trials
        self.get_starts_stops()
        # Initialize sensor variables
        self.exp_response_sensor, self.trial_sensors, self.h_moving_sensor, self.reward_zone_sensor, self.lick, self.trial_num = 0, 0, 0, 0, 0, 0
        self.time_vector, self.images, self.bout_vector = [], [], []
        self.trial_rewarded = False
        self.filename = None
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
        self.right_index_base, self.right_index_tip, self.right_middle_base, self.right_middle_tip, self.right_third_base, self.right_third_tip, self.right_end_base, self.right_end_tip = [], [], [], [], [], [], [], []
        self.left_index_base, self.left_index_tip, self.left_middle_base, self.left_middle_tip, self.left_third_base, self.left_third_tip, self.left_end_base, self.left_end_tip = [], [], [], [], [], [], [], []
        self.start_trial_indice, self.trial_cut_vector, self.block_cut_vector, self.handle_velocity, self.bout_reach = [], [], [], [], []
        self.handle_moved, self.gif_save_path, self.prob_right_index, self.prob_left_index, self.l_pos_index, self.r_pos_index = 0, 0, 0, 0, 0, 0
        self.x_robot, self.y_robot, self.z_robot, self.uninterpolated_right_palm, self.uninterpolated_left_palm = [], [], [], [], []
        self.prob_left_digit, self.prob_right_digit, self.left_digit_filter_index, self.right_digit_filter_index = [], [], [], []
        self.reaching_mask, self.right_arm_speed, self.left_arm_speed, self.reprojections, self.interpolation_gaps = [], [], [], [], []
        self.left_palm_speed, self.right_palm_speed, self.handle_speed, self.right_arm_velocity, self.left_arm_velocity = [], [], [], [], []
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
        self.prob_right_index, self.prob_left_index, self.bi_pos_index, self.r_reach_vector, self.l_reach_vector, self.left_prob_index, self.right_prob_index = [
            0, 0, 0, 0, 0, 0, 0]
        self.prob_list, self.pos_list, self.interpolation_rmse_list, self.valid_rmse_list, self.outlier_rmse_list = [], [], [], [], []
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
        vu.mkdir_p(self.sstr + '/timeseries/reaches')
        return

    def get_block_data(self):
        """ Function to fetch block positional and sensor data from rat database. """
        for kin_items in self.d:
            sess = kin_items.columns.levels[1]
            date = kin_items.columns.levels[2]
            self.dim = kin_items.columns.levels[3]
            if sess[0] in self.session:
                if date[0][-2:] in self.date:
                    print('Hooked block positions for date  ' + date[0] + '     and session  ' + sess[0])
                    self.kinematic_block = kin_items
        self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        return

    def threshold_data_with_probabilities(self, p_vector, p_thresh):
        """ Function to threshold input position vectors by the probability of this position being present. The mean
            over multiple cameras is used to better estimate error.
        """
        p_vector_hold = np.mean(p_vector, axis=1)
        low_p_idx = np.where(p_vector_hold < p_thresh)  # Filter positions by ind p values
        return np.asarray(low_p_idx)

    def get_starts_stops(self):
        """ Obtain the start and stop times of coarse behavior from the sensor block. """
        self.trial_start_vectors = self.block_exp_df['r_start'].values[0]
        self.trial_stop_vectors = self.block_exp_df['r_stop'].values[0]
        print('Number of Trials: ' + str(len(self.trial_start_vectors)))
        return

    def extract_sensor_data(self, idxstrt, idxstp, filter=True, check_lick=True):
        """ Function to extract probability thresholded sensor data from ReachMaster. Data is coarsely filtered.
        """
        self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        self.h_moving_sensor = np.asarray(np.copy(self.block_exp_df['moving'].values[0][idxstrt:idxstp]))
        self.hmove = np.any(self.h_moving_sensor)
        self.lick_index = np.asarray(np.copy(self.block_exp_df['lick'].values[0]))  # Lick DIO sensor
        self.reward_zone_sensor = np.asarray(np.copy(self.block_exp_df['RW'].values[0][idxstrt:idxstp]))
        self.time_vector = np.asarray(self.block_exp_df['time'].values[0][
                                      idxstrt:idxstp])  # extract trial timestamps from SpikeGadgets
        self.exp_response_sensor = np.asarray(self.block_exp_df['exp_response'].values[0][idxstrt:idxstp])
        r, theta, phi, self.x_robot, self.y_robot, self.z_robot = utils.forward_xform_coords(
            self.block_exp_df['x_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['y_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['z_pot'].values[0][idxstrt:idxstp])
        if check_lick:
            self.check_licking_times_and_make_vector()
        if filter:
            self.h_moving_sensor[self.prob_filter_index] = 0
            self.reward_zone_sensor[self.prob_filter_index] = 0
            self.lick[self.prob_filter_index] = 0
            self.exp_response_sensor[self.prob_filter_index] = 0
            self.x_robot[self.prob_filter_index] = 0
            self.y_robot[self.prob_filter_index] = 0
            self.z_robot[self.prob_filter_index] = 0
        self.sensor_data_list = [self.h_moving_sensor, self.lick_vector, self.reward_zone_sensor,
                                 self.exp_response_sensor,
                                 self.x_robot, self.y_robot, self.z_robot]
        return

    def segment_and_filter_kinematic_block_single_trial(self, cl1, cl2, p_thresh=0.4, coarse_threshold=0.4,
                                                        preprocess=True, get_ints=True):
        """ Function to segment, filter, and interpolate positional data across all bodyparts, using start and stop indices across
            whole-trial kinematics. """
        # Probabilities, mean value across all 3 cameras, used for graphing
        self.prob_right_arm = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[cl1:cl2, :], axis=1))
        self.prob_left_arm = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values[cl1:cl2, :], axis=1))
        self.prob_right_digit = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[21 + 81:24 + 81]].values[cl1:cl2, :], axis=1))
        self.prob_left_digit = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[cl1:cl2, :], axis=1))
        self.prob_nose = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values[cl1:cl2, :], axis=1))
        self.prob_right_shoulder = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[45 + 81:48 + 81]].values[cl1:cl2,
                    :], axis=1))
        self.prob_left_shoulder = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[9 + 81:12 + 81]].values[cl1:cl2, :],
                    axis=1))

        self.prob_filter_index = np.where(self.prob_nose < coarse_threshold)[0]
        self.left_arm_filter_index = np.where(self.prob_left_arm < p_thresh)[0]  # Index of left removal values
        self.right_arm_filter_index = np.where(self.prob_right_arm < p_thresh)[0]  # Index of right removal values
        # Body parts, XYZ, used in ReachViz
        self.nose = vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[6:9]].values[cl1:cl2, :])
        self.handle = np.mean(
            [vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[0:3]].values[cl1:cl2, :]),
             vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[3:6]].values[cl1:cl2, :])], axis=0)
        self.left_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[9:12]].values[cl1:cl2, :])  # 21 end
        self.right_shoulder = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])  # 57 end
        self.left_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        self.right_forearm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])  # 57 end
        self.left_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[15:18]].values[cl1:cl2, :])  # 21 end
        self.right_wrist = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])  # 57 end
        self.left_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:21]].values[cl1:cl2, :])
        self.right_palm = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        # Digits, optional for now
        self.right_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[27:30]].values[cl1:cl2, :])
        self.right_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[30:33]].values[cl1:cl2, :])
        self.right_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[36:39]].values[cl1:cl2, :])
        self.right_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[39:42]].values[cl1:cl2, :])
        self.right_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[42:45]].values[cl1:cl2, :])
        self.right_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])
        self.right_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])
        self.right_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])
        self.left_index_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        self.left_index_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[57:60]].values[cl1:cl2, :])
        self.left_middle_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[60:63]].values[cl1:cl2, :])
        self.left_middle_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[66:69]].values[cl1:cl2, :])
        self.left_third_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[69:72]].values[cl1:cl2, :])
        self.left_third_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[72:75]].values[cl1:cl2, :])
        self.left_end_base = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[75:78]].values[cl1:cl2, :])
        self.left_end_tip = vu.norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[78:81]].values[cl1:cl2, :])
        w = 81
        # Probabilities
        self.nose_p = self.kinematic_block[self.kinematic_block.columns[6 + w:9 + w]].values[cl1:cl2, :]
        self.handle_p = self.kinematic_block[self.kinematic_block.columns[3 + w:6 + w]].values[cl1:cl2, :]
        self.left_shoulder_p = self.kinematic_block[self.kinematic_block.columns[9 + w:12 + w]].values[cl1:cl2,
                               :]  # 21 end
        self.right_shoulder_p = self.kinematic_block[self.kinematic_block.columns[45 + w:48 + w]].values[cl1:cl2,
                                :]  # 57 end
        self.left_forearm_p = self.kinematic_block[self.kinematic_block.columns[12 + w:15 + w]].values[cl1:cl2,
                              :]  # 21 end
        self.right_forearm_p = self.kinematic_block[self.kinematic_block.columns[48 + w:51 + w]].values[cl1:cl2,
                               :]  # 57 end
        self.left_wrist_p = self.kinematic_block[self.kinematic_block.columns[15 + w:18 + w]].values[cl1:cl2,
                            :]  # 21 end
        self.right_wrist_p = self.kinematic_block[self.kinematic_block.columns[51 + w:54 + w]].values[cl1:cl2,
                             :]  # 57 end
        self.left_palm_p = self.kinematic_block[self.kinematic_block.columns[18 + w:21 + w]].values[cl1:cl2, :]
        self.right_palm_p = self.kinematic_block[self.kinematic_block.columns[54 + w:57 + w]].values[cl1:cl2, :]
        # Digits, optional for now
        self.right_index_base_p = self.kinematic_block[self.kinematic_block.columns[27 + w:30 + w]].values[cl1:cl2, :]
        self.right_index_tip_p = self.kinematic_block[self.kinematic_block.columns[30 + w:33 + w]].values[cl1:cl2, :]
        self.right_middle_base_p = self.kinematic_block[self.kinematic_block.columns[36 + w:39 + w]].values[cl1:cl2, :]
        self.right_middle_tip_p = self.kinematic_block[self.kinematic_block.columns[39 + w:42 + w]].values[cl1:cl2, :]
        self.right_third_base_p = self.kinematic_block[self.kinematic_block.columns[42 + w:45 + w]].values[cl1:cl2, :]
        self.right_third_tip_p = self.kinematic_block[self.kinematic_block.columns[45 + w:48 + w]].values[cl1:cl2, :]
        self.right_end_base_p = self.kinematic_block[self.kinematic_block.columns[48 + w:51 + w]].values[cl1:cl2, :]
        self.right_end_tip_p = self.kinematic_block[self.kinematic_block.columns[51 + w:54 + w]].values[cl1:cl2, :]
        self.left_index_base_p = self.kinematic_block[self.kinematic_block.columns[54 + w:57 + w]].values[cl1:cl2, :]
        self.left_index_tip_p = self.kinematic_block[self.kinematic_block.columns[57 + w:60 + w]].values[cl1:cl2, :]
        self.left_middle_base_p = self.kinematic_block[self.kinematic_block.columns[60 + w:63 + w]].values[cl1:cl2, :]
        self.left_middle_tip_p = self.kinematic_block[self.kinematic_block.columns[66 + w:69 + w]].values[cl1:cl2, :]
        self.left_third_base_p = self.kinematic_block[self.kinematic_block.columns[69 + w:72 + w]].values[cl1:cl2, :]
        self.left_third_tip_p = self.kinematic_block[self.kinematic_block.columns[72 + w:75 + w]].values[cl1:cl2, :]
        self.left_end_base_p = self.kinematic_block[self.kinematic_block.columns[75 + w:78 + w]].values[cl1:cl2, :]
        self.left_end_tip_p = self.kinematic_block[self.kinematic_block.columns[78 + w:81 + w]].values[cl1:cl2, :]
        self.extract_sensor_data(cl1, cl2, filter=False,
                                 check_lick=False)  # Get time vectors for calculating kinematics.
        self.positions = [self.nose, self.handle, self.left_shoulder, self.left_forearm, self.left_wrist,
                          self.left_palm, self.left_index_base,
                          self.left_index_tip, self.left_middle_base, self.left_middle_tip, self.left_third_base,
                          self.left_third_tip,
                          self.left_end_base, self.left_end_tip, self.right_shoulder, self.right_forearm,
                          self.right_wrist, self.right_palm,
                          self.right_index_base, self.right_index_tip, self.right_middle_base, self.right_middle_tip,
                          self.right_third_base, self.right_third_tip, self.right_end_base, self.right_end_tip]
        self.probabilities = [self.nose_p, self.handle_p, self.left_shoulder_p, self.left_forearm_p, self.left_wrist_p,
                              self.left_palm_p, self.left_index_base_p,
                              self.left_index_tip_p, self.left_middle_base_p, self.left_middle_tip_p,
                              self.left_third_base_p,
                              self.left_third_tip_p,
                              self.left_end_base_p, self.left_end_tip_p, self.right_shoulder_p, self.right_forearm_p,
                              self.right_wrist_p, self.right_palm_p,
                              self.right_index_base_p, self.right_index_tip_p, self.right_middle_base_p,
                              self.right_middle_tip_p,
                              self.right_third_base_p, self.right_third_tip_p, self.right_end_base_p,
                              self.right_end_tip_p]
        self.vel_list = [self.nose_v, self.handle_v, self.left_shoulder_v, self.left_forearm_v, self.left_wrist_v,
                         self.left_palm_v, self.left_index_base_v,
                         self.left_index_tip_v, self.left_middle_base_v, self.left_middle_tip_v, self.left_third_base_v,
                         self.left_third_tip_v, self.left_end_base_v,
                         self.left_end_tip_v, self.right_shoulder_v, self.right_forearm_v, self.right_wrist_v,
                         self.right_palm_v, self.right_index_base_v,
                         self.right_index_tip_v, self.right_middle_base_v, self.right_middle_tip_v,
                         self.right_third_base_v, self.right_third_tip_v,
                         self.right_end_base_v, self.right_end_tip_v]
        self.speed_list = [self.nose_s, self.handle_s, self.left_shoulder_s, self.left_forearm_s, self.left_wrist_s,
                           self.left_palm_s, self.left_index_base_s,
                           self.left_index_tip_s, self.left_middle_base_s, self.left_middle_tip_s,
                           self.left_third_base_s, self.left_third_tip_s, self.left_end_base_s,
                           self.left_end_tip_s, self.right_shoulder_s, self.right_forearm_s, self.right_wrist_s,
                           self.right_palm_s, self.right_index_base_s,
                           self.right_index_tip_s, self.right_middle_base_s, self.right_middle_tip_s,
                           self.right_third_base_s, self.right_third_tip_s,
                           self.right_end_base_s, self.right_end_tip_s]
        self.acc_list = [self.nose_a, self.handle_a, self.left_shoulder_a, self.left_forearm_a, self.left_wrist_a,
                         self.left_palm_a, self.left_index_base_a,
                         self.left_index_tip_a, self.left_middle_base_a, self.left_middle_tip_a, self.left_third_base_a,
                         self.left_third_tip_a, self.left_end_base_a,
                         self.left_end_tip_a, self.right_shoulder_a, self.right_forearm_a, self.right_wrist_a,
                         self.right_palm_a, self.right_index_base_a,
                         self.right_index_tip_a, self.right_middle_base_a, self.right_middle_tip_a,
                         self.right_third_base_a, self.right_third_tip_a,
                         self.right_end_base_a, self.right_end_tip_a]
        self.outlier_list = [self.nose_o, self.handle_o, self.left_shoulder_o, self.left_forearm_o, self.left_wrist_o,
                             self.left_palm_o, self.left_index_base_o,
                             self.left_index_tip_o, self.left_middle_base_o, self.left_middle_tip_o,
                             self.left_third_base_o,
                             self.left_third_tip_o, self.left_end_base_o,
                             self.left_end_tip_o, self.right_shoulder_o, self.right_forearm_o, self.right_wrist_o,
                             self.right_palm_o, self.right_index_base_o,
                             self.right_index_tip_o, self.right_middle_base_o, self.right_middle_tip_o,
                             self.right_third_base_o, self.right_third_tip_o,
                             self.right_end_base_o, self.right_end_tip_o]
        # Optional: Pre-Process the 3-D Coordinates
        if get_ints:
            self.uninterpolated_left_palm = np.asarray(
                vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[18:21]].values[cl1:cl2, :]))
            self.uninterpolated_right_palm = np.asarray(
                vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :]))
            self.uninterpolated_left_forearm = np.asarray(
                vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :]))
            self.uninterpolated_right_forearm = np.asarray(
                vu.norm_coordinates(self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :]))
            self.uninterpolated_left_palm_v, s, self.uninterpolated_left_palm_s = \
                self.calculate_kinematics_from_position(self.uninterpolated_left_palm)
            self.uninterpolated_right_palm_v, s, self.uninterpolated_right_palm_s = \
                self.calculate_kinematics_from_position(self.uninterpolated_right_palm)
            self.uninterpolated_right_forearm_v, s, self.uninterpolated_right_forearm_s = \
                self.calculate_kinematics_from_position(self.uninterpolated_right_forearm)
            self.uninterpolated_left_forearm_v, s, self.uninterpolated_left_forearm_s = \
                self.calculate_kinematics_from_position(self.uninterpolated_left_forearm)
        self.int_gaps = []
        self.int_indices = []
        pos_holder = []
        vel_holder = []
        speed_holder = []
        acc_holder = []
        if preprocess:
            for di, pos in enumerate(self.positions):
                o_positions = np.asarray(pos)
                # Obtain Outlier Indices
                probs = self.probabilities[di]
                prob_outliers = self.threshold_data_with_probabilities(probs, p_thresh=p_thresh)
                # Segment reaches after filtering.
                self.vel_list[di], cc, cd = self.calculate_kinematics_from_position(pos)
                v_outlier_index = np.where(self.vel_list[di] > 1.2)
                # Interpolate and re-sample over outliers where applicable
                possi, num_int, gap_ind = vu.interpolate_3d_vector(np.copy(pos), v_outlier_index, prob_outliers)
                self.int_gaps.append(num_int)
                self.int_indices.append(gap_ind)
                # Filter positions using hamming window
                pos_holder.append(vu.filter_vector_hamming(possi))
                # Calculate kinematics post-smoothing
                v, a, s = self.calculate_kinematics_from_position(pos_holder[di])
                vel_holder.append(v)
                acc_holder.append(a)
                speed_holder.append(s)
                # Obtain and save still-present outliers in the data
                self.outlier_indexes = np.where(s > 1.2)[0]
                self.outlier_list.append(np.squeeze(np.union1d(self.outlier_indexes, self.prob_filter_index)))
                r_outliers = np.where(s < 1.2)[0]
                # Generate RMSE's for outliers
                self.outlier_rmse.append(np.sqrt((pos_holder[di][self.outlier_indexes] -
                                                  o_positions[self.outlier_indexes]) ** 2))
                self.valid_rmse.append(np.sqrt((pos_holder[di][r_outliers] - o_positions[r_outliers]) ** 2))

        # Assign final variables
        self.nose, self.handle, self.left_shoulder, self.left_forearm, self.left_wrist, \
        self.left_palm, self.left_index_base, self.left_index_tip, self.left_middle_base, \
        self.left_middle_tip, self.left_third_base, self.left_third_tip, self.left_end_base, self.left_end_tip, \
        self.right_shoulder, self.right_forearm, self.right_wrist, self.right_palm_v, self.right_index_base, \
        self.right_index_tip, self.right_middle_base, self.right_middle_tip, self.right_third_base, \
        self.right_third_tip, self.right_end_base, self.right_end_tip = pos_holder

        self.nose_v, self.handle_v, self.left_shoulder_v, self.left_forearm_v, self.left_wrist_v, \
        self.left_palm_v, self.left_index_base_v, self.left_index_tip_v, self.left_middle_base_v, \
        self.left_middle_tip_v, self.left_third_base_v, self.left_third_tip_v, self.left_end_base_v, self.left_end_tip_v, \
        self.right_shoulder_v, self.right_forearm_v, self.right_wrist_v, self.right_palm_v, self.right_index_base_v, \
        self.right_index_tip_v, self.right_middle_base_v, self.right_middle_tip_v, self.right_third_base_v, \
        self.right_third_tip_v, self.right_end_base_v, self.right_end_tip_v = vel_holder

        self.nose_s, self.handle_s, self.left_shoulder_s, self.left_forearm_s, self.left_wrist_s, self.left_palm_s, \
        self.left_index_base_s, self.left_index_tip_s, self.left_middle_base_s, self.left_middle_tip_s, \
        self.left_third_base_s, self.left_third_tip_s, self.left_end_base_s, self.left_end_tip_s, self.right_shoulder_s, \
        self.right_forearm_s, self.right_wrist_s, self.right_palm_s, self.right_index_base_s, self.right_index_tip_s, \
        self.right_middle_base_s, self.right_middle_tip_s, self.right_third_base_s, self.right_third_tip_s, \
        self.right_end_base_s, self.right_end_tip_s = speed_holder

        self.nose_a, self.handle_a, self.left_shoulder_a, self.left_forearm_a, self.left_wrist_a, self.left_palm_a, \
        self.left_index_base_a, self.left_index_tip_a, self.left_middle_base_a, self.left_middle_tip_a, self.left_third_base_a, \
        self.left_third_tip_a, self.left_end_base_a, self.left_end_tip_a, self.right_shoulder_a, self.right_forearm_a, \
        self.right_wrist_a, self.right_palm_a, self.right_index_base_a, self.right_index_tip_a, self.right_middle_base_a, \
        self.right_middle_tip_a, self.right_third_base_a, self.right_third_tip_a, self.right_end_base_a, self.right_end_tip_a = acc_holder

        # self.nose_o, self.handle_o, self.left_shoulder_o, self.left_forearm_o, self.left_wrist_o, self.left_palm_o, \
        # self.left_index_base_o, self.left_index_tip_o, self.left_middle_base_o, self.left_middle_tip_o, \
        # self.left_third_base_o, self.left_third_tip_o, self.left_end_base_o, self.left_end_tip_o, self.right_shoulder_o, \
        # self.right_forearm_o, self.right_wrist_o, self.right_palm_o, self.right_index_base_o, self.right_index_tip_o, \
        # self.right_middle_base_o, self.right_middle_tip_o, self.right_third_base_o, self.right_third_tip_o, \
        # self.right_end_base_o, self.right_end_tip_o = self.outlier_list

        # Interpolation filter for graphing
        # Get unfiltered speeds to
        rps, lps, rfs, lfs = self.get_uf_speeds()
        self.left_palm_f_x = np.union1d(np.where(lps > 1.2),
                                        self.threshold_data_with_probabilities(self.left_palm_p, p_thresh=p_thresh))
        self.right_palm_f_x = np.union1d(np.where(rps > 1.2),
                                         self.threshold_data_with_probabilities(self.right_palm_p, p_thresh=p_thresh))
        self.left_forearm_f_x = np.union1d(np.where(lfs > 1.2),
                                           self.threshold_data_with_probabilities(self.left_forearm_p,
                                                                                  p_thresh=p_thresh))
        self.right_forearm_f_x = np.union1d(np.where(rfs > 1.2),
                                            self.threshold_data_with_probabilities(self.right_forearm_p,
                                                                                   p_thresh=p_thresh))
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
                          #            # Class Labels
                          'reach_hand': self.arm_id_list,
                          # Sensor Data
                          'handle_moving_sensor': self.h_moving_sensor, 'lick_beam': self.lick_vector,
                          'reward_zone': self.reward_zone_sensor, 'time_vector': self.time_vector,'lick_vector': self.lick_vector,
                          'response_sensor': self.exp_response_sensor, 'x_rob': self.x_robot, 'y_rob': self.y_robot,
                          'z_rob': self.z_robot,
                          # Sorting variables
                          'rat': self.rat, 'date': self.date, 'session': self.session, 'dim': self.dim
                          }
        df = pd.DataFrame({key: pd.Series(np.asarray(value)) for key, value in self.save_dict.items()})
        df.to_pickle(self.sstr + '/data/' + str(trial_num) + 'save_dict.pkl')
        return

    def whole_segment_outlier_block(self):
        """ Helper function to snag data. """
        self.segment_and_filter_kinematic_block_single_trial(0, -1, get_ints=False)  # Returns all data from session
        # self.segment_data_into_reaches()
        self.get_total_outlier_RMSE_session()
        return

    def get_reach_dataframe_from_block(self):
        """ Function that obtains a trialized pandas dataframe for a provided experimental session. """
        for ix, sts in enumerate(self.trial_start_vectors):
            self.trial_index = sts
            stp = self.trial_stop_vectors[ix]
            self.trial_num = int(ix)
            bi = self.block_video_path.rsplit('.')[0]
            self.sstr = bi + '/trial' + str(ix)
            self.make_paths()
            print('Making dataframe for trial:' + str(ix))
            # Segment, analyze each trial
            self.segment_and_filter_kinematic_block_single_trial(sts, stp)
            self.extract_sensor_data(sts, stp)
            self.segment_reaches_with_position()
            self.analyze_and_classify_reach_vector(sts)
            try:
                self.segment_data_into_reaches()
            except:
                print('No Reaches from this trial')
            self.seg_num = 0
            self.start_trial_indice = []
            self.images = []
        return self.reaching_dataframe, self.valid_rmse_list, self.interpolation_rmse_list, self.outlier_rmse_list

    def plot_interpolation_variables_palm(self, filtype):
        """ Plots displaying feature variables for the left and right palms.
        """
        filename_pos = self.sstr + '/timeseries/' + str(filtype) + 'interpolation_timeseries.png'
        times = np.around((self.time_vector - self.time_vector[0]), 2)
        fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))
        # plt.subplots_adjust(top=0.95, bottom=0.95)
        times_mask_left = np.around((np.asarray(self.time_vector)[self.left_palm_f_x] - self.time_vector[0]), 2)
        times_mask_right = np.around((np.asarray(self.time_vector)[self.right_palm_f_x] - self.time_vector[0]), 2)
        left_outliers = np.zeros(self.left_palm_p.shape[0])
        left_outliers[self.left_palm_o] = 1
        right_outliers = np.zeros(self.right_palm_p.shape[0])
        right_outliers[self.right_palm_o] = 1
        ax1.set_title('Left Palm')
        ax2.set_title('Right Palm')
        ax3.set_title('Probabilities and Experimental Features')
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
                 label='Post-Interpolation and Hamming Filter: Left Palm X')
        ax1.plot(times[2:-1], self.left_palm[2:-1, 1], color='darkred', linestyle='dashed',
                 label='Post-Interpolation and Hamming Filter: Left Palm Y')
        ax1.plot(times[2:-1], self.left_palm[2:-1, 2], color='salmon', linestyle='dashed',
                 label='Post-Interpolation and Hamming Filter: Left Palm Z')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 0], color='navy', linestyle='dashed',
                 label='Post-Interpolation and Hamming Filter: Right Palm X')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 1], color='b', linestyle='dashed',
                 label='Post-Interpolation and Hamming Filter: Right Palm Y')
        ax2.plot(times[2:-1], self.right_palm[2:-1, 2], color='c', linestyle='dashed',
                 label='Post-Interpolation and Hamming Filter: Right Palm Z')
        # Scatter Plot Interpolation Points
        ax1.scatter(times_mask_left, self.left_palm[:, 0][self.left_palm_f_x], color='m', linestyle='dashed',
                    label='Interpolation Intervals')
        ax1.scatter(times_mask_left, self.left_palm[:, 1][self.left_palm_f_x], color='m', linestyle='dashed')
        ax1.scatter(times_mask_left, self.left_palm[:, 2][self.left_palm_f_x], color='m', linestyle='dashed')
        ax2.scatter(times_mask_right, self.right_palm[:, 0][self.right_palm_f_x], color='m', linestyle='dashed',
                    label='Interpolation Intervals')
        ax2.scatter(times_mask_right, self.right_palm[:, 1][self.right_palm_f_x], color='m', linestyle='dashed')
        ax2.scatter(times_mask_right, self.right_palm[:, 2][self.right_palm_f_x], color='m', linestyle='dashed')
        try:
            for tsi, segment_trials in enumerate(self.start_trial_indice):
                ax1.axvline(times[segment_trials], color='black', label='Trial ' + str(tsi))
                ax2.axvline(times[segment_trials], color='black', label='Trial ' + str(tsi))
                ax3.axvline(times[segment_trials], color='black', label='Trial ' + str(tsi))
        except:
            pass
        ax2.set_xlabel('Time (s) ')
        ax2.set_ylabel('Distance (M) ')
        ax1.legend()
        ax2.legend()
        ax3.plot(times, np.mean(self.left_palm_p, axis=1), color='r', label='Left Palm Mean Probability')
        ax3.plot(times, np.mean(self.right_palm_p, axis=1), color='b', label='Right Palm Mean Probability')
        ax3.plot(times, np.mean(self.nose_p, axis=1), color='m', label='Location Probability')
        ax3.plot(times, self.lick_vector / 10, color='y', label='Licks Occurring')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Probabilities')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('M ')
        ax3.legend()
        plt.savefig(filename_pos)
        plt.close()
        return

    def calculate_kinematics_from_position(self, pos_v):
        """ Function that calculates velocity, speed, and acceleration on a per-bodypart basis."""
        v_holder = np.zeros(pos_v.shape)
        a_holder = np.zeros(pos_v.shape)
        pos_v = pos_v[0:len(self.time_vector)]  # Make sure we don't have a leading number ie extra "time" frame
        for ddx in range(0, pos_v.shape[0]):
            v_holder[ddx, :] = np.copy(
                (pos_v[ddx, :] - pos_v[ddx - 1, :]) / (
                        self.time_vector[ddx] - self.time_vector[ddx - 1]))
        # Interpolate over NAN velocity values
        for i in range(0, v_holder.shape[1]):
            mask_v = np.isnan(v_holder[:, i])
            try:
                v_holder[mask_v, i] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_holder[~mask_v, i])
            except:
                pass
        for ddx in range(0, pos_v.shape[0]):
            a_holder[ddx, :] = np.copy(
                (v_holder[ddx, :] - v_holder[ddx - 1, :]) / (self.time_vector[ddx] - self.time_vector[ddx - 1]))
        speed_holder = np.mean(abs(v_holder), axis=1)
        return np.asarray(v_holder), np.asarray(a_holder), np.asarray(speed_holder)

    def get_uf_speeds(self):
        right_speeds = np.mean(self.uninterpolated_right_palm_v, axis=1)
        left_speeds = np.mean(self.uninterpolated_left_palm_v, axis=1)
        right_f_speeds = np.mean(self.uninterpolated_right_forearm_v, axis=1)
        left_f_speeds = np.mean(self.uninterpolated_left_forearm_v, axis=1)
        return right_speeds, left_speeds, right_f_speeds, left_f_speeds

    def quantify_prob_threshold(self, list_of_mean_thresholds, trial_num=0):
        total_speeds = []
        total_gaps = []
        interpolated_speeds = []
        for ix, sts in enumerate(self.trial_start_vectors):
            # Check for null trial, if null don't perform visualization
            if ix > trial_num:
                self.fps = 30
                self.trial_num = int(ix)
                bi = self.block_video_path.rsplit('.')[0]
                self.sstr = bi + '/trial' + str(ix)
                stp = self.trial_stop_vectors[ix]
                self.make_paths()
                gaps = []
                speeds = []
                reaches = []
                for t in list_of_mean_thresholds:
                    self.segment_and_filter_kinematic_block_single_trial(sts, stp)  # Segment filtered data
                    self.extract_sensor_data(sts, stp)  # Extract sensor data

                rs, ls = self.get_uf_speeds()
                speeds.append((rs, ls))
                total_speeds.append(interpolated_speeds)
                total_gaps.append(gaps)
        return total_speeds, total_gaps, speeds, reaches

    def vid_splitter_and_grapher(self, plot=True, timeseries_plot=True, plot_reach=True, trial_num=0):
        """ Function to split and visualize reaching behavior from a given experimental session. """
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
                self.split_trial_video(sts, stp)
                print('Split Trial' + str(ix) + ' Video')
                self.segment_and_filter_kinematic_block_single_trial(sts, stp)
                self.extract_sensor_data(sts, stp)
                self.segment_reaches_with_position()
                rr = self.analyze_and_classify_reach_vector(sts)
                print('Finished Plotting!   ' + str(ix))
                if timeseries_plot:
                    self.plot_interpolation_variables_palm('total')
                if plot:
                    self.plot_predictions_videos(plot_digits=False)
                    self.make_gif_reaching()
                    self.images = []
                    self.make_combined_gif()
                    print('GIF MADE for  ' + str(ix))

                for tsi, segment_trials in enumerate(self.start_trial_indice):
                    if tsi < 1:  # Just split off first detected reach to create the GIF/plots
                        if segment_trials - sts - 30 < 0:  # are we close to the start of the trial?
                            segment_trials += 15  # spacer to keep array values
                        self.split_trial_video(self.trial_index, self.trial_index + self.first_lick_signal + 55,
                                               segment=True, num_reach=tsi)
                        self.segment_and_filter_kinematic_block_single_trial(self.trial_index,
                                                                             self.trial_index + self.first_lick_signal + 55)
                        self.extract_sensor_data(self.trial_index, self.trial_index + self.first_lick_signal + 55)
                        self.segment_data_into_reach_dict(sts)
                        if timeseries_plot:
                            self.plot_interpolation_variables_palm('reach')
                            self.plot_timeseries_features()
                        if plot_reach:
                            self.plot_predictions_videos(segment=True, plot_digits=True)
                            self.make_gif_reaching(segment=True, nr=tsi)
                            print('Reaching GIF made for reach ' + str(tsi) + 'in trial ' + str(ix))
                            self.images = []
                            self.make_combined_gif(segment=True, nr=tsi)
                self.seg_num = 0
                self.start_trial_indice = []
                self.images = []
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
            for l in self.lick:
                if l >= self.time_vector[0]:  # Is this lick happening before or at the first moment of reaching?
                    for trisx, t in enumerate(self.time_vector):
                        if t in self.lick:  # If this time index is in the lick vector
                            self.lick_vector[trisx] = 1  # Mask Array for licking

        self.lick = np.asarray(self.lick)
        return

    def analyze_and_classify_reach_vector(self, trial_frame_lag=20, trial_gap=5, verbose=True):
        """ Function to analyze and classify different reaches into unique behavioral catagories necessary for single-trial
            aggregate comparisons of kinematics.
        """
        rsv = list(np.diff(np.where(self.reach_vector > 0)))
        reaching_indexes = np.flatnonzero(np.diff(np.r_[0, self.reach_vector, 0]) != 0).reshape(-1, 2) - [0, 1]
        self.seg_num = 0
        # Segment the demarcated points of interest into appropriate iterations of reaching behavior
        if rsv[0].any():  # If there are trials present in the reach array
            self.seg_num += 1
            self.start_trial_indice.append(reaching_indexes[0, 0])
            for r in range(0, np.asarray(reaching_indexes).shape[0]):
                try:
                    if reaching_indexes[r, 1] - reaching_indexes[r, 0] > trial_gap:  # is the gap bigger than ?
                        self.start_trial_indice.append(reaching_indexes[r, 0])  # In frames
                    self.seg_num += 1
                except:
                    pass
        if verbose:
            print('Number of total reaches found through position thresholding is ' + str(self.start_trial_indice))

        # Basic Reach Type Classification
        # We Can use the L R reach vector to ID which arm is doing which reaching
        try:
            self.first_lick_signal = np.where(self.lick_vector == 1)[0][0]
        except:
            self.first_lick_signal = 0  # first detected reach + time to lick
        for rei, segs in enumerate(self.start_trial_indice):
            # Arm Checking
            r_reach_vector = self.r_reach_vector[segs - trial_frame_lag:segs + trial_frame_lag]  # This is in idx
            l_reach_vector = self.l_reach_vector[segs - trial_frame_lag:segs + trial_frame_lag]
            if np.nonzero(r_reach_vector):  # More than 2x Window
                self.arm_id_list.append('R')
                if np.nonzero(l_reach_vector):
                    self.arm_id_list.append('B')
            if np.nonzero(l_reach_vector):
                self.arm_id_list.append('L')
            handle_speed = np.mean(self.handle_v[segs:segs + 40, :], axis=1)
            hidx = np.where(abs(handle_speed) > .2)
            hid = np.zeros(handle_speed.shape[0])
            hid[hidx] = 1
            if np.nonzero(hid):
                self.handle_moved = True
            else:
                self.handle_moved = False
            self.trial_cut_vector.append(np.array([segs + self.trial_index, 2, self.arm_id_list[rei]]))
            if self.handle_moved:
                if np.nonzero(self.lick_vector[segs:segs + 40]):
                    self.trial_cut_vector.append(
                        np.array([segs + self.trial_index, 'Successful', self.arm_id_list[rei]]))
                    print('rewarded')
                    if np.nonzero(self.h_moving_sensor[segs:segs + 200]):  # Is there a reaching command going on
                        self.bout_vector.append(np.array(segs))
                        print('Bout')
                else:
                    self.trial_cut_vector.append(
                        np.array([segs + self.trial_index, 'Mis-Grasp', self.arm_id_list[rei]]))
                    print('Mis-Grasp')
            else:
                self.trial_cut_vector.append(np.array([segs + self.trial_index, 'Missed', self.arm_id_list[rei]]))
                if verbose:
                    print('Missed Reach ')
        if self.trial_cut_vector:
            self.block_cut_vector.append(self.trial_cut_vector)
        self.handle_moved = 0
        self.trial_cut_vector = []
        return

    def segment_reaches_with_position(self, posthresh=0.205, v_thresh=0.1, pthresh=0.4):
        """ Function to segment out reaches using a positional and velocity threshold. """
        self.reach_vector = np.zeros((self.left_palm.shape[0]))
        self.l_reach_vector = np.zeros((self.left_palm.shape[0]))
        self.r_reach_vector = np.zeros((self.left_palm.shape[0]))
        self.pos_index = np.zeros((self.left_palm.shape[0]))
        self.r_pos_index = np.zeros((self.left_palm.shape[0]))
        self.l_pos_index = np.zeros((self.left_palm.shape[0]))
        self.bi_pos_index = np.zeros((self.left_palm.shape[0]))
        for rv in range(self.reach_vector.shape[0]):
            # Find all X indices > posthresh (this is past reward zone)
            if self.left_palm[rv, 0] > posthresh:
                if min(self.left_palm_p[rv, :]) > pthresh:
                    if self.left_palm_v[rv, 0] > v_thresh:  # is the palm moving?
                        if self.left_wrist_v[rv, 0] > v_thresh:
                            self.l_pos_index[rv] = 1
                            self.pos_index[rv] = 1
            if self.right_palm[rv, 0] > posthresh:
                if min(self.right_palm_p[rv, :]) > pthresh:
                    if self.right_palm_v[rv, 0] > v_thresh:
                        if self.right_wrist_v[rv, 0] > v_thresh:
                            self.r_pos_index[rv] = 1
                            self.pos_index[rv] = 1
        bi_reach_idx = np.intersect1d(np.nonzero(self.r_pos_index), np.nonzero(self.l_pos_index))
        try:
            self.bi_reach_vector[np.nonzero(bi_reach_idx)] = 1
            self.bi_reach_vector = np.asarray(self.bi_reach_vector)
        except:
            pass
        self.reach_vector[np.nonzero(self.pos_index)] = 1
        self.r_reach_vector[np.nonzero(self.r_pos_index)] = 1
        self.l_reach_vector[np.nonzero(self.l_pos_index)] = 1
        self.r_reach_vector = np.asarray(self.r_reach_vector)
        self.l_reach_vector = np.asarray(self.l_reach_vector)
        return

    def plot_timeseries_features(self):
        """ Function to plot diagnostic time-series plots from single-trial data. """
        filename_pos = self.sstr + '/timeseries/' + 'full_pos_timeseries.png'
        filename_vel = self.sstr + '/timeseries/' + 'full_vel_timeseries.png'
        filename_speed = self.sstr + '/timeseries/' + 'full_speed_timeseries.png'
        xyzpalms = self.sstr + '/timeseries/' + 'xyzpalms_timeseries.png'
        rightvsleftxy = self.sstr + '/timeseries/' + 'rlxypalms_timeseries.png'
        axel = plt.figure(figsize=(10, 4))
        frames_n = np.around(self.time_vector, 2)
        frames = frames_n - frames_n[0]  # normalize frame values by first entry.
        plt.title('Palm Positions During Trial')
        plt.plot(frames, self.right_palm[:, 0], c='g', label='X Right Palm')
        plt.plot(frames, self.right_palm[:, 1], c='g', linestyle='dashed', label='Y Right Palm')
        plt.plot(frames, self.right_palm[:, 2], c='g', linestyle='dotted', label='Z Right Palm')
        plt.plot(frames, self.left_palm[:, 0], c='r', label='X Left Palm')
        plt.plot(frames, self.left_palm[:, 1], c='r', linestyle='dashed', label='Y Left Palm')
        plt.plot(frames, self.left_palm[:, 2], c='r', linestyle='dotted', label='Z Left Palm')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        axel.tight_layout(pad=0.005)
        axel.legend()
        axel.savefig(filename_pos, bbox_inches='tight')
        plt.close()
        axel1 = plt.figure(figsize=(10, 4))
        plt.title('Palm Velocities During Trial')
        plt.plot(frames, self.vel_list[19][:, 0], c='g', label='X Right Palm')
        plt.plot(frames, self.vel_list[19][:, 1], c='g', linestyle='dashed', label='Y Right Palm')
        plt.plot(frames, self.vel_list[19][:, 2], c='g', linestyle='dotted', label='Z Right Palm')
        plt.plot(frames, self.vel_list[7][:, 0], c='r', label='X Left Palm')
        plt.plot(frames, self.vel_list[7][:, 1], c='r', linestyle='dashed', label='Y Left Palm')
        plt.plot(frames, self.vel_list[7][:, 2], c='r', linestyle='dotted', label='Z Left Palm')
        plt.ylim(-1.2, 1.2)
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
        plt.ylim(0, 1.2)
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
                         s=150 + 300 * (self.prob_right_arm[isx]), c='skyblue',
                         alpha=(self.prob_right_arm[isx]), label='Right Palm')
            axel.scatter(self.left_palm[isx - self.lag: isx, 0], self.left_palm[isx - self.lag: isx, 1],
                         self.left_palm[isx - self.lag: isx, 2], marker='.',
                         s=150 + 300 * (self.prob_left_arm[isx]), c='salmon',
                         alpha=(self.prob_left_arm[isx]), label='Left Palm')
            if multi_segment_plot:
                axel.scatter(self.right_forearm[isx - self.lag: isx, 0], self.right_forearm[isx - self.lag: isx, 1],
                             self.right_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.prob_right_arm[isx]), c='royalblue',
                             alpha=(self.prob_right_arm[isx]), label='Right Forearm')
                axel.scatter(self.right_wrist[isx - self.lag: isx, 0], self.right_wrist[isx - self.lag: isx, 1],
                             self.right_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.prob_right_arm[isx]), c='b',
                             alpha=(self.prob_right_arm[isx]), label='Right Wrist')
                axel.scatter(self.left_wrist[isx - self.lag: isx, 0], self.left_wrist[isx - self.lag: isx, 1],
                             self.left_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.prob_left_arm[isx]), c='salmon',
                             alpha=(self.prob_left_arm[isx]), label='Left Wrist ')
                axel.scatter(self.left_forearm[isx - self.lag: isx, 0], self.left_forearm[isx - self.lag: isx, 1],
                             self.left_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.prob_left_arm[isx]), c='r',
                             alpha=(self.prob_left_arm[isx]), label='Left Forearm')
                axel.scatter(self.left_shoulder[isx - self.lag: isx, 0], self.left_shoulder[isx - self.lag: isx, 1],
                             self.left_shoulder[isx - self.lag: isx, 2],
                             s=150 + 300 * (self.prob_left_shoulder[isx]), c='darkred',
                             alpha=(self.prob_left_shoulder[isx]), label='Left Shoulder ')
                axel.scatter(self.right_shoulder[isx - self.lag: isx, 0], self.right_shoulder[isx - self.lag: isx, 1],
                             self.right_shoulder[isx - self.lag: isx, 2],
                             s=100 + 300 * (self.prob_right_shoulder[isx]), c='navy',
                             alpha=(self.prob_right_shoulder[isx]), label='Right Shoulder')

            if draw_skeleton:
                axel.plot([self.right_wrist[isx, 0], self.right_forearm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_forearm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_forearm[isx, 2]],
                          alpha=(self.prob_right_arm[isx]),
                          markersize=55 + 50 * np.mean(self.prob_right_arm[isx])
                          , c='r', linestyle='dashed')
                axel.plot([self.right_forearm[isx, 0], self.right_shoulder[isx, 0]],
                          [self.right_forearm[isx, 1], self.right_shoulder[isx, 1]],
                          [self.right_forearm[isx, 2], self.right_shoulder[isx, 2]],
                          alpha=(self.prob_right_arm[isx]),
                          markersize=55 + 50 * np.mean(self.prob_right_arm[isx])
                          , c='b', linestyle='dashed')
                axel.plot([self.left_forearm[isx, 0], self.left_shoulder[isx, 0]],
                          [self.left_forearm[isx, 1], self.left_shoulder[isx, 1]],
                          [self.left_forearm[isx, 2], self.left_shoulder[isx, 2]],
                          alpha=(self.prob_left_arm[isx]),
                          markersize=55 + 50 * np.mean(self.prob_left_arm[isx]),
                          c='r', linestyle='dashed')
                axel.plot([self.right_wrist[isx, 0], self.right_palm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_palm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_palm[isx, 2]],
                          alpha=(self.prob_right_arm[isx]),
                          markersize=55 + 35 * np.mean(self.prob_right_arm[isx]),
                          c='r', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_forearm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_forearm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_forearm[isx, 2]],
                          alpha=(self.prob_left_arm[isx]),
                          markersize=55 + 50 * np.mean(self.prob_left_arm[isx]),
                          c='c', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_palm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_palm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_palm[isx, 2]],
                          alpha=(self.prob_left_arm[isx]),
                          markersize=55 + 40 * np.mean(self.prob_left_arm[isx]),
                          c='c', linestyle='dashed')
            if plot_digits:
                axel.scatter(self.left_index_base[isx - self.lag: isx, 0], self.left_index_base[isx - self.lag: isx, 1],
                             self.left_index_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='pink',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Index Base ')
                axel.scatter(self.right_index_base[isx - self.lag: isx, 0],
                             self.right_index_base[isx - self.lag: isx, 1],
                             self.right_index_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='dodgerblue',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Index Base ')
                axel.scatter(self.left_middle_base[isx - self.lag: isx, 0],
                             self.left_middle_base[isx - self.lag: isx, 1],
                             self.left_middle_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='pink',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Middle Base ')
                axel.scatter(self.right_middle_base[isx - self.lag: isx, 0],
                             self.right_middle_base[isx - self.lag: isx, 1],
                             self.right_middle_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='skyblue',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Middle Base ')
                axel.scatter(self.left_third_base[isx - self.lag: isx, 0], self.left_third_base[isx - self.lag: isx, 1],
                             self.left_third_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='pink',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Third Base ')
                axel.scatter(self.right_third_base[isx - self.lag: isx, 0],
                             self.right_third_base[isx - self.lag: isx, 1],
                             self.right_third_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='skyblue',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Third Base ')
                axel.scatter(self.left_end_base[isx - self.lag: isx, 0], self.left_end_base[isx - self.lag: isx, 1],
                             self.left_end_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='pink',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left End Base ')
                axel.scatter(self.right_end_base[isx - self.lag: isx, 0],
                             self.right_end_base[isx - self.lag: isx, 1],
                             self.right_end_base[isx - self.lag: isx, 2], marker='D',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='skyblue',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right End Base ')
                axel.scatter(self.left_index_tip[isx - self.lag: isx, 0], self.left_index_tip[isx - self.lag: isx, 1],
                             self.left_index_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='mistyrose',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Index Tip ')
                axel.scatter(self.right_index_tip[isx - self.lag: isx, 0],
                             self.right_index_tip[isx - self.lag: isx, 1],
                             self.right_index_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='skyblue',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Index Tip ')
                axel.scatter(self.left_middle_tip[isx - self.lag: isx, 0],
                             self.left_middle_tip[isx - self.lag: isx, 1],
                             self.left_middle_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='mistyrose',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Middle Tip ')
                axel.scatter(self.right_middle_tip[isx - self.lag: isx, 0],
                             self.right_middle_tip[isx - self.lag: isx, 1],
                             self.right_middle_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='azure',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Middle Tip ')
                axel.scatter(self.left_third_tip[isx - self.lag: isx, 0], self.left_third_tip[isx - self.lag: isx, 1],
                             self.left_third_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='mistyrose',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left Third Tip ')
                axel.scatter(self.right_third_tip[isx - self.lag: isx, 0],
                             self.right_third_tip[isx - self.lag: isx, 1],
                             self.right_third_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='azure',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right Third Tip ')
                axel.scatter(self.left_end_tip[isx - self.lag: isx, 0], self.left_end_tip[isx - self.lag: isx, 1],
                             self.left_end_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_left_digit[isx]), c='mistyrose',
                             alpha=np.mean(self.prob_left_digit[isx]), label='Left End Tip ')
                axel.scatter(self.right_end_tip[isx - self.lag: isx, 0],
                             self.right_end_tip[isx - self.lag: isx, 1],
                             self.right_end_tip[isx - self.lag: isx, 2], marker='_',
                             s=50 + 300 * np.mean(self.prob_right_digit[isx]), c='azure',
                             alpha=np.mean(self.prob_right_digit[isx]), label='Right End Tip ')
            axel.set_xlabel('mm (X)')
            axel.set_ylabel('mm(Y)')
            axel.set_zlabel('mm (Z)')
            axel.view_init(10, -60)
            left_palm_vis = scipy.signal.medfilt(self.left_palm_s, 3)
            right_palm_vis = scipy.signal.medfilt(self.right_palm_s, 3)
            axel2.plot(frames[0:isx], left_palm_vis[0:isx], marker='.', c='red', label='Left Palm Speed')
            axel2.plot(frames[0:isx], right_palm_vis[0:isx], marker='.', c='skyblue', label='Right Palm Speed')
            if self.start_trial_indice:
                for ere, tre in enumerate(self.start_trial_indice):
                    if isx - 10 > tre > isx + 10:
                        axel2.axvline(tre, 0, 1, c='black', markersize=30, linestyle='dotted',
                                      label='REACH ' + str(ere))
                        print('Plotted a reach..')
            axel2.set_xlabel('Time from trial onset (s)')
            axel2.set_ylabel('m/ s')
            axel2.set_ylim(0, 1.2)  # Phys max speed is .9
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
