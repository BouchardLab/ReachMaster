import pandas as pd
import pdb
import pickle
import matplotlib.pyplot as plt
from Analysis_Utils import preprocessing_df as preprocessing
import DataStream_Vis_Utils as utils
from moviepy.editor import *
import skvideo
import cv2
import imageio
import numpy as np
from scipy import ndimage
from errno import EEXIST, ENOENT
ffm_path = 'C:/Users/bassp/OneDrive/Desktop/ffmpeg/bin/'
skvideo.setFFmpegPath(ffm_path)
import skvideo.io


# Public Functions
def loop_over_rats_and_extract_reaches(prediction_dataframe, e_dataframe, dummy_video_path, rat):
    global r_mask, reaching, bout
    save_path = '/Users/bassp/OneDrive/Desktop/Classification Project/reach_thresholds_RM15/'
    # Get rat, date, session for each block we need to process.
    k_dataframe = pd.read_pickle(prediction_dataframe)
    # pdb.set_trace()
    for kk in k_dataframe:
        session = kk.columns[2][1]
        date = kk.columns[2][0][2:4]
        print(session, date)
        R = ReachViz(date, session, e_dataframe, dummy_video_path, prediction_dataframe, rat)
        reaching, mask, bout = R.reach_splitter_threshold(save_dir=save_path)
    return reaching, r_mask, bout

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        os.makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise


def rm_dir(mypath):
    '''Deletes a directory. equivalent to using rm -rf on the command line'''
    try:
        shutil.rmtree(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == ENOENT:
            pass
        else:
            raise


def import_robot_data(df_path):
    df = pd.read_pickle(df_path)
    df = preprocessing(df)
    return df


def norm_coordinates(kin_three_vector, transform=True, filtering=False):
    xkin_three_vector = np.zeros(kin_three_vector.shape
                                 )
    if transform:
        xkin_three_vector[:, 0] = kin_three_vector[:, 0] * -2.5 + .25  # flip x-axis
        xkin_three_vector[:, 1] = kin_three_vector[:, 1] * -0.2 + .25  # flip y-axis
        xkin_three_vector[:, 2] = kin_three_vector[:, 2] * 1.5 + .5
    if filtering:
        xkin_three_vector[:, 0] = ndimage.median_filter(np.copy(xkin_three_vector[:, 0]), size=1)  # flip x-axis
        xkin_three_vector[:, 1] = ndimage.median_filter(np.copy(xkin_three_vector[:, 1]), size=1)  # flip y-axis
        xkin_three_vector[:, 2] = ndimage.median_filter(np.copy(xkin_three_vector[:, 2]), size=1)
    return np.copy(xkin_three_vector)


def rescale_frame(framez, percent=200):
    width = int(framez.shape[1] * percent / 100)
    height = int(framez.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(framez, dim, interpolation=cv2.INTER_AREA)


class ReachViz:
    def __init__(self, date, session, data_path, block_vid_file, kin_path, rat):
        self.rat = rat
        self.date = date
        self.session = session
        self.kinematic_data_path = kin_path
        self.block_exp_df = []
        self.data_path = data_path
        self.sensors = 0
        self.kinematic_block = []
        self.d = []
        self.load_data()  # get exp/kin dataframes
        self.trial_start_vectors = 0
        self.trial_stop_vectors = 0
        self.sstr = 0
        self.block_video_path = block_vid_file
        self.get_block_data()
        self.get_starts_stops()
        self.trial_sensors = 0
        self.h_moving_sensor = 0
        self.exp_response_sensor = 0
        self.time_vector = []
        self.reward_zone_sensor = 0
        self.lick = 0
        self.trial_rewarded = False
        self.bout_vector = []
        self.trial_num = 0
        self.images = []
        self.filename = None
        self.left_palm_velocity = 0
        self.right_palm_velocity = 0
        self.lag = 0
        self.clip_path = 0
        self.fps = 20
        self.lick_vector = 0
        self.reach_vector = 0
        self.prob_index = 0
        self.pos_index = 0
        self.seg_num = 0
        self.start_trial_indice = []
        self.trial_cut_vector = []
        self.block_cut_vector = []
        self.handle_velocity = []
        self.bout_reach = []
        self.handle_moved = 0
        self.gif_save_path = 0
        self.x_robot = []
        self.y_robot = []
        self.z_robot = []
        self.prob_right_index = 0
        self.prob_left_index = 0
        self.l_pos_index = 0
        self.r_pos_index = 0
        self.reprojections = []
        self.reaching_mask = []
        self.bout_flag = False
        self.nose, self.handle, self.body_prob, self.central_body_mass, self.prob_left_shoulder, self.prob_right_shoulder = [0,0,0,0,0,0]
        self.left_shoulder, self.right_forearm, self.left_forearm, self.right_wrist, self.left_wrist, self.right_palm, self.left_palm = [0,0,0,0,0,0,0]
        self.prob_nose, self.prob_right_arm, self.prob_left_arm, self.right_digits, self.left_digits, self.right_shoulder = [0,0,0,0,0,0]
        self.robot_handle_speed, self.reprojected_handle, self.reprojected_nose, self.reprojected_bhandle, self.reprojected_left_palm = [0,0,0,0,0]
        self.reprojected_right_palm, self.reprojected_left_wrist, self.reprojected_right_wrist, self.reprojected_left_shoulder, self.reprojected_right_shoulder = [0,0,0,0,0]
        self.prob_right_index, self.prob_left_index, self.bi_pos_index, self.r_reach_vector, self.l_reach_vector, self.left_prob_index, self.right_prob_index = [0,0,0,0,0,0,0]

    def load_data(self):
        df = import_robot_data(self.data_path)
        self.sensors = df.reset_index(drop=True)
        with (open(self.kinematic_data_path, "rb")) as openfile:
            self.d = pickle.load(openfile)
        return

    def get_block_data(self):
        for kin_items in self.d:
            sess = kin_items.columns.levels[1]
            date = kin_items.columns.levels[2]
            if sess[0] in self.session:
                if date[0][-2:] in self.date:
                    print('Hooked block positions for date  ' + date[0] + '     and session  ' + sess[0])
                    self.kinematic_block = kin_items
        self.block_exp_df = self.sensors.loc[self.sensors['Date'] == self.date].loc[self.sensors['S'] == self.session]
        return

    def filter_data_with_probabilities(self, gen_p_thresh=0.2):
        # If mean of palm probabilities < .2, discard as "rat gone"
        prob_nose = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values, axis=1))
        prob_right_arm = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[45 + 81:48 + 81]].values, axis=1))
        prob_left_arm = np.squeeze(
            np.mean(self.kinematic_block[self.kinematic_block.columns[9 + 81:12 + 81]].values, axis=1))
        bad_nose = np.where(prob_nose < gen_p_thresh)
        bad_right_arm = np.where(prob_right_arm < gen_p_thresh)
        bad_left_arm = np.where(prob_left_arm < gen_p_thresh)
        total_lowp_vector = [np.intersect1d(bad_left_arm, bad_right_arm, bad_nose)]
        self.kinematic_block.values[total_lowp_vector] = 0
        return

    def get_starts_stops(self):
        self.trial_start_vectors = self.block_exp_df['r_start'].values[0]
        self.trial_stop_vectors = self.block_exp_df['r_stop'].values[0]
        return

    def extract_sensor_data_for_reaching_predictions(self, idxstrt, idxstp):
        self.h_moving_sensor = np.copy(self.block_exp_df['moving'].values[0][idxstrt:idxstp])
        self.lick = np.copy(self.block_exp_df['lick'].values[0])  # Lick DIO sensor
        self.reward_zone_sensor = np.copy(self.block_exp_df['RW'].values[0][idxstrt:idxstp])
        self.time_vector = self.block_exp_df['time'].values[0][
                           idxstrt:idxstp]  # extract trial timestamps from SpikeGadgets
        self.exp_response_sensor = self.block_exp_df['exp_response'].values[0][idxstrt:idxstp]

        r, theta, phi, self.x_robot, self.y_robot, self.z_robot = utils.forward_xform_coords(
            self.block_exp_df['x_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['y_pot'].values[0][idxstrt:idxstp],
            self.block_exp_df['z_pot'].values[0][idxstrt:idxstp])
        self.check_licktime()
        return

    def segment_kinematic_block(self, cl1, cl2):
        self.nose = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[6:9]].values[cl1:cl2, :])
        self.handle = np.mean(
            [norm_coordinates(self.kinematic_block[self.kinematic_block.columns[0:3]].values[cl1:cl2, :]),
             norm_coordinates(self.kinematic_block[self.kinematic_block.columns[3:6]].values[cl1:cl2, :])], axis=0)
        self.left_shoulder = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[9:12]].values[cl1:cl2, :])  # 21 end
        self.right_shoulder = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[45:48]].values[cl1:cl2, :])  # 57 end
        self.left_forearm = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        self.right_forearm = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[48:51]].values[cl1:cl2, :])  # 57 end
        self.left_wrist = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[12:15]].values[cl1:cl2, :])  # 21 end
        self.right_wrist = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[51:54]].values[cl1:cl2, :])  # 57 end
        self.left_palm = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[15:18]].values[cl1:cl2, :])
        self.right_palm = norm_coordinates(self.kinematic_block[self.kinematic_block.columns[54:57]].values[cl1:cl2, :])
        self.left_digits = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[18:45]].values[cl1:cl2, :])
        self.right_digits = norm_coordinates(
            self.kinematic_block[self.kinematic_block.columns[57:81]].values[cl1:cl2, :])
        self.prob_right_arm = self.kinematic_block[self.kinematic_block.columns[54 + 81:57 + 81]].values[cl1:cl2,
                              :]
        self.prob_left_arm = self.kinematic_block[self.kinematic_block.columns[18 + 81:21 + 81]].values[cl1:cl2, :]
        self.prob_nose = self.kinematic_block[self.kinematic_block.columns[6 + 81:9 + 81]].values[cl1:cl2, :]
        self.prob_right_shoulder = self.kinematic_block[self.kinematic_block.columns[45 + 81:48 + 81]].values[cl1:cl2,
                                   :]
        self.prob_left_shoulder = self.kinematic_block[self.kinematic_block.columns[9 + 81:12 + 81]].values[cl1:cl2, :]
        self.central_body_mass = np.mean((self.left_shoulder, self.right_shoulder), axis=0)
        self.body_prob = np.mean((np.mean(self.prob_right_shoulder, axis=1),
                                 np.mean(self.prob_left_shoulder, axis=1)), axis=0)
        return

    def compute_palm_velocities_from_positions(self):
        self.left_palm_velocity = np.zeros(self.left_palm.shape)
        self.right_palm_velocity = np.zeros(self.right_palm.shape)
        self.handle_velocity = np.zeros(self.handle.shape)
        self.robot_handle_speed = np.zeros(self.handle.shape)
        for ddx in range(0, self.right_palm_velocity.shape[0]):
            self.robot_handle_speed[ddx, :] = (np.copy(self.x_robot[ddx] - self.x_robot[ddx - 1]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]) +
                                               np.copy((self.y_robot[ddx] - self.y_robot[ddx - 1]) / (
                                                       self.time_vector[ddx] - self.time_vector[ddx - 1]) +
                                                       np.copy((self.z_robot[ddx] - self.z_robot[ddx - 1]) / (
                                                               self.time_vector[ddx] - self.time_vector[
                                                           ddx - 1]) / 3)))
            self.handle_velocity[ddx, :] = np.copy(
                (self.handle[ddx, :] - self.handle[ddx - 1, :]) / (self.time_vector[ddx] - self.time_vector[ddx - 1]))
            self.left_palm_velocity[ddx, :] = np.copy((self.left_palm[ddx, :] - self.left_palm[ddx - 1, :]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
            self.right_palm_velocity[ddx, :] = np.copy((self.right_palm[ddx, :] - self.right_palm[ddx - 1, :]) / (
                    self.time_vector[ddx] - self.time_vector[ddx - 1]))
        for rx in range(self.left_palm_velocity.shape[0]):
            if self.left_prob_index[rx] == 0:
                self.left_palm_velocity[rx, :] = 0
            if self.right_prob_index[rx] == 0:
                self.right_palm_velocity[rx, :] = 0
        np.nan_to_num(self.handle_velocity, 0)
        np.nan_to_num(self.right_palm_velocity, 0)
        np.nan_to_num(self.left_palm_velocity, 0)
        return

    def vid_splitter_classifier(self):
        for ix, sts in enumerate(self.trial_start_vectors):
            bi = self.block_video_path.rsplit('.')[0]
            self.sstr = bi + '/trial' + str(ix)
            self.make_paths()
            self.clip_path = self.sstr + '/videos/trial_video.mp4'
            stp = self.trial_stop_vectors[ix]
            self.split_trial_video(sts, stp)
            print('Saved Video  ' + str(ix))
        return

    def make_gif_reaching(self, segment=False, nr=0):
        if segment:
            try:
                imageio.mimsave(self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4', self.images,
                                fps=self.fps)
                self.gif_save_path = self.sstr + '/videos/reaches/reach_' + str(nr) + '3d_movie.mp4'
            except:
                pdb.set_trace()
        else:
            imageio.mimsave(self.sstr + '/videos/total_3d_movie.mp4', self.images, fps=self.fps)
            self.gif_save_path = self.sstr + '/videos/total_3d_movie.mp4'
        return

    def make_combined_gif(self, segment=False, nr=0):
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
            # Pad borders
            nimg2 = np.ones(img1.shape).astype(np.uint8)
            nimg2[0:img2.shape[0], 0:img2.shape[1], :] = img2
            new_image = np.vstack((img1, nimg2))
            writer.append_data(new_image)
        writer.close()
        return

    def plot_arm_timeseries(self):
        fixit1 = plt.figure(figsize=(10, 6))
        axel = fixit1.add_subplot(1, 1, 1, projection='3d', label='Position of Left and Right Arm during Trial')
        axel.scatter(self.handle[:, 0], self.handle[:, 1], self.handle[:, 2], label='Handle')
        axel.scatter(self.nose[:, 0], self.nose[:, 1], self.nose[:, 2], label='Nose')
        axel.scatter(self.right_palm[:, 0], self.right_palm[:, 1], self.right_palm[:, 2], label='Right Palm')
        axel.scatter(self.left_palm[:, 0], self.left_palm[:, 1], self.left_palm[:, 2], label='Left Palm')
        plt.title('Trial')
        plt.xlabel('mm (X)')
        plt.ylabel('mm (Y)')
        axel.set_zlabel('mm (Z)')
        plt.legend()
        plt.savefig(self.sstr + '/plots/' + 'arm_timeseries.png')
        plt.close()
        return

    def split_trial_video(self, start_frame, stop_frame, segment=False, overlay_reprojections=False, num_reach=0):
        vc = cv2.VideoCapture(self.block_video_path)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        nfr = rescale_frame(frame)
        (h, w) = nfr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if segment:
            self.clip_path = self.sstr + '/videos/reaches/reach_' + str(num_reach) + '_trial.mp4'
        out = cv2.VideoWriter(self.clip_path, fourcc, self.fps, (w, h))
        c = 0
        while rval:
            rval, frame = vc.read()
            nfr = rescale_frame(frame)
            if start_frame <= c < stop_frame:
                if overlay_reprojections:
                    self.load_reprojections()
                    self.fetch_and_draw_reprojections(frame, c)
                out.write(nfr)
            if c == stop_frame:
                break
            c += 1
        vc.release()
        out.release()
        if overlay_reprojections:
            self.plot_reprojections_timeseries(num_reach, start_frame, stop_frame)
        return

    def make_paths(self):
        mkdir_p(self.sstr)
        mkdir_p(self.sstr + '/videos')
        mkdir_p(self.sstr + '/videos/reaches')
        mkdir_p(self.sstr + '/plots')
        mkdir_p(self.sstr + '/plots/reaches')
        mkdir_p(self.sstr + '/large_videos')
        # mkdir_p(self.sstr + '/reprojected_plots')
        return

    def load_reprojections(self):
        self.reprojections = pd.read_hdf('reprojections.h5')
        return

    def slice_reprojections(self, frame_index):
        self.reprojected_right_palm = self.reprojections.values[frame_index, 102:108]
        self.reprojected_left_palm = self.reprojections.values[frame_index, 30:36]
        self.reprojected_handle = self.reprojections.values[frame_index, 0:6]
        self.reprojected_bhandle = self.reprojections.values[frame_index, 6:12]
        self.reprojected_nose = self.reprojections.values[frame_index, 12:18]
        self.reprojected_left_shoulder = self.reprojections.values[frame_index, 18:24]
        self.reprojected_right_shoulder = self.reprojections.values[frame_index, 90:96]
        self.reprojected_right_wrist = self.reprojections.values[frame_index, 102:108]
        self.reprojected_left_wrist = self.reprojections.values[frame_index, 24:30]
        return

    def fetch_and_draw_reprojections(self, img, frame_index, rad=10):
        self.slice_reprojections(frame_index)
        reprojected_parts = [self.reprojected_left_palm, self.reprojected_left_wrist, self.reprojected_left_shoulder,
                             self.reprojected_nose,
                             self.reprojected_handle, self.reprojected_left_palm, self.reprojected_left_wrist,
                             self.reprojected_left_shoulder]
        red_solid_right = (0, 0, 255)
        red_less_right = (0, 100, 255)
        red_even_less_right = (0, 165, 255)
        blue_nose = (255, 0, 0)
        blue_handle = (255, 150, 0)
        green_solid_left = (0, 255, 0)
        green_less_left = (100, 255, 0)
        green_sless_left = (255, 255, 0)
        colors = [red_solid_right, red_less_right, red_even_less_right, blue_nose, blue_handle, green_solid_left,
                  green_less_left, green_sless_left]
        for isx, parts in enumerate(reprojected_parts):
            col = int(parts[0])
            row = int(parts[1])
            dcolor = colors[isx]
            pdb.set_trace()
            cv2.rectangle(img, (row - 8, col - 8), (row + 8, col + 8), dcolor, thickness=20)
        return

    def plot_reprojections_timeseries(self, nr, start_idz, stop_idz):
        plt.plot()
        plt.savefig(self.sstr + '/reprojected_plots/trial_' + str(nr) + 'reprojections.png')
        plt.close()
        return

    def reach_splitter_threshold(self, trial_num=0, save_dir=False):
        global mask_df, reaching_df, bout_df
        for ix, sts in enumerate(self.trial_start_vectors):
            # Check for null trial, if null don't perform visualization
            try:
                if ix >= trial_num:
                    self.trial_num = int(ix)
                    try:
                        stp = self.trial_stop_vectors[ix]
                    except:
                        stp = self.trial_start_vectors[ix] + 50
                    self.filter_data_with_probabilities()
                    self.segment_kinematic_block(sts, stp)
                    self.extract_sensor_data_for_reaching_predictions(sts, stp)
                    self.segment_reaches_with_position()
                    self.compute_palm_velocities_from_positions()
                    self.analyze_reach_vector(sts)
                    self.seg_num = 0
                    self.start_trial_indice = []
                    self.images = []
            except:
                print('Problem with reaches on block' + str(ix))
                self.seg_num = 0
                self.start_trial_indice = []
                self.images = []
        try:
            if save_dir:
                mkdir_p(save_dir)
            reaching_df = pd.DataFrame(self.block_cut_vector)
            reaching_df.to_csv(
                save_dir + 'reaching_vector_RM' + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                index=False, header=False)
            print('Saved reaching indices and bouts for block..' + str(self.date) + '...' + str(self.session))
            mask_df = pd.DataFrame(self.reaching_mask)
            mask_df.to_csv(save_dir + 'mask_df_vector_RM' + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                           index=False, header=False)
            bout_df = pd.DataFrame(self.bout_vector)
            bout_df.to_csv(save_dir + 'bout_df_RM' + str(self.rat) + str(self.date) + str(self.session) + '.csv',
                           index=False, header=False)
        except:
            pdb.set_trace()
        return reaching_df, mask_df, bout_df

    def vid_splitter_and_grapher(self, gif_plot=True, trial_num=0):
        # for each trial, make a directory, cut and save a video, and save plotss in sub-dir
        # To-Do: Input Classifier Array, cut on segment estimates
        for ix, sts in enumerate(self.trial_start_vectors):
            # Check for null trial, if null don't perform visualization
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
                self.filter_data_with_probabilities()
                self.segment_kinematic_block(sts, stp)
                self.extract_sensor_data_for_reaching_predictions(sts, stp)
                self.segment_reaches_with_position()
                self.compute_palm_velocities_from_positions()
                self.analyze_reach_vector(sts)
                print('Finished Plotting!   ' + str(ix))
                if gif_plot:
                    self.plot_predictions_videos()
                    self.make_gif_reaching()
                    self.images = []
                    self.make_combined_gif()
                    print('GIF MADE for  ' + str(ix))
                    for tsi, segment_trials in enumerate(self.start_trial_indice):
                        # Check segment value to see how close we are to ends
                        if segment_trials - sts - 30 < 0:  # are we close to the start of the trial?
                            segment_trials += 15  # spacer to keep array values ok
                        self.fps = 10
                        self.split_trial_video(segment_trials, segment_trials + 30, segment=True, num_reach=tsi)
                        self.segment_kinematic_block(segment_trials, segment_trials + 30)
                        self.extract_sensor_data_for_reaching_predictions(segment_trials, segment_trials + 30)
                        self.compute_palm_velocities_from_positions()
                        self.plot_predictions_videos(segment=True, seg_vector=segment_trials, trial_start=sts)
                        self.make_gif_reaching(segment=True, nr=tsi)
                        self.images = []
                        self.make_combined_gif(segment=True, nr=tsi)
                        print('Reaching GIF made for reach ' + str(tsi) + 'in trial ' + str(ix))
                self.seg_num = 0
                self.start_trial_indice = []
                self.images = []
        reaching_trial_indices_df = pd.DataFrame(self.block_cut_vector)
        reaching_trial_indices_df.to_csv(
            'reaching_vector_RM' + str(self.rat) + str(self.rat) + str(self.date) + str(self.session) + '.csv',
            index=False, header=False)
        print('Saved reaching indices and bouts for block..' + str(self.date) + str(self.session))
        return

    def set_lag_plot_term(self, isx):
        if isx == 1:
            self.lag = 1
        if isx == 2:
            self.lag = 2
        if isx == 3:
            self.lag = 3
        if isx == 4:
            self.lag = 4
        return

    def plot_predictions_videos(self, segment=False, seg_vector=False, trial_start=False, multi_segment_plot=True,
                                draw_skeleton=True):
        tldr = self.handle.shape[0]  # set range in single trial
        tlde = 0
        if segment:
            tlde = 0
            tldr = 30
        for isx in range(tlde, tldr):
            self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            self.set_lag_plot_term(isx)
            fixit1 = plt.figure(figsize=(52 / 4, 44 / 4))
            axel = plt.subplot2grid((52, 44), (0, 0), colspan=44, rowspan=46, projection='3d',
                                    label='Position of Left and Right Arm during Trial')
            axel2 = plt.subplot2grid((52, 44), (46, 7), colspan=40, rowspan=8)
            fixit1.suptitle('Detected Reaching Trial ')
            axel.set_xlim(0, 0.4)
            axel.set_ylim(0, 0.5)
            axel.set_zlim(0.4, 0.60)
            axel.plot([0.205, 0.205], [0, 0.5], [0.5, 0.5], c='r', linestyle='dashed', linewidth=5,
                      markersize=10, label='Trial Reward Zone')  # make a 3-D line for our "reward zone"
            X, Y = np.meshgrid(np.linspace(0.2, 0, 10), np.linspace(0, 0.5, 10))
            Z = np.reshape(np.linspace(0.4, 0.4, 100), X.shape)
            axel.plot_surface(X, Y, Z, alpha=0.3, zorder=0)
            axel.grid(False)
            axel.scatter(self.handle[isx - self.lag: isx, 0], self.handle[isx - self.lag, 1],
                         self.handle[isx - self.lag:isx, 2],
                         s=400, c='m', label='Handle')
            axel.scatter(self.central_body_mass[isx - self.lag: isx, 0], self.central_body_mass[isx - self.lag: isx, 1],
                         self.central_body_mass[isx - self.lag: isx, 2],
                         s=150 + 300 * np.mean(self.prob_left_arm[isx, :]), c='black',
                         alpha=np.mean(self.prob_left_arm[isx, :]), label='Rat' + str(self.rat))
            axel.scatter(self.nose[isx - self.lag: isx, 0], self.nose[isx - self.lag, 1],
                         self.nose[isx - self.lag:isx, 2],
                         s=50 + 100 * np.mean(self.prob_nose[isx, :]), alpha=np.mean(self.prob_nose[isx, :]),
                         label='Nose')
            axel.scatter(self.right_palm[isx - self.lag: isx, 0], self.right_palm[isx - self.lag: isx, 1],
                         self.right_palm[isx - self.lag: isx, 2],
                         s=150 + 300 * np.mean(self.prob_right_arm[isx, :]), c='darkred',
                         alpha=np.mean(self.prob_right_arm[isx, :]), label='Right Palm')
            axel.scatter(self.left_palm[isx - self.lag: isx, 0], self.left_palm[isx - self.lag: isx, 1],
                         self.left_palm[isx - self.lag: isx, 2],
                         s=150 + 300 * np.mean(self.prob_left_arm[isx, :]), c='lawngreen',
                         alpha=np.mean(self.prob_left_arm[isx, :]), label='Left Palm')

            if multi_segment_plot:
                axel.scatter(self.right_forearm[isx - self.lag: isx, 0], self.right_forearm[isx - self.lag: isx, 1],
                             self.right_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * np.mean(self.prob_right_arm[isx, :]), c='#F39C12',
                             alpha=np.mean(self.prob_right_arm[isx, :]), label='Right Forearm')
                # axel.scatter(self.right_forearm[isx - self.lag: isx, 0], self.right_forearm[isx - self.lag: isx, 1],
                #             self.right_forearm[isx - self.lag: isx, 2],
                #             s=100 + 300 * np.mean(self.prob_right_arm[isx, :]), c='yellow',
                #             alpha=np.mean(self.prob_right_arm[isx, :]), label='Right Shoulder')
                # axel.scatter(self.left_forearm[isx - self.lag: isx, 0], self.left_forearm[isx - self.lag: isx, 1],
                #             self.left_forearm[isx - self.lag: isx, 2],
                #             s=100 + 300 * np.mean(self.prob_left_arm[isx, :]), c='c',
                #             alpha=np.mean(self.prob_left_arm[isx, :]), label='Left Shoulder')
                axel.scatter(self.right_wrist[isx - self.lag: isx, 0], self.right_wrist[isx - self.lag: isx, 1],
                             self.right_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * np.mean(self.prob_right_arm[isx, :]), c='r',
                             alpha=np.mean(self.prob_right_arm[isx, :]), label='Right Wrist')
                axel.scatter(self.left_forearm[isx - self.lag: isx, 0], self.left_forearm[isx - self.lag: isx, 1],
                             self.left_forearm[isx - self.lag: isx, 2],
                             s=100 + 300 * np.mean(self.prob_left_arm[isx, :]), c='c',
                             alpha=np.mean(self.prob_left_arm[isx, :]), label='Left Forearm')
                axel.scatter(self.left_wrist[isx - self.lag: isx, 0], self.left_wrist[isx - self.lag: isx, 1],
                             self.left_wrist[isx - self.lag: isx, 2],
                             s=150 + 300 * np.mean(self.prob_left_arm[isx, :]), c='aquamarine',
                             alpha=np.mean(self.prob_left_arm[isx, :]), label='Left Wrist ')
            if draw_skeleton:
                # Draw lines from right shoulder to forearm to wrist to palm
                # axel.plot([self.right_shoulder[isx, 0], self.right_forearm[isx, 0]],
                #          [self.right_shoulder[isx, 1], self.right_forearm[isx, 1]],
                #          [self.right_shoulder[isx, 2], self.right_forearm[isx, 2]],
                #          alpha=np.mean(self.prob_right_arm[isx, :]), markersize=15 + 50 * np.mean(self.prob_right_arm[isx, :]),
                #          c='g', linestyle='dashed')
                axel.plot([self.right_wrist[isx, 0], self.right_forearm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_forearm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_forearm[isx, 2]],
                          alpha=np.mean(self.prob_right_arm[isx, :]),
                          markersize=15 + 50 * np.mean(self.prob_right_arm[isx, :])
                          , c='g', linestyle='dashed')
                axel.plot([self.right_wrist[isx, 0], self.right_palm[isx, 0]],
                          [self.right_wrist[isx, 1], self.right_palm[isx, 1]],
                          [self.right_wrist[isx, 2], self.right_palm[isx, 2]],
                          alpha=np.mean(self.prob_right_arm[isx, :]),
                          markersize=15 + 30 * np.mean(self.prob_right_arm[isx, :]),
                          c='g', linestyle='dashed')
                # axel.plot([self.left_shoulder[isx, 0], self.left_forearm[isx, 0]],
                #          [self.left_shoulder[isx, 1], self.left_forearm[isx, 1]],
                #          [self.left_shoulder[isx, 2], self.left_forearm[isx, 2]],
                #          alpha=np.mean(self.prob_left_arm[isx, :]), markersize=15 + 50 * np.mean(self.prob_left_arm[isx, :]),
                #          c='r', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_forearm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_forearm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_forearm[isx, 2]],
                          alpha=np.mean(self.prob_left_arm[isx, :]),
                          markersize=15 + 50 * np.mean(self.prob_left_arm[isx, :]),
                          c='r', linestyle='dashed')
                axel.plot([self.left_wrist[isx, 0], self.left_palm[isx, 0]],
                          [self.left_wrist[isx, 1], self.left_palm[isx, 1]],
                          [self.left_wrist[isx, 2], self.left_palm[isx, 2]],
                          alpha=np.mean(self.prob_left_arm[isx, :]),
                          markersize=15 + 30 * np.mean(self.prob_left_arm[isx, :]),
                          c='r', linestyle='dashed')
            plt.title('Frame' + str(isx))
            plt.xlabel('mm (X)')
            plt.ylabel('mm (Y)')
            axel.set_zlabel('mm (Z)')
            axel.view_init(0, -90)
            axel.legend()
            frames = np.linspace(0, self.handle.shape[0], self.handle.shape[0])
            axel2.plot(frames[0:isx], self.lick_vector[0:isx], c='b', label='Lick')
            axel2.plot(frames[0:isx], self.right_palm_velocity[0:isx, 0],
                       markersize=15 + 30 * np.mean(self.prob_right_arm[isx, :]), c='r',
                       linestyle='dashed', label='Right Arm X Velocity')
            axel2.plot(frames[0:isx], self.left_palm_velocity[0:isx, 0], c='g',
                       markersize=15 + 30 * np.mean(self.prob_left_arm[isx, :]), label='Left Arm X Velocity')
            axel2.plot(frames[0:isx], self.right_palm_velocity[0:isx, 1],
                       markersize=15 + 30 * np.mean(self.prob_right_arm[isx, :]), c='r',
                       linestyle='dashed', label='Right Arm Y Velocity')
            axel2.plot(frames[0:isx], self.left_palm_velocity[0:isx, 1], c='g',
                       markersize=15 + 30 * np.mean(self.prob_left_arm[isx, :]), label='Left Arm Y Velocity')
            if self.start_trial_indice:
                for ere, tre in enumerate(self.start_trial_indice):
                    if isx - 10 > tre > isx + 10:
                        axel2.axvline(tre, 0, 3, c='black', markersize=30, linestyle='dotted',
                                      label='REACH ' + str(ere))
                        print('Plotted a reach..')
            plt.xlabel('Frames')
            plt.ylabel('m/ s')
            axel2.set_ylim(0, 3)
            axel2.set_xlim(0, self.handle.shape[0] + 5)
            axel2.legend(loc="upper right")
            self.filename = self.sstr + '/plots/' + str(isx) + 'palm_prob_timeseries.png'
            if segment:
                plt.close()
            else:
                plt.savefig(self.filename)
                plt.close()
            self.images.append(imageio.imread(self.filename))
        return

    def check_licktime(self):
        self.lick_vector = np.zeros((len(self.time_vector)))
        self.lick = list(np.around(np.array(self.lick), 2))
        self.time_vector = list(np.around(np.array(self.time_vector), 2))
        for l in self.lick:
            if l >= self.time_vector[0]:
                for trisx, t in enumerate(self.time_vector):
                    if t in self.lick:
                        self.lick_vector[trisx] = 1
                if l <= self.time_vector[25]:  # is there a rapid reward? Bout-like interactions...
                    self.bout_flag = True
                if l <= self.time_vector[-1]:
                    self.trial_rewarded = True
                    break
        return

    def handle_moving_signal(self, seg, handle_thresh=0.001):
        print(seg)
        if np.all(self.robot_handle_speed[seg:seg + 20] > handle_thresh):
            self.handle_moved = 1
        return

    def analyze_reach_vector(self, id_start, trial_frame_lag=15, int_num=15, verbose=False):
        rsv = list(np.diff(np.where(self.reach_vector > 0)) - 1)
        rind = np.where(self.reach_vector > 0)[0]
        if rsv[0].any():  # If we detect a trial
            try:
                self.start_trial_indice.append(rind[0] - trial_frame_lag + id_start)
                self.seg_num = 1
            except:
                pdb.set_trace()
            cxx = 0
            if rsv[0][1:-1].any():
                for ma in np.nditer(rsv[0][1:-1]):
                    cxx += 1
                    if ma > int_num:
                        self.seg_num += 1
                        self.start_trial_indice.append(rind[cxx] - trial_frame_lag + id_start)
        if verbose:
            print('Number of total reaches found through position thresholding is ' + str(self.seg_num))
        # Basic Reach Type Classification
        for segs in self.start_trial_indice:
            rds = 0
            for rzs in self.reward_zone_sensor[segs - id_start:segs - id_start + 40]:  # check redzones
                if rzs == 1:
                    rds = 1
            self.handle_moving_signal(segs - id_start)
            if self.handle_moved:
                if rds:
                    self.trial_cut_vector.append([segs, 1, id_start])
                    if verbose:
                        print('Rewarded Reach ')
                    if self.bout_flag:
                        self.bout_vector.append(segs)
                        print('Bout!')
                else:
                    self.trial_cut_vector.append([segs, 2, id_start])
                    if verbose:
                        print('Failed Reach ')
            else:
                self.trial_cut_vector.append([segs, 3, id_start])
                if verbose:
                    print('Missed Reach ')
        self.block_cut_vector.append(self.trial_cut_vector)
        self.handle_moved = 0
        self.trial_cut_vector = []
        return

    def segment_reaches_with_position(self, pthresh=0.8, posthresh=0.22):
        self.prob_index = np.zeros((self.left_palm.shape[0]))
        self.right_prob_index = np.zeros((self.left_palm.shape[0]))
        self.left_prob_index = np.zeros((self.left_palm.shape[0]))
        self.reach_vector = np.zeros((self.left_palm.shape[0]))
        self.l_reach_vector = np.zeros((self.left_palm.shape[0]))
        self.r_reach_vector = np.zeros((self.left_palm.shape[0]))
        self.pos_index = np.zeros((self.left_palm.shape[0]))
        self.r_pos_index = np.zeros((self.left_palm.shape[0]))
        self.l_pos_index = np.zeros((self.left_palm.shape[0]))
        self.bi_pos_index = np.zeros((self.left_palm.shape[0]))
        for rv in range(self.prob_left_arm.shape[0]):
            if np.median(self.prob_left_arm[rv, :]) > pthresh or np.median(self.prob_right_arm[rv, :]) > pthresh:
                self.prob_index[rv] = 1
                if np.median(self.prob_left_arm[rv, :]) > pthresh:
                    self.left_prob_index[rv] = 1
                if np.median(self.prob_right_arm[rv, :]) > pthresh:
                    self.right_prob_index[rv] = 1
            # Find all X indices > posthresh (this is past reward zone)
            if self.left_palm[rv, 0] > posthresh:
                self.l_pos_index[rv] = 1
                self.pos_index[rv] = 1
            if self.right_palm[rv, 0] > posthresh:
                self.r_pos_index[rv] = 1
                self.pos_index[rv] = 1
        # Check and see if any of the prob indices overlay with the X indices
        mask_id = (self.prob_index == 1) & (self.pos_index == 1)
        right_mask_id = np.intersect1d(self.r_pos_index, self.right_prob_index)
        left_mask_id = np.intersect1d(self.l_pos_index, self.left_prob_index)
        for sx, m in enumerate(mask_id):
            if m:
                try:
                    if right_mask_id[sx] == 1:
                        self.r_reach_vector[sx] = 1
                        if left_mask_id[sx] == 1:
                            self.bi_pos_index[sx] = 1
                    if left_mask_id[sx] == 1:
                        self.l_reach_vector[sx] = 1
                except:
                    continue
                self.reach_vector[sx] = 1
        total_rv = np.vstack((self.r_reach_vector, self.l_reach_vector, self.reach_vector))
        self.reaching_mask.append(total_rv)
        return
