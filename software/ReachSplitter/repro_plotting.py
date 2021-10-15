import pandas as pd
import pdb
import matplotlib.pyplot as plt
import cv2


class ReproPlots:
    def __init__(self, reprojection_filename):
        load_reprojections(reprojection_filename)
        return


def load_reprojections(self, reprojection_filename):
    self.reprojections = pd.read_hdf(reprojection_filename)
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
