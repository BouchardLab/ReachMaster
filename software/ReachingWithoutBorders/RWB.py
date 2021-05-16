"""Functions intended to run RWB (Reaching Without Borders) across multiple session directories across rats."""

import os
import os
import software.preprocessing.reaching_without_borders.rwb as rwb


class RWB:
    def __init__(self):
        self.experimenter = "Brian Gereke"
        self.rat = "RM17"
        self.date = "040120"
        self.root_dir = "D:\\rm_data\\"
        self.pns_dirs = os.listdir(self.root_dir + "pns\\" + self.rat + "\\" + self.date + "\\")
        self.cns_dirs = os.listdir(self.root_dir + "cns\\" + self.rat + "\\" + self.date + "\\")
        self.pns_session_dirs = [d for d in self.pns_dirs if d.startswith('S')]
        self.cns_session_dirs = [d for d in self.cns_dirs if d.startswith('S')]
        self.fetch_rwb()
    # create a separate nwb file per session
    def fetch_rwb(self):
        for s in self.pns_session_dirs:
            session_dir = self.root_dir + "pns\\" + self.rat + "\\" + self.date + "\\" + s
            data_dirs = os.listdir(session_dir)
            # initialize nwb file
            nwb_file = rwb.init_nwb_file(file_name=self.rat + "_"+ self.date + "_" + s,
                                         source_script=__file__,
                                         experimenter=self.experimenter,
                                         session_start_time=0000)
            if 'controller_data' in data_dirs:
                controller_dir = session_dir + "\\controller_data\\"
                nwb_file = rwb.controller_to_nwb(nwb_file, controller_dir)
            if 'config' in data_dirs:
                config_dir = session_dir + "\\config_data\\"
                nwb_file = rwb.config_to_nwb(nwb_file, config_dir)
            if 'videos' in data_dirs:
                video_dir = session_dir + "\\videos\\"
                nwb_file = rwb.link_videos(nwb_file, video_dir)
                try:
                    nwb_file = rwb.link_DLC_predictions(nwb_file, video_dir)
                except:
                    print('Cant add DLC predictions to NWB')
                try:
                    nwb_file = rwb.link_DLC_predictions(nwb_file, video_dir)
                except:
                    print('Cant add filtered DLC predictions to NWB')
                try:
                    nwb_file = rwb.link_3d_coordinates(nwb_file, video_dir)
                except:
                    print('Cant fetch 3d coordinates')
                try:
                    nwb_file = rwb.add_reachsplitter_predictions(nwb_file, video_dir)
                except:
                    print('Could not find classification vector')

            if 'calibration_videos' in data_dirs:
                calibration_video_dir = session_dir + "\\calibration_videos\\"
                nwb_file = rwb.link_calibration_videos(nwb_file, calibration_video_dir)
            if s in self.cns_session_dirs:
                session_dir = self.root_dir + "cns\\" + self.rat + "\\" + self.date + "\\" + s + '\\'
                trodes_name = os.listdir(session_dir)[0]
                nwb_file = rwb.trodes_to_nwb(nwb_file, data_dir=session_dir, trodes_name=trodes_name)
        rwb.save_nwb_file(nwb_file, save_dir=self.root_dir)












