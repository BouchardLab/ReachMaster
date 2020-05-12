import os
from . import rwb

experimenter = "Brian Gereke"
rat = "RM17"
date = "040120"
root_dir = "D:\\rm_data\\"
pns_dirs = os.listdir(root_dir + "pns\\" + rat + "\\" + date + "\\")
cns_dirs = os.listdir(root_dir + "cns\\" + rat + "\\" + date + "\\")
pns_session_dirs = [d for d in pns_dirs if d.startswith('S')]
cns_session_dirs = [d for d in cns_dirs if d.startswith('S')]

# create a separate nwb file per session
for s in pns_session_dirs:
    session_dir = root_dir + "pns\\" + rat + "\\" + date + "\\" + s
    data_dirs = os.listdir(session_dir)
    # initialize nwb file
    nwb_file = rwb.init_nwb_file(file_name=rat + "_" + date + "_" + s,
                                 source_script=__file__,
                                 experimenter=experimenter,
                                 session_start_time=start_time)
    if 'controller_data' in data_dirs:
        controller_dir = session_dir + "\\controller_data\\"
        nwb_file = rwb.controller_to_nwb(nwb_file, dir=controller_dir)
    if 'config' in data_dirs:
        nwb_file = rwb.config_to_nwb(nwb_file, dir=config_dir)
    if 'videos' in data_dirs:
        nwb_file = rwb.link_video(nwb_file, dir=video_dir)
    if 'calibration_videos' in pns_dirs:
        nwb_file = rwb.link_video(nwb_file, dir=video_dir)
    # trodes data
    if s in cns_session_dirs:
        session_dir = root_dir + "cns\\" + rat + "\\" + date + "\\" + s + '\\'
        trodes_name = os.listdir(session_dir)[0]
        nwb_file = rwb.trodes_to_nwb(nwb_file, data_dir=session_dir, trodes_name=trodes_name)
    rwb.save_nwb_file(nwb_file, save_dir=root_dir)






