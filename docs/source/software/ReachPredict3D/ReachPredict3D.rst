ReachPredict3D
====================

Reach Prediction
------------------
Here we outline our process to predict user-chosen appendage locations with marker-less pose estimation
using DeepLabCut ( https://github.com/DeepLabCut/DeepLabCut ).


Find Cam Files
^^^^^^^^^^^^^^^^^^

.. automodule:: predictions.find_cam_files
    :members:

Analyze Experimental Videos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: predictions.run_analysis_videos
    :members:

Main function
^^^^^^^^^^^^^^^^^
.. automodule:: predictions.run_main
    :members:


Main Session 3D loader
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.return_block_kinematic_df
    :members:

Function to save list of dataframes as big DF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.save_kinematics
    :members:


Video Splitter
^^^^^^^^^^^^^^^^^
.. automodule:: video_split_batching.mainrun_split
    :members:
