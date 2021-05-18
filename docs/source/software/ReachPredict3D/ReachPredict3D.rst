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

Reach3D
----------
Here we outline the process to transform our 2-D predictions into
3-D euclidean predictions using Direct Linear Transformation (DLT)

Main Session 3D loader
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.return_block_kinematic_df
    :members:

Function to save list of dataframes as big DF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.save_kinematics
    :members:

Video Splitting
-----------------
Here we outline a now redundant process to split experimental videos
from a single video into individual camera videos.

Video Splitter
^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachPredict3D.video_split_batching.mainrun_split
    :members:
