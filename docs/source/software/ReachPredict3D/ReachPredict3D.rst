ReachPredict3D
====================

Reach Prediction
------------------
Here we outline our software process to predict user-chosen appendage locations with marker-less pose estimation
using DeepLabCut ( https://github.com/DeepLabCut/DeepLabCut ) in 3-D. We choose 27 unique positions to track across
all rats in our experiments. These positions form the basis for our pose estimation during reaching behavior. We then
reconstruct the individual camera 2-D scene spaces into a pre-calibrated euclidean 3-D space using Direct Linear Transformations.
These 3-D predictions and their associated confidence intervals are then saved into the NWB format, per session.

More information about the ReachPredict3D pipeline can be found on our tutorial!

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


.. automodule:: reconstruct_predictions

Main Session 3D loader
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.return_block_kinematic_df
    :members:

Function to save list of dataframes as big DF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.save_kinematics
    :members:

Main Class
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reconstruct_predictions.Reconstruct3D
    :members:

3-D Reach Reconstruction Main Functions
-------------------------------------------
Above are main functions to import, export, and load the various utilities necessary for 3-D reconstruction of our
inferred points. More detail is paid to individual utility functions in the preprocessing software documentation.

