Data Preprocessing
=======================================

Trodes Data Extraction
---------------------------------------
Here we go over a collection of modules for extracting,
reading, parsing and preprocessing trodes data produced
by various ReachMaster routines. This data forms the backbone of the
experimental dataframe used to contain our sensor data.

Extract trodes data to python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.readTrodesExtractedDataFile3
    :members:

Calibration data parser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.calibration_data_parser
    :members:

Experimental data parser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.experiment_data_parser
    :members:


Controller Data
---------------------------------------
.. automodule:: controller_data.controller_data_parser
    :members:

ReachMaster Configuration Data
---------------------------------------
.. automodule:: config_data.config_parser
    :members:

3D Reconstruction
------------------------
Here we go over the functions that take our 2-D DLC predictions
from 2-D to 3-D using Direct Linear Transformations.
The documentation for DLT can be found http://www.kwon3d.com/theory/dlt/dlt.html
Our lab uses the workflow described here https://biomech.web.unc.edu/dltdv/
Further tutorials can be found in the tutorials section.

3-D Reconstruction
^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.dlt_reconstruct
    :members:

3-D reconstruction main loop
^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.reconstruct_3d
    :members:

Create Probability and Position Objects, per Session
^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.loop_pdata
    :members:

Create Multiprocessing Function Object
^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.r_d_i
    :members:

Find Camera Files for 3D reconstruction
^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.find_cam_files
    :members:

Find Each Session's File Set's
^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.get_file_sets
    :members:

Obtain Each Rat's Kinematic 3-D Reconstructions
^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.get_kinematic_data
    :members:

Check to see if we have all 3 predictions inside the directory
^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.filter_cam_lists
    :members:





