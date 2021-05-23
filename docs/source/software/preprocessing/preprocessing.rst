 Raw Data Preprocessing and Neurodata Without Borders Functionality
========================================================================

Trodes SpikeGadgets DIO/ANALOG Data Extraction
---------------------------------------------------
Here we go over a collection of modules for extracting,
reading, parsing and preprocessing analog and DIO trodes data produced
by various ReachMaster routines during each experiment. This data forms the backbone of the
experimental dataframe containing our various sensor data streams.

Extract trodes data to python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.readTrodesExtractedDataFile3
    :members:

Calibration Data Parser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.calibration_data_parser
    :members:

Experimental DIO/ANALOG Data Parser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: trodes_data.experiment_data_parser
    :members:


Controller Data Loader
---------------------------------------
.. automodule:: controller_data.controller_data_parser
    :members:

ReachMaster Configuration Data Loader
---------------------------------------
.. automodule:: config_data.config_parser
    :members:

3D Reconstruction Functions
=======================================
Here we go over the functions that take our 2-D DLC predictions
from 2-D to 3-D using Direct Linear Transformations.
The documentation for DLT can be found http://www.kwon3d.com/theory/dlt/dlt.html
Our lab uses the workflow described here https://biomech.web.unc.edu/dltdv/
Further tutorials can be found in the tutorials section.

3-D Reconstruction
^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.dlt_reconstruct
    :members:

3-D reconstruction main loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.reconstruct_3d
    :members:

Create Probability and Position Objects, per Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.loop_pdata
    :members:

Create Multiprocessing Function Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.r_d_i
    :members:

Find Camera Files for 3D reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.find_cam_files
    :members:

Find Each Session's File Set's
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.get_file_sets
    :members:

Obtain Each Rat's Kinematic 3-D Reconstructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.get_kinematic_data
    :members:

Check to see if we have all 3 predictions inside the directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: video_data.DLC.Reconstruction.filter_cam_lists
    :members:


Reaching Without Borders Utilities
========================================

Reaching Without Borders is our labs software platform to preprocess and ready behavioral time-series data
for use in NWB format. In this section, we go over basic utilities that import, export, and save each portion
of our data. The full workflow is described in more detail in our tutorial.

Initialize RWB file
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: reaching_without_borders.rwb_utils.init_nwb_file
    :members:







