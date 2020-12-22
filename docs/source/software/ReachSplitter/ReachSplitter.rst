*************
ReachSplitter
*************
ReachSplitter is a software pipeline developed to segment and classify coarsely derived trials from the
ReachMaster experimental system. ReachSplitter uses
SciKit Learn and Ruptures to segment and classify individual reaches.
Functions assign a hierarchy of labels to newly segmented reaches.

Trial Classification Hierarchy
##############################
[Image]

Within-Trial Reach Segmentation
################################


Overall Reach Extraction Pipeline Structure
############################################

Inputs, Outputs and Local Data File Requirements
################################################
ReachSplitter is designed to work with two files as input, a dataframe containing predicted positions for various end-effectors and a dataframe containing raw sensor and interpreted data from the ReachMaster system. *In order to run ReachSplitter, one needs these files*. 
ReachSplitter outputs predicted reach start indices from a given trial block, hierarchal classification of each reach segment, and metadata necessary for transmission of data into Neurodata Without Borders. Additional functionality includes direct extraction of data into Neurodata Without Borders.

ReachSplitter Main Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: _main.py
    :members:


ReachSplitter Classification Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.Classification_Utils.py
    :members:

ReachSplitter Segmentation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.Segmentation_Utils.py
    :members:

ReachSplitter Visualization Functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.Classification_Visualization.py
    :members:

ReachSplitter DataStream Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.DataStream_Vis_Utils.py
    :members:

ReachSplitter Kinematics Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.Reaching_Kinematics.py
    :members:

ReachSplitter Trial Data Extraction Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.Trial_Data_Utils.py
    :members:

ReachSplitter Trial Data Visualization Functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.TDV.py
    :members:

ReachSplitter Kinematic Data Visualization Functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.ReachSplitter.RKV.py
    :members:




