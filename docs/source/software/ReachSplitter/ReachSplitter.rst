*************
ReachSplitter
*************
ReachSplitter is a software pipeline developed to segment and classify coarsely derived trials from the
ReachMaster experimental system. ReachSplitter uses
SciKit Learn and Ruptures to segment and classify individual reaches.
Functions assign a hierarchy of labels to newly segmented reaches.

Trial Classification Hierarchy
--------------------------------
Coarsely segmented trials, found through thresholding POI dev values,
are not fine-grained enough to be useful in any sort of
quantitative behavioral study. As we allow free, unrestricted,
natural reaching behavior there are many opportunities for
misbehavior or failed reaching from our rats.

To avoid this, it is important to segment out specific portions
of the continuous data that contain only reaching behaviors. There is
convenience in assigning various pieces of information, like the hand
of a reach. To automate this tiresome process, ReachSplitter leverages
supervised classification to train models to classify reaches.

Within-Trial Reach Segmentation
-----------------------------------
Time-series anomaly detection and segmentation is a well-established problem.
Python has several packages that perform change-point detection on a multi-variate
data stream like our predicted kinematic positions. One such package is...

As trials are classified in our datastream, an appropriate class defined
is the number of reaches in a trial. From this point, the start indices of each
changepoint are predicted using [method etc]


Overall Reach Extraction Pipeline Structure
----------------------------------------------

ReachSplitter is designed to work across all experimental blocks,
however this depends on integration to the data extraction pipeline. From
this pipeline experimental and 3-D predicted positional data streams
are integrated into a block-by-block classification and segmentation pipeline.


Inputs, Outputs and Local Data File Requirements
--------------------------------------------------------
ReachSplitter is designed to work with two files as input, a dataframe containing predicted
positions for various end-effectors and a dataframe containing raw sensor and interpreted data
from the ReachMaster system. *In order to run ReachSplitter, one needs these files*.
ReachSplitter outputs predicted reach start indices from a given trial block,
hierarchal classification of each reach segment, and metadata necessary for transmission of data into Neurodata
Without Borders. Additional functionality includes direct extraction of data into Neurodata Without Borders.
ReachSplitter is designed to work within Bouchard Lab's data interface system.



Main Extraction Function
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: Classification_Utils::is_tug_no_tug
    :members:






