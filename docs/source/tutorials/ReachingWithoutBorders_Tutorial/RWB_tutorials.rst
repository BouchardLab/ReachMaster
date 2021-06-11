Reaching Without Borders
=============================
Reaching Without Borders is the Bouchard Lab's name for our custom-built software platform to house our lab's
behavioral and ECOG electrophysiology data from our reaching experiments.


What's in our RWB file?
------------------------

ReachMaster saves all data (and every change in the data at each preprocessing step in our pipeline !) inside of
a NeuroData Without Borders file. These files are made to be easy exchangeable, allowing for seamless analysis between individual
scientists. When we say all of our data, we mean all of it..

What gets saved in our RWB file, for every session?
-----------------------------------------------------


Sensor Data
------------
ReachMaster saves a variety of sensor data from our digital/analog signal array as well as from a pair of Arduino micro-controllers.

Digital signals include a licking beam-break sensor, experimental triggers and current signals from our robot. Analog data
from our robot potentiometers is also recorded. Arduino Teensy microcontrollers signal commands to the robot and other micro-systems
inside ReachMaster. These commands, such as when an experiment ends, when the handle enters the reward zone, and when the handle is
moving is saved.

Video Data
-----------
ReachMaster records reaching behavior in the arena using 3 high-speed XIMEA video cameras. This data is linked in RWB as
.mp4 files for individual cameras.

DLT calibration data
---------------------
Wand calibration is recorded using the video camera system described above at a regular interval (daily). These video files
are linked in the RWB file.

DeepLabCut Predicted Positions
--------------------------------
DeepLabCut is used to generate predictions for 27 body parts on single rats performing reaching behaviors across an entire
behavioral session. Body parts include two points of reference for the handle, nose, and markers on each arm for the shoulder,
forearm, wrist, palm, as well as base and tip position markers for digits. This is done for each camera.
More information about our DeepLabCut prediction system can be found in the ReachPredict3D documentation.

This data is then saved to our RWB file.
Optionally, one can filter the DeepLabCut positions using several methods, including Gaussian kernel filtering. This
data is also saved to the RWB file and used in the below 3-D reconstruction, if specified.

3-D Predicted Positions
-------------------------
After DeepLabCut is used to extract the 2-D predicted positions across each of our 3 cameras, we then use DLT to
reconstruct the positions from 2-D to 3-D. The reconstructed 3-D positions are then saved to the RWB file, along with
the DLT file used to perform the reconstruction.

Predicted Reach Metadata
--------------------------
ReachMaster generates predictions for our segmented, within trial-block behavior by assessing the state of the camera trial
trigger. When the camera systems deviate from specific regions of interest (ROI), a signal is triggered that a tentative
trial has been initiated. Our lab utilizes machine learning approaches to automatically classify these coarse trial interactions
into more fine-grained, individual reaching behaviors. For more on our classification strategy and the conspecific identity of the
reaching classification vector, refer to the ReachSplitter documentation. This classification vector is stored inside the RWB
vector.

Predicted Reach Timestamp
---------------------------
Within a single, coarse trial we can have many reaches. In order to segment the time-series into an appropriate
set of motifs, we utilize both thresholding and unsupervised machine learning to generate the exact time-stamps of the start of
a reach. These indices are saved in the NWB file.



