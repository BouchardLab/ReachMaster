# ReachMaster software
All software packages used to operate the ReachMaster robotic system and run common analyses on the resulting data. 

reach_master - the graphical user interface used to run experiments and collect data

controller_scripts - Arduino scripts used to perform various tasks on the experiment and robot microcontrollers (may be absorbed by reach_master)

robot_calibration - R and Python scripts for parsing and analysing robot calibration data (may be absorbed by reachmaster and preprocessing)

preprocessing - utilities for reading, parsing, and exporting various types of ReachMaster data

ReachingWithoutBorders - our Neurodata Without Borders pipeline

ReachSplitter - software pipeline to extract discrete reaches from continous data

ReachPredict3D - pipeline to extract DeepLabCut-generated predictions of bodyparts, then
transform these predictions into 3-D.

visualizations - sets of visualizations to determine accuracy and scale of 
our predicted coordinates in 2 and 3-D, using ground truth labels.
