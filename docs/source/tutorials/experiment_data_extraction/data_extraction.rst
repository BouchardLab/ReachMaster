Organize and clean experimental data using our data extraction pipeline
==========================================

Overview of Problem
-----------------------------------------

While our experiments generate high-dimensional data that is useful in the analysis of complex motor behavior,
extracting and cleaning these large data sources can be a challenge.
Fortunately, we have developed a extraction pipeline intended to automate each of the
extraction and pre-processing steps necessary to obtain quality experimental data.

Obtaining and Organizing Kinematics
-----------------------------------------
We rely on DeepLabCut to automate kinematic predictions for (currently) 27 regions of interest during an experiment.
These regions are selected to best represent both gross and fine bi-manual reaching movements.
Predictions are generated across entire experiments, then filtered using experimental parameters such as trial times.

Pre-processing robot data sources
-----------------------------------------

Syncing Hardware
-----------------------------------------

Applying 3-D Reconstruction
-----------------------------------------

DeepLabCut
-----------------------------------------

Final Result
-----------------------------------------
