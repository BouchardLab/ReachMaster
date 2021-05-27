Camera Calibration
=======================================

Transforming multiple estimates of 2-D pose (camera scenes) into a 3-D pose (real life, euclidean coordinates) is a
classic problem within computer vision. We prefer our method for pose estimation to be robust across a large volume,
amenable to a 3+ camera configuration, and practical in nature. This has led Bouchard Lab to use the
field calibration method of DLT triangulation as a means of producing 3-D pose estimations from our multiple 2-D pose data sources.

Our solution is to use a wand-based field calibration system that requires us to capture daily or weekly video of
a two-point wand. Two discrete end points are then tracked across the camera frames using DLTdv, then the resulting
tracked endpoints are uses to estimate the 3-D reaching volume using EasyWand5.

Pose Estimation with DLT
------------------------
A pose, or a 2-D scene, is recorded in a single frame of a video. We record our rat behavior from 3 cameras placed to
record a maximal volume of our reaching space. As our scene stays relatively constant frame to frame
(our cameras are fixed in place), we would like to transform our 2-D scenes into a full natural 3-D space.

Direct Linear Transformation (DLT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct Linear Transformation is an algorithmic approach to transform multiple 2-D poses, or scenes,  into a 3-D volume.
Here, we overview some basic theory behind DLT. For a more detailed, mathematical conversation,
refer to http://www.kwon3d.com/theory/dlt/dlt.html#3d .


The projected plane (camera) is related to an object within the frame. This object (a wand, for our purpose!) is captured
across each scene. We seek to relate the object-space reference frame to the image-plane reference frame using the fact that
the mapping between images of our object is co-linear. Describing this mapping, then, is our challenge.

Briefly, there exists some transformation for our system that will transform multiple image-plane coordinates
([u1,v1], [u2,v2]..) into object-space coordinates [x,y,z]. This is done by finding, using algorithmic approaches or
Sparse Bundle Adjustment, transformations of low or lowest error.

EasyWand and DLTdv7: Field Calibration Software for the People
-------------------------------------------------------------------
Credit to Hedrick Lab for all calibration software, instructions, and examples!
Our lab uses the software DLTdv to label daily approximately minute-long pre-recorded wand calibration videos.

EasyWand software
^^^^^^^^^^^^^^^^^^

.. image:: /art/DLTDV.png
	:align: center
	:width: 400

Tutorials and software can be found at https://biomech.web.unc.edu/dltdv/ .

DLTDV software
^^^^^^^^^^^^^^^
.. image:: /art/DLTDV_I.png
	:align: center
	:width: 400

EasyWand Procedure
^^^^^^^^^^^^^^^^^^^

We label between 50-100 images that capture a robust span of the reaching volume. These labels are then
loaded into software intended to perform the iterative DLT calibration routine. This software, EasyWand5, can be
found at https://biomech.web.unc.edu/wand-calibration-tools/ . We curate points used in the calibration algorithm by hand (ie. manually selecting high outliers,
recalibrating, and readjusting our measurements) before using Sparse Bundle Adjustments to obtain a rotation and translational
co-efficient matrix.

3-D Reconstructed Reaching Volume
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: /art/EASYWAND_VOLUME.png
	:align: center
	:width: 600


DLT matrix coefficients for translation and rotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We obtain, from the iterative DLTdv process, a set of co-efficients that represent the rotation and translation of our
effective coordinate system. This .csv file is the main input into our ReachPredict3D software pipeline, the
other being unique 2-D individual camera DLC predictions.

Benchmarking our pose estimation in 3-D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our lab's initial pilot behavioral data, we have benchmarked a series of daily calibrations.

.. image:: /art/CAL_ACC.png
	:align: center
	:width: 400

We then compared calibrations across days, using a random daily calibration to determine changes in in the root-mean
square error.

.. image:: /art/ACROSS_CALS.png
	:align: center
	:width: 400


We have included our most accurate calibration file for general reconstruction. Time-specific reconstruction is not supported
at this time in the general software pipeline.


