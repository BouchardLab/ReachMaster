ReachSplitter: Time-Series Segmentation for 3-D Reaching Data
=====================================================================
Our lab records sensor data continuously across a given experimental session. As we allow our rats to freely behave
inside their behavioral arena, our data requires both coarse behavioral segmentation to separate reaching behavior
from other behaviors such as grooming or locomotion. Additionally, we require a more fine-grained segmentation within
a given coarsely-defined reaching period. This fine-grained segmentation of individual reaches uses a combination
of supervised and unsupervised learning to determine the type of behavior at a temporal resolution of ms. Our behavioral
classification pipeline is fully extendable to realtime positional predictions from DLC-LIVE as well. This gives future
experimenters flexibility in designing experiments with controlled feedback based on predicted behaviors while capturing
continuous, naturalistic reaching behavior.

The Problem
^^^^^^^^^^^
After an experiment is initiated, a rat is free to interact with the water-delivering handle. However, we allow the rat
to also behave freely in our arena.  This means a rat may interact with the handle quite frequently (defined as a bout
of reaching behavior, 5+ reaches over a minute) or rarely. As the rat advances in training, interactions become more
frequent and with longer interactions to the handle. In addition to temporal variance, reaching behavior itself is highly
varying between different phenotypes. Examples include handedness of a reach, the success or fail of a reach, and other
categorical differences in observed reaching behaviors of rats. For a given "detected" coarse
reaching behavior, many individual reaches with their appropriate class labels may also be inside.
Teasing out the relative importance of these variables to our neural data is of high importance to the
Bouchard Lab.
The resulting variance in our general observed behavior necessitates a method of segmenting our individual reaching
trial data into discrete time-series representations of single reaches with appropriate class labels,
mainly to avoid analyzing our neural data out of context.

Our Solution
^^^^^^^^^^^^
Our lab previously has shown that we can extract 3-D coordinates with highly accurate spatial and temporal resolution
across both arms during reaching behaviors. Briefly, continuous video from 3 cameras inside our arena is tracked using
DeepLabCut. The stored predictions are then translated from 2-D image space to a shared 3-D space. Time-series of various
sensor and positional data may then be analyzed in the context of collected neural data.

In this scenario, correctly assigning the behavioral phenotype to a given range of time-series values is a critical
operation. Indeed, within a given behavioral phenotype there may be large biases or errors which may confound any
analysis that seeks to encode or decode brain activity w.r.t. body position. Additionally, neural signals may be
behavioral-dependent as well as positional-dependent, creating a necessity inside our paradigm to segment and classify
our continuous time-series into well-defined behavioral categories.

Our lab has created a supervised algorithm to sub-classify a coarse reach into fine, individually classified reaching
behaviors.

