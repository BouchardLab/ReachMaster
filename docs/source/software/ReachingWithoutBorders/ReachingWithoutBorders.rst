Reaching Without Borders
===========================
In this part of the documentation, we outline functions and methods to transform
our reaching and electrophysiological data from individualized files into the compact, transparent
format of Neurodata Without Borders (NWB). For more documentation on NWB, please visit https://www.nwb.org.

RWB class manager
-------------------
Loading our data into RWB utilizes a class manager to handle data instances across sessions.

Initiate Manager
^^^^^^^^^^^^^^^^^^^^
.. automodule:: __init__
    :members:

Run Function
^^^^^^^^^^^^^^^
.. automodule:: fetch_rwb
    :members:

Functionalized RWB integrators
------------------------------------
For a given session, we have several experimental datatypes across multiple levels of preprocessing we wish to extract.
Each datatype has a well-defined path within for a given session, which is used to load and save the data.

Initialization an of NWB file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: software.preprocessing.reaching_without_borders.rwb_utils.init_nwb_file
    :members:

