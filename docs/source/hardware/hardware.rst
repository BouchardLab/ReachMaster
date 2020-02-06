Hardware
=======================================
Here you will find a detailed list of parts, build instructions, and discussion of alternative design choices.

Enclosure
---------
.. image:: /art/enclosure.png
	:align: center
	:width: 800

The enclosure houses the experiments and is made up of two compartments. The leftmost is a behavioral compartment which houses the animals. The rightmost is a hardware compartment which houses the robot and other peripheral devices (i.e., lights, solenoid, etc.). The CAD assembly for the enclosure can be on `our github <https://github.com/BouchardLab/ReachMaster/blob/master/hardware/designs/rat_enclosure.iam>`_ .

The entire assembly is constructed from extruded aluminum (i.e., 80/20) and fastened using mostly anchor fasteners. This offers a high degree of modularity and adjustability. For example, the relative positioning of the animal and robot can be easily adjusted, and there are many options for connecting or removing peripheral hardware. 80/20 is pretty straightforward to work with; however, it is highly recommended to go over the company's many tips and tutorials:

https://8020.net/product-basics

The behavioral compartment is relatively large at 24"x24"x38" (lwh). These dimensions were chosen with a range of `in vivo` neurophysiology experiments in mind. For example, the base dimensions are suitable for free-foraging hippocampal place cell or entorhinal grid cell experiments. The height was chosen so that a recording tether could reach all corners of the compartment without causing too much strain on the animal's head. 

Currently, the enclosure has two primary configurations: one for freely behaving animals, and a second for restrained animals. The configuration shown in the image above corresponds to the restrained condition. In that case, the body and/or head retraining device is merely placed inside the behavioral compartment, and the animal is given unimpeded access to the robot. In the free-moving condition, a plexiglass wall is inserted between the behavioral and hardware compartments. The animals is then allowed to freely explore the behavioral compartment and must reach through a slot in the plexiglass in order to interact with the robot.  

Robot
-----
.. image:: /art/robot.png
	:align: center
	:width: 800

The ReachMaster robot is a pneumatically-actuated, passively balanced, parallel robot with two rotational and one translational degrees of freedom (dof). The two rotational dof's are controlled by two low-friction double-acting cylinders each connected to the base in parallel by a 2-dof gimbal. These two actuators are joined in series, by spherical joints, to a third cylinder which controls the translational dof. The translational actuator is also connected to the base by a 2-dof gimabal, and to the robot's handle and reward delivery unit. The reward delivery unit consists of a solenoid-driven liquid delivery spout, an IR beam-based lick detector, and an option LED to provide visual cues. Lastly, fast high resolution position sensing is achieved by low-friction linear potentiometers attached to each of the cylinder rods. All data from the potentiometers, solenoid, LED, and IR sensors are recorded by a SpikeGadgets acquisition system (see below). 

The robot workspace (shown in red) can be empirically estimated by acquiring potentiometer data from the robot as it explores its full range of motion, passing the trajectory through an analytically-derived forward kinematics transformation, and then fitting a surface to the extrema of the resulting scatter plot (see link_to_code). Similary, command positions can be derived by sampling points from some relevant subspace of the robot workspace (e.g., rodent workspace shown in blue), and then passing those points through an analytically-derived inverse kinematics transformation that returns the corresponding potentiometer values (see :meth:`reachmaster.interfaces.robot_interface.load_config_commands`).  

Kinematics
^^^^^^^^^^
<figure>
Make new figure with all relevant kinematic variables labeled. Provide mathematical derivation of forward and inverse kinematics. Link to relevant sections of user interface and data preprocessing codes.

Air Delivery
^^^^^^^^^^^^^^
.. image:: /art/compressor_plus_valves.png
	:align: center
	:width: 400

Actuation
^^^^^^^^^^^^^^^^^^^
.. image:: /art/pneumatic_cylinder.jpeg
	:align: center
	:width: 400

Position Sensing
^^^^^^^^^^^^^^^^
.. image:: /art/potentiometer.jpg
	:align: center
	:width: 400

Gimbal Parts
^^^^^^^^^^^^^^^^
picture

Handles
^^^^^^^
pictures

Mounting
^^^^^^^^
picture

Cameras
-------
.. image:: /art/cameras.png
	:align: center
	:width: 400

Lighting
--------
.. image:: /art/neopixels.png
	:align: center
	:width: 400

Reward Delivery
---------------
.. image:: /art/solenoid.png
	:align: center
	:width: 250

Lick Detection
--------------
picture

Data Acquisition
----------------

Computers
---------
picture

Build Instructions
------------------
coming soon






