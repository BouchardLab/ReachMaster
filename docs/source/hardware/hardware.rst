Hardware
========
Here you will find a detailed list of parts, build instructions, and discussion of alternative design choices.

Enclosure
---------
.. image:: /art/enclosure.png
	:align: center
	:width: 800

The enclosure houses the experiments and is made up of two compartments. The leftmost is a behavioral compartment which houses the animals. The rightmost is a hardware compartment which houses the robot and other peripheral devices (i.e., lights, solenoid, etc.). The CAD assembly for the enclosure is on `our github <https://github.com/BouchardLab/ReachMaster/blob/master/hardware/designs/rat_enclosure.iam>`_ .

The entire assembly is constructed from extruded aluminum (i.e., 80/20) and fastened using mostly anchor fasteners. This offers a high degree of modularity and adjustability. For example, the relative positioning of the animal and robot can be easily adjusted, and there are many options for connecting or removing peripheral hardware. 80/20 is pretty straightforward to work with; however, it is highly recommended to go over the company's many tips and tutorials:

https://8020.net/product-basics

The behavioral compartment is relatively large at 26"x26"x52" (lwh). These dimensions were chosen with a range of `in vivo` neurophysiology experiments in mind. For example, the base dimensions are suitable for free-foraging hippocampal place cell or entorhinal grid cell experiments. The height was chosen so that a recording tether could reach all corners of the compartment without causing too much strain on the animal's head. 

Currently, the enclosure has two primary configurations: one for freely behaving animals, and a second for restrained animals. The configuration shown in the image above corresponds to the restrained condition. In that case, the body and/or head retraining device is merely placed inside the behavioral compartment, and the animal is given unimpeded access to the robot. In the free-moving condition, a plexiglass wall is inserted between the behavioral and hardware compartments. The animals is then allowed to freely explore the behavioral compartment and must reach through a slot in the plexiglass in order to interact with the robot.  

Robot
-----
.. image:: /art/robot.png
	:align: center
	:width: 800

The ReachMaster robot is a pneumatically-actuated, passively balanced, parallel robot with two rotational and one translational degrees of freedom (dof). The two rotational dof's are controlled by two low-friction double-acting cylinders (:ref:`Actuation`) each connected to the base (:ref:`Mounting`) in parallel by a 2-dof gimbal (:ref:`Gimbal`). These two actuators are joined in series, by spherical joints, to a third cylinder which controls the translational dof. The translational actuator is also connected to the base by a 2-dof gimabal, and to the robot's handle (:ref:`Handles`) and reward delivery unit (:ref:`Reward Delivery`). The reward delivery unit consists of a solenoid-driven liquid delivery spout, an IR beam-based lick detector (:ref:`Lick Detection`), and an option LED to provide visual cues. Lastly, fast high resolution position sensing is achieved by low-friction linear potentiometers (:ref:`Position Sensing`) attached to each of the cylinder rods. All data from the potentiometers, solenoid, LED, and IR sensors is recorded by a SpikeGadgets acquisition system (:ref:`Data Acquisition`). 

The robot workspace (shown in red) can be empirically estimated by acquiring potentiometer data from the robot as it explores its full range of motion, passing the trajectory through an analytically-derived forward kinematics transformation (:ref:`Kinematics`), and then fitting a surface to the extrema of the resulting scatter plot (see link_to_code). Similary, command positions can be derived by sampling points from a relevant subspace of the robot workspace (e.g., the rodent workspace shown in blue), and then passing those points through an analytically-derived inverse kinematics transformation that returns the corresponding potentiometer values.  

Kinematics
^^^^^^^^^^
<figure>

Make new figure with all relevant kinematic variables labeled. Provide mathematical derivation of forward and inverse kinematics. Link to relevant sections of user interface and data preprocessing codes.

<link_to_forward_kinematics_code>

Link to inverse kinematics code:

:func:`reachmaster.interfaces.robot_interface.load_config_commands`

Air Delivery
^^^^^^^^^^^^
.. image:: /art/compressor_plus_valves.png
	:align: center
	:width: 400

Add discussion of compressor(s), manifold(s), valve(s), vents, circuit(s), tubing, etc. 

Actuation
^^^^^^^^^
.. image:: /art/pneumatic_cylinder.jpeg
	:align: center
	:width: 400

Add discussion of our current SMC cylinders, and Airpel as an alternative.

Position Sensing
^^^^^^^^^^^^^^^^
.. image:: /art/potentiometer.jpg
	:align: center
	:width: 400

Add discussion of linear potentiometers and contactless alternatives.

Gimbal 
^^^^^^
picture

Describe each of the 3D printed gimbal pieces.

Handles
^^^^^^^
Discuss various handles types, threading and weight requirements.

Mounting
^^^^^^^^
Discuss mounting issues with the base plate and other Thor Labs components.

Pressure Sensing
^^^^^^^^^^^^^^^^
Discuss pressure sensors. Link to Todorov PID paper

Cameras
-------
.. image:: /art/cameras.png
	:align: center
	:width: 400

Discuss current Ximea XiQ USB3.0 cameras highspeed color cameras, and IR and newer PCIe altenernatives. Lenses and resolution tradeoffs, trigger and synchronization options. Point to SpikeGadgets slack channel for discussion of additional options? Trodes camera module? USB card requirements/limitations.

Lighting
--------
.. image:: /art/neopixels.png
	:align: center
	:width: 400

Discuss neopixels and what you can do with them. Discuss electrical noise issues, IR lighting alternatives, more diffuse lighting alternatives. Point to Whishaw rat pellet reaching chapter for product.

Reward Delivery
---------------
.. image:: /art/solenoid.png
	:align: center
	:width: 250

Discuss the Lee solenoids and proper maintainance, other alternatives. Calibration. Link to code and website tutorial.

Lick Detection
--------------
<picture>

Discuss the IR emitters/receivers, the driver circuit, and other variants. Link to website tutorial(s).

Controllers
-----------
<picture>

Discuss Arduino, Teensy, mbed, general requirements, and alternatives.

Data Acquisition
----------------
<picture>

Discuss SpikeGadgets and link to thier site. Discuss National Instruments options?

Computers
---------
<picture>

Discuss CPU, motherboard, GPU and USB/PCIe requirements. 

Build Instructions
------------------
coming soon






