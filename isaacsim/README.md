# Installation
This document goes over how to install IsaacSim 5.1.0 for Ubuntu 22.04 with compatability with ROS2 Humble

The important detail with this installation is that IsaacSim 5.1 requires Python 3.11, but Ubuntu 22.04, and therefore the binaries for ROS2 Humble, use Python 3.10. So sourcing a standard ROS2 Humble installation to run the ROS2 Bridge in IsaacSim will cause an error. There are a few ways to get around this:


1. (Untested) Set up and build a Docker container that has ROS2 Humble built for Python 3.11 and Isaacsim 5.1.0. Instructions can be found [here](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#running-ros-in-docker-containers)
  - Avoids messing with the default Python version provided with 22.04, but requires all simulation work be done in the container
  - Will have to look into compatibility with the container we are trying to build for the overall project

2. (Untested) Build ROS2 Humble from source in a virtual environment that utilizes Python 3.11. In theory this should force a build with the newer Python version and establish compatibility with Isaacsim. Install Isaacsim normally as shown [here](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
  - Only need to source one version of ROS2
  - Mixed signals on whether or not ROS2 fully functions in virtual environments

3. (Tested) [Install ROS2 Humble](https://docs.ros.org/en/humble/Installation.html#) and use the [ROS2 Workspaces Repository](https://github.com/isaac-sim/IsaacSim-ros_workspaces) to build a second ROS2 Humble that is compatible with IsaacSim (uses Python 3.11). more detailed instructions for this second step are found [here](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#enabling-rclpy-custom-ros-2-packages-and-workspaces-with-python-3-11)
  - Avoids the need to change Ubuntu's default Python version, which can have unintended consequences 
  - Does mean that you will need to be careful which ROS installation you source when running programs (when running Isaacsim, run the locally-built version)

4. (Not recommended) Change the default Python version for Ubuntu 22.04 and build ROS2 Humble from source
  - Enforces compatibility and allows running without virtual environments or docker containers
  - Risk of breaking some Ubuntu packages that relied on Python 3.10
