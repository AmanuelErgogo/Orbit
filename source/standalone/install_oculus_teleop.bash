#!/bin/bash

# Set the root directory where the Orbit is 
ORBIT_ROOT_DIR="/home/aman"
# Root directory where the catkin workspace will be created
ROOT_DIR="${ORBIT_ROOT_DIR}/Orbit/source/standalone"

# Set the name of the catkin workspace
WORKSPACE_NAME="teleop_ws"

# Create the catkin workspace directory and change into it
mkdir -p "${ROOT_DIR}/${WORKSPACE_NAME}/src"
cd "${ROOT_DIR}/${WORKSPACE_NAME}/src"
catkin init

# Clone the repositories into the 'src' folder
cd "${ROOT_DIR}/${WORKSPACE_NAME}/src"
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
git clone https://github.com/AmanuelErgogo/oculus-isaac-sim-robot-teleop.git
# Add more repositories as needed

# Change back to the root of the workspace and build it
cd "${ROOT_DIR}/${WORKSPACE_NAME}"
catkin_make
