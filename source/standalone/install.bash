#!/bin/bash

# Specify the name of the catkin workspace
workspace_name="teleop_ws"

# Specify the name of the source directory
source_dir="src"

# Specify the GitHub repository URLs to clone
repo_urls=(
  "https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git"
  "https://github.com/AmanuelErgogo/oculus-isaac-sim-robot-teleop.git"
)

# Create the catkin workspace
mkdir -p /$workspace_name/$source_dir
cd /$workspace_name/$source_dir
catkin_init_workspace

# Clone the repositories into the source directory
for repo_url in "${repo_urls[@]}"
do
  git clone $repo_url
done

# Build the catkin workspace
cd /$workspace_name
catkin_make

cd /$workspace_name
source devel/setup.bash
roslaunch ros_tcp_endpoint endpoint.launch
