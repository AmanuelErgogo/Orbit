#!/bin/bash

mkdir -p teleop_ws/src
cd teleop_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
git clone https://github.com/AmanuelErgogo/oculus-isaac-sim-robot-teleop.git
cd ..
catkin_make
source devel/setup.bash
roslaunch ros_tcp_endpoint endpoint.launch

