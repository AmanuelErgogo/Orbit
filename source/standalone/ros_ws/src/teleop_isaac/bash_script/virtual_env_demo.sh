#!/bin/bash

# Setup and Run ROS Master
export ROS_MASTER_URI=http://192.168.0.134:11311
export ROS_IP=192.168.0.134
roscore 

# Launch unity ros connector
cd /home/aman/Orbit/source/standalone/ros_ws
source devel/setup.bash
roslaunch ros_tcp_endpoint endpoint.launch 

# Run teleop demo
cd /home/aman/Orbit/source/standalone/ros_ws
source devel/setup.bash
cd /home/aman/Orbit
./orbit.sh -p source/standalone/ros_ws/src/teleop_isaac/scripts/teleop_demo.py 

# Follow th instruction and teleop it!