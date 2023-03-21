#!/usr/bin/env python

import rospy
from teleop_isaac.teleop_device import TeleopDeviceSubscriber
import time


if __name__ == '__main__':
    rospy.init_node('sample_node_', anonymous=True)
    print("up!")
    teleop_device = TeleopDeviceSubscriber()
    while True:
        print("Delta Pose: ", teleop_device.get_delta_pose())
        time.sleep(3)
        print("Delta Pose: ", teleop_device.get_delta_pose())