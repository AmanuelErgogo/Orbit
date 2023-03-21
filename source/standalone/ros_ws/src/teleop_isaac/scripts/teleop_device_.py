#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
# from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from source.standalone.ros_ws.src.teleop_isaac.scripts.teleop_demo import set_par

# global delta_pose
# delta_pose = [0, 0, 0, 0, 0, 0, 0]

rospy.init_node("teleop_device_subscriber", anonymous=True)


def isaac_cmd_vel_clb(msg):
    # global delta_pose
    # convert angular rot to quat
    quat = quaternion_from_euler(msg.angular.x, msg.angular.y, msg.angular.z)
    delta_pose = [msg.linear.x, msg.linear.y, msg.linear.z, quat[0], quat[1], quat[2], quat[3]]
    set_par(delta_pose_=delta_pose)
    rospy.loginfo(delta_pose)


# def read_delta_pose():
#     global delta_pose
#     return delta_pose


def main():
    global delta_pose
    rospy.Subscriber("/isaac/cmd_vel", Twist, isaac_cmd_vel_clb)
    print("del pos: ", delta_pose)
    # rospy.Subscriber("/vr/joy_command", Joy, joy_cmd_clb)
    # rospy.Subscriber("/vr/right_controller_pose", PosRot, right_controler_clb)
    # rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, compressed_img_clb)

    rospy.spin()


if __name__ == '__main__':
    main()
