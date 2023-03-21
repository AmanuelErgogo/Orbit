#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Pose
from unity_robotics_demo_msgs.msg import PosRot
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler
import time


prev_pose = None
global grip_pos_old
global follow_active
follow_active = False
grip_pos_old = 0.0

global planner_hz, cnt
planner_hz, cnt = 10, 0


def denormalize_zero_to_one_scale(x_normalized, max, min):
    return x_normalized * (max - min) + min


def unity_to_ros_pose(pose):
    """Create a new Pose object in ROS coordinates from unity coordinate"""
    ros_pose = Pose()
    # Convert the position coordinates from Unity to ROS coordinates
    ros_pose.position.x = -pose.pos_z
    ros_pose.position.y = -pose.pos_x
    ros_pose.position.z = pose.pos_y
    # Convert the orientation quaternion from Unity to ROS coordinates
    ros_pose.orientation.x = -pose.rot_z
    ros_pose.orientation.y = -pose.rot_x
    ros_pose.orientation.z = pose.rot_y
    ros_pose.orientation.w = pose.rot_w
    return ros_pose


def change_in_pose(current_pose: Pose):
    """Calculate the change in Pose"""
    global prev_pose
    d_pose = Pose()

    d_pose.position.x = current_pose.position.x - prev_pose.position.x
    d_pose.position.y = current_pose.position.y - prev_pose.position.y
    d_pose.position.z = current_pose.position.z - prev_pose.position.z
    d_pose.orientation.x = current_pose.orientation.x - prev_pose.orientation.x
    d_pose.orientation.y = current_pose.orientation.y - prev_pose.orientation.y
    d_pose.orientation.z = current_pose.orientation.z - prev_pose.orientation.z
    d_pose.orientation.w = current_pose.orientation.w - prev_pose.orientation.w
    
    prev_pose = current_pose

    return d_pose


def scale_down_pose(cont_pose: Pose, scaling_factor=0.7):
    """Scale the controller pose by a scaling factor Args: scaling_factor [0.3 0.7]"""
    down_scaled_pose = Pose()
    down_scaled_pose.position.x = cont_pose.position.x * scaling_factor
    down_scaled_pose.position.y = cont_pose.position.y * scaling_factor
    down_scaled_pose.position.z = cont_pose.position.z * scaling_factor
    # down_scaled_pose.orientation.x = cont_pose.orientation.x * 0.5
    # down_scaled_pose.orientation.y = cont_pose.orientation.y * 0.5
    # down_scaled_pose.orientation.z = cont_pose.orientation.z * 0.5
    # down_scaled_pose.orientation.w = cont_pose.orientation.w * 0.5

    down_scaled_pose.orientation.x = 0.9956441983869033
    down_scaled_pose.orientation.y = 0.08808622778073098
    down_scaled_pose.orientation.z = 0.030352420833315182
    down_scaled_pose.orientation.w = 0.003489590723817486
    print("scaled down pose goal: ", scale_down_pose)
    return down_scaled_pose


def joy_cmd_clb(msg):
    global grip_pos_old, follow_active
    if msg.buttons[0] == 1:
        # gripper_move(10)
        # plan_joint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # print("moved to home position")
        print("x pressed")
    if msg.buttons[1] == 1:
        # gripper_move(500)
        print("y pressed")
    # if abs(msg.axes[2] - grip_pos_old) > 0.08:
    #     grip(msg.axes[2])
    #     grip_pos_old = msg.axes[2]
    if msg.axes[3] > 0.8:
        follow_active = True
    else:
        follow_active = False


class TeleopDeviceSubscriber(object):
    """device: str["mobile_app", "oculus"] default "mobile_app" """
    def __init__(self, device="oculus"):
        self.device = device
        if self.device == "mobile_app":
            rospy.Subscriber("/isaac/cmd_vel/xy", Twist, self.isaac_cmd_vel_xy_clb)
            rospy.Subscriber("/isaac/cmd_vel/z", Twist, self.isaac_cmd_vel_z_rot_z_clb)
            rospy.Subscriber("/isaac/cmd_vel/rot_xy", Twist, self.isaac_cmd_vel_rot_xy_clb)
            self.delta_pose = [0, 0, 0, 0, 0, 0, 0]
        elif self.device == "oculus":
            rospy.Subscriber("/vr/right_controller_pose", PosRot, self.oculus_delta_pose_cb)
            rospy.Subscriber("/vr/joy_command", Joy, self.joy_cmd_clb)
            self.delta_pose = [0, 0, 0, 0, 0, 0, 0]
            self.follow_cmd = False
        else:
            raise ValueError(f"Device {self.device} is not supported. Valid: oculus, mobile_app")
        self.data_stream_active = False
        # rate = rospy.Rate(5)
        # while not (self.data_stream_active):
        #     rospy.loginfo("starting up Teleop Device Stream")
        #     rate.sleep()

    def isaac_cmd_vel_xy_clb(self, msg):
        quat = quaternion_from_euler(msg.angular.x, msg.angular.y, msg.angular.z)
        self.delta_pose = [msg.linear.x, msg.linear.y, msg.linear.z, quat[0], quat[1], quat[2], quat[3]]

    def isaac_cmd_vel_z_rot_z_clb(self, msg):
        quat = quaternion_from_euler(msg.angular.x, msg.angular.y, msg.angular.z)
        self.delta_pose = [msg.linear.x, msg.linear.y, msg.linear.z, quat[0], quat[1], quat[2], quat[3]]

    def isaac_cmd_vel_rot_xy_clb(self, msg):
        quat = quaternion_from_euler(msg.angular.x, msg.angular.y, msg.angular.z)
        self.delta_pose = [msg.linear.x, msg.linear.y, msg.linear.z, quat[0], quat[1], quat[2], quat[3]]

    def oculus_delta_pose_cb(self, msg):
        global planner_hz, cnt, prev_pose, follow_active
        pose = unity_to_ros_pose(msg)
        # Reset prev pose
        if self.follow_cmd is False:
            prev_pose = pose
        if cnt % planner_hz == 0 and self.follow_cmd is True:
            if prev_pose is None:
                prev_pose = pose
            delta_pose = change_in_pose(pose)
            # rospy.sleep(3)
            # update self.delta_pose
            self.delta_pose = [delta_pose.position.x, delta_pose.position.y, delta_pose.position.z, delta_pose.orientation.w, delta_pose.orientation.x, delta_pose.orientation.y, delta_pose.orientation.z]
            # print("delta_pose set!")
        cnt = cnt + 1

    def joy_cmd_clb(self, msg):
        if msg.axes[3] > 0.8:
            self.follow_cmd = True
        else:
            self.follow_cmd = False

    def get_delta_pose(self):
        # print("connected to: ", self.device)
        return self.delta_pose

    def is_follow(self):
        return self.follow_cmd


if __name__ == '__main__':
    rospy.init_node('teleop_device_subscriber', anonymous=True)
    teleop_device = TeleopDeviceSubscriber()
    print("--delta pose--", teleop_device.get_delta_pose())
    time.sleep(3)
    print("--delta pose--", teleop_device.get_delta_pose())
