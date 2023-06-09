# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a oculus teleoperation with Orbit manipulation environments."""

"""Launch Isaac Sim Simulator first."""

"""source catkin workspace with teleop_isaac pkg before running eg. cd orbit/source/standalone/ros_ws/; source devel/setup.bash """
"""call device, """


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=10.0, help="Sensitivity factor.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch
import rospy

import carb

from omni.isaac.orbit.devices import Se3Keyboard, Se3SpaceMouse
from teleop_isaac.teleop_device import TeleopDeviceSubscriber

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg



def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running oculus teleoperation with Orbit manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task or "Scoop" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
        # add teleoperation key for env reset
        teleop_interface.add_callback("L", env.reset)
        # reset environment
        env.reset()
        teleop_interface.reset()
    elif args_cli.device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.07 * args_cli.sensitivity
        )
        # add teleoperation key for env reset
        teleop_interface.add_callback("L", env.reset)
        env.reset()
        teleop_interface.reset()
    elif args_cli.device.lower() == "oculus":
        teleop_interface = TeleopDeviceSubscriber(
        )
        # reset environment
        env.reset()

    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
    # print helper for keyboard
    print(teleop_interface)

    # teleop_device = TeleopDeviceSubscriber()
    # print("Teleop Device: ", teleop_device.device, "\n - Sensitivity: ", (0.1 * args_cli.sensitivity), "\n===========================")

    # simulate environment
    while simulation_app.is_running():
        # get keyboard command
        delta_pose, gripper_command = teleop_interface.advance()
        print(f'[INFO] delta pos rel: {delta_pose}')
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        # pre-process actions
        actions = pre_process_actions(delta_pose, gripper_command)
        # print(f'[INFO] action: {actions}')
        # apply actions
        _, _, _, _ = env.step(actions)
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
