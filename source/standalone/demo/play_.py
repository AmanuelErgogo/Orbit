# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import time

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="xarm7", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view


import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.objects.rigid import RigidObject

from omni.isaac.orbit.robots.config.xarm import XARM_ARM_WITH_XARM_GRIPPER_CFG
from omni.isaac.orbit.robots.config.xarm_spoon import XARM_ARM_WITH_SPOON_CFG
from omni.isaac.orbit_envs.manipulation.scoop import ScoopEnvCfg

from stable_baselines3 import PPO

"""
Main
"""


def main():
    """Spawns a single-arm manipulator and applies commands through inverse kinematics control."""

    # Load kit helper
    sim = SimulationContext(
        stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda:0"
    )
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Enable GPU pipeline and flatcache
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=8.0, num_per_row=8)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Markers
    ee_marker = StaticMarker("/Visuals/ee_current", count=args_cli.num_envs, scale=(0.03, 0.03, 0.03))
    goal_marker = StaticMarker("/Visuals/ee_goal", count=args_cli.num_envs, scale=(0.1, 0.1, 0.1))
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-0.03542)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 1000.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 1000.0, "color": (1.0, 1.0, 1.0)},
    )
    # -- Hospital 
    hospital_usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/scenes/geo_o_room3.usd"
    # -- Hospital with people
    # hospital_usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/scenes/geo_o_room3_people_sim.usd"

    prim_utils.create_prim("/World/envs/env_0/Hospital_room", usd_path=hospital_usd_path)

    # -- Food in bowl
    # macaroni_usd = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/macaroni_single_deformable.usd"
    # mac_prim = prim_utils.create_prim("/World/envs/env_0/Macaroni", usd_path=macaroni_usd, position=(1.55, -1.55, 0.93), scale=(0.01, 0.01, 0.01))

    # -- Food
    food_usd = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/cecina.usd"
    food_cecina_prim = prim_utils.create_prim("/World/envs/env_0/Cecina", usd_path=food_usd, position=(1.55, -1.55, 0.93), scale=(0.01, 0.01, 0.01))
    stage = stage_utils.get_current_stage()

    # -- Bowl
    bowl = RigidObject(cfg=ScoopEnvCfg().bowl)
    bowl.spawn("/World/envs/env_0/Bowl", translation=(1.55, -1.55, 0.8), orientation=(1, 0.0, 0.0, 0.0))

    # -- Macroni
    # time.sleep(3)
    # macroni = RigidObject(cfg=ScoopEnvCfg().macroni)
    # macroni.spawn("/World/envs/env_0/Macro", translation=(1.55, -1.55, 0.85))
    # -- Robot
    # resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    elif args_cli.robot == "xarm7":
        robot_cfg = XARM_ARM_WITH_XARM_GRIPPER_CFG
    elif args_cli.robot == "xarm7_spoon":
        robot_cfg = XARM_ARM_WITH_SPOON_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, xarm7, xarm7_spoon")
    # configure robot settings to use IK controller
    robot_cfg.data_info.enable_jacobian = True
    robot_cfg.rigid_props.disable_gravity = True
    # spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/env_0/Robot", translation=(1.69104, -0.77251, 0.931), orientation=(0.707, 0.0, 0.0, 0.707))

    # Clone the scene
    num_envs = args_cli.num_envs
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    envs_positions = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths)
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=sim.device)
    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )
    ee_goals = [
        [1.69104 + 0.5, -0.77251 + 0.5, 0.931 + 0.7, 0.707, 0, 0.707, 0],
        [1.69104 + 0.5, -0.77251 - 0.4, 0.931 + 0.6, 0.707, 0.707, 0.0, 0.0],
        [1.69104 + 0.5, -0.77251 + 0, 0.931 + 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goal_idx = 0

    # Run inference on the trained policy
    num_observations = 9
    model_reach = PPO.load("/home/aman/Orbit/models/ppo_reach.zip", print_system_info=False)
    obs = torch.zeros((num_envs, num_observations))

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/envs/env_.*/Robot")
    # Reset states
    robot.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    # Note: We need to update buffers before the first step for the controller.
    robot.update_buffers(sim_dt)
    # bowl.update_buffers(sim_dt)

    # Simulate physics
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            sim_time = 0.0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            bowl_new_state = bowl.data.root_state_w
            # bowl_new_state[:, 0:3] = ee_goals[ee_goal_idx]
            print(bowl_new_state)
            # bowl.set_root_state(bowl_new_state)
            
        # set the controller commands
        # in some cases the zero action correspond to offset in actuators
        # so we need to subtract these over here so that they can be added later on
        # offset actuator command with position offsets
        # note: valid only when doing position control of the robot
        # apply actions
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run IK example with Manipulator
    main()
    # Close the simulator
    simulation_app.close()
