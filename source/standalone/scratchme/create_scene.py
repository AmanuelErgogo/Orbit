# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the physics engine to simulate a single-arm manipulator.

We currently support the following robots:

* Franka Emika Panda
* Universal Robot UR10

From the default configuration file for these robots, zero actions imply a default pose.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp
from dataclasses import MISSING

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit.robots.config.xarm import XARM_ARM_WITH_XARM_GRIPPER_CFG
from omni.isaac.orbit.robots.config.xarm_spoon import XARM_ARM_WITH_SPOON_CFG
from omni.isaac.orbit.robots.config.ridgeback_xarm7 import RIDGEBACK_XARM_WITH_GRIPPER_CFG

from omni.isaac.core.prims import GeometryPrim, RigidPrim, RigidPrimView, GeometryPrimView, XFormPrimView

from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysxSchema

from omni.isaac.orbit.objects.deformable import DeformableObject
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit_envs.manipulation.scoop import ScoopEnvCfg
# from omni.isaac.orbit_envs.manipulation.lift import LiftEnvCfg
"""
Main
"""


'''
class DeformableObject():
    """deformable object class
    Ref: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_deformable_body_a_p_i.html#a8f60f84ad6738d7e37f798f35d6e060e
    """
    def __init__(self, deform_path: str) -> None:
        self.prim_path = deform_path

    def get_deformable_body_from_stage(self):
        """ Returns PhysxDeformableBodyAPI to access local properties of deformable body """
        prim = get_current_stage().GetPrimAtPath(self.prim_path)
        return PhysxSchema.PhysxDeformableBodyAPI(prim)

    def get_object_view(self):
        """ Return XformView to access global properties of deformable object"""
        return XFormPrimView(prim_paths_expr=self.prim_path, name="DeformableSphere")
    
    def get_object_pose_w(self, deformable_object):
        return deformable_object.get_world_poses()
    
    def get_collision_indeces(self, deformable_body):
        return deformable_body.GetCollisionIndicesAttr().Get()
    
    def get_simulation_points(self, deformable_body):
        return deformable_body.GetSimulationPointsAttr().Get()
    
    """ Ref --- how to get global prop of soft body
    

    deform_path = “/World/Cube”
    prim = get_current_stage().GetPrimAtPath(deform_path)
    deformable_local = PhysxSchema.PhysxDeformableBodyAPI(prim)
    deformable_global = XFormPrimView(prim_paths_expr=deform_path, name=“Deformable_Cube”)

    local_collision_point = np.array(deformable_local.GetCollisionPointsAttr().Get())
    for i in range(3):
    local_collision_point[:,i] = local_collision_point[:,i]

    global_collision_point = deformable_global.get_world_poses()[0]
    final = (local_collision_point + global_collision_point) # N X 3 Numpy Array

    """

    '''


def main():
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Enable gpu dynamics of physics scene
    phyx_context = sim.get_physics_context()
    phyx_context.enable_gpu_dynamics(True)
    phyx_context.enable_ccd(True)
    phyx_context.enable_stablization(True)
    phyx_context.set_broadphase_type("GPU")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-0.03542)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    cfg_: ScoopEnvCfg = None
    fruit_berry = RigidObject(cfg=cfg_.berry)
    # obj = RigidObject(cfg=LiftEnvCfg.object)


    # particle usd
    # particle_usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/extscache/omni.warp-0.6.1+cp37/data/scenes/example_particles.usd"
    # prim_utils.create_prim("/World/envs/env_0/Particle", usd_path=particle_usd_path)
    # Table
    # table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    # prim_utils.create_prim("/World/Table_1", usd_path=table_usd_path, translation=(0.55, -1.0, 0.0))
    # prim_utils.create_prim("/World/Table_2", usd_path=table_usd_path, translation=(0.55, 1.0, 0.0))

    # -- Hospital with people
    hospital_usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/scenes/geo_o_room3.usd"
    # hospital_usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/rigid_defor_cubes.usd"
    prim_utils.create_prim("/World/envs/env_0/Hospital_room", usd_path=hospital_usd_path)
    # deformable macroni
    # macaroni_usd = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/macaroni_single_deformable.usd"
    # prim_utils.create_prim("/World/envs/env_0/Macaroni", usd_path=macaroni_usd, position=(1.55, -1.55, 0.93), scale=(0.01, 0.01, 0.01))
    # mac_prim = GeometryPrim("/World/envs/env_0/Macaroni/macaroni/macaroni_HP/macaroni_HP")

    # sim.physics_sim_view.create_soft_body_view()

    # Get deformable object from mesh
    deform_path = "/World/envs/env_0/Hospital_room/soft_sphere/Sphere_01"
    deform_food_path = "/World/envs/env_0/Hospital_room/cecina/Cecina/Cecina"
    # prim = get_current_stage().GetPrimAtPath(deform_path)
    # deformable_body = PhysxSchema.PhysxDeformableBodyAPI(prim)
    # deformable_global = XFormPrimView(prim_paths_expr=deform_path, name="DeformableSphere")
    # CollisionIndices = deformable_body.GetCollisionIndicesAttr().Get()
    # simulation_points = deformable_body.GetSimulationPointsAttr().Get()
    # print("========\n Sim points \n ======", simulation_points)

    # get deformables from stage
    soft_sphere = DeformableObject(deform_path=deform_food_path)
    soft_sphere_body = soft_sphere.get_deformable_body_from_stage()
    soft_sphere_object = soft_sphere.get_object_view()

    # Robots
    # -- Spawn robot
    xarm_robot = SingleArmManipulator(cfg=XARM_ARM_WITH_SPOON_CFG)
    xarm_mm = MobileManipulator(cfg=RIDGEBACK_XARM_WITH_GRIPPER_CFG)
    xarm_robot.spawn("/World/Xarm_robot", translation=(1.69104, -0.77251, 0.931), orientation=(0.707, 0.0, 0.0, 0.707))
    xarm_mm.spawn("/World/Xarm_mobile", translation=(-1.0, 0.0, 0.0))

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    xarm_robot.initialize("/World/Xarm_robot")
    xarm_mm.initialize("/World/Xarm_mobile")
    # Reset states
    xarm_robot.reset_buffers()
    xarm_mm.reset_buffers()
    # fruit_berry.reset_buffers()
    # obj.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(xarm_robot.count, xarm_robot.num_actions, device=xarm_robot.device)
    has_gripper = xarm_robot.cfg.meta_info.tool_num_dof is not MISSING

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
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
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = xarm_robot.get_default_dof_state()
            xarm_robot.set_dof_state(dof_pos, dof_vel)
            xarm_robot.reset_buffers()
            # reset command
            actions = torch.rand(xarm_robot.count, xarm_robot.num_actions, device=xarm_robot.device)
            # reset gripper
            if has_gripper:
                actions[:, -1] = -1
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 200 and has_gripper:
            # flip command for the gripper
            actions[:, -1] = -actions[:, -1]
        # apply action to the robot
        xarm_robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        print("global_def_pos ", soft_sphere.get_object_pose_w(soft_sphere_object))
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            xarm_robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()