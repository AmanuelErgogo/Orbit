import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects.rigid import RigidObject
from omni.isaac.orbit_envs.manipulation.scoop import ScoopEnvCfg
from gym import spaces
import numpy as np
import torch
import time


def sample_pre_scope_pose():

    bbox_low_lim = torch.tensor([-0.1, -0.1, 0.4, 0, 0, 0, 0.99])
    bbox_upp_lim = torch.tensor([0.1, 0.1, 0.3, 0.001, 0.01, 0.001, 1])

    # bbox = spaces.Box(low=low_lim, high=upp_lim)
    # print(bbox)

    # bbox = (bbox_low_lim - bbox_upp_lim * torch.rand((7)) + bbox_upp_lim).to(dtype=torch.float)
    bbox = bbox_low_lim + torch.rand((7)) * (bbox_upp_lim - bbox_low_lim)
    return bbox



"""
Main
"""


def main():
    """Spawns lights in the stage and sets the camera view."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
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

    soft_sphere_usd = "/home/aman/Desktop/deformable_sphere.usd"
    prim_utils.create_prim("/World/sphere_soft", usd_path=soft_sphere_usd, position=(1,1,5))

    bowl = RigidObject(cfg=ScoopEnvCfg().bowl)
    bowl.spawn("/World/envs/env_0/Bowl", translation=(0, 0, 0), orientation=(0, 1, 0.0, 0.0))

    bbox_marker = StaticMarker("/Visuals/pre_scoop_pose_marker", count=1, scale=(0.00008, 0.00008, 0.00008), usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/others/spoon_LowPoly.usd")

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        if count % 150 == 0:
            pre_scoop_pose = sample_pre_scope_pose()
            pre_scoop_pose = pre_scoop_pose.reshape((1, 7))
            print(pre_scoop_pose.shape)
            print(pre_scoop_pose[:, 3:7])
            bbox_marker.set_world_poses(positions=pre_scoop_pose[:, 0:3], orientations=pre_scoop_pose[:, 3:7])

        sim.step()
        count += 1


if __name__ == "__main__":
    # Run empty stage
    main()
    # Close the simulator
    simulation_app.close()