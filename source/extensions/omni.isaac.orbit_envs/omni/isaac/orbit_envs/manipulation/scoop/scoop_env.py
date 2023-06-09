# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List
import time
import random

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .scoop_cfg import ScoopEnvCfg, RandomizationCfg


class ScoopEnv(IsaacEnv):
    """Environment for scooping a food with a spoon mounted on single-arm manipulator."""

    def __init__(self, cfg: ScoopEnvCfg = None, headless: bool = False):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)

        self.object = RigidObject(cfg=self.cfg.object)    # Dex cube
        self.out_of_scene_pos = (0.6, 0.3, -5)

        # Sliced fruits
        self.berry = RigidObject(cfg=self.cfg.berry)    
        self.apple = RigidObject(cfg=self.cfg.apple)
        self.banana = RigidObject(cfg=self.cfg.banana)
        self.mango = RigidObject(cfg=self.cfg.mango)
        self.melon = RigidObject(cfg=self.cfg.melon)
        self.rs_berry = RigidObject(cfg=self.cfg.rs_berry)

        # self.sliced_fruits = {'berry': {'object': self.berry, 'num_slices': 1}, 'banana': {'object': self.banana, 'num_slices': 1}, 
        #                       'mango': {'object': self.mango, 'num_slices': 1}, 'rs_berry': {'object': self.rs_berry, 'num_slices': 1}}   # add 'melon': self.melon, 'apple': self.apple
        self.sliced_fruits = {'banana': {'object': self.banana, 'num_slices': 100}, 
                              'mango': {'object': self.mango, 'num_slices': 100}, 'rs_berry': {'object': self.rs_berry, 'num_slices': 100}}   # add 'melon': self.melon, 'apple': self.apple


        self.env_variations = {'table': self.cfg.table.usd_path, 'hospital': self.cfg.hospital.usd_path, 'hospital_reduced': self.cfg.hospital_reduced.usd_path}
        # Food
        self.macron = RigidObject(cfg=self.cfg.macron)

        # Feed Care Objects
        self.bowl = RigidObject(cfg=self.cfg.bowl)
        self.tray = RigidObject(cfg=self.cfg.tray)
        self.plate = RigidObject(cfg=self.cfg.plate)

        self.feed_care_objects = {'bowl': self.bowl, 'tray': self.tray, 'plate': self.plate}
        self.serving_way = {'in_bowl': [self.feed_care_objects['bowl']],
                            'on_plate': [self.feed_care_objects['plate']],
                            'bowl_on_plate': [self.feed_care_objects['plate'], self.feed_care_objects['bowl']],
                            'bowl_on_tray': [self.feed_care_objects['tray'], self.feed_care_objects['bowl']],
                            'plate_on_tray': [self.feed_care_objects['tray'], self.feed_care_objects['plate']],
                            'bowl_on_plate_on_tray': [self.feed_care_objects['tray'], self.feed_care_objects['plate'], self.feed_care_objects['bowl']]}
        
        # initialize the base class to setup the scene.
        super().__init__(self.cfg, headless=headless)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()
        # collect hide all the scene object 
        self._stock_scene_objects(default_scene=True)
        self.rand_object_key = self.cfg.randomization.way_of_serving.feed_object_default  # random.choice(list(self.feed_care_objects))
        self.rand_fruit_key =  self.cfg.randomization.food_type.food_default  # random.choice(list(self.sliced_fruits))

        # Define work space
        self.work_space = {'x':{'min': 1.521, 'max': 1.856},'y':{'min': -0.445, 'max': -0.238}, 'z':{'min': 0.9, 'max': 1}}

        # prepare the observation manager
        self._observation_manager = ScoopObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = ScoopRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        # print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions 
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        #print ("[INFO] Shape of Observation space:", (num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")
        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.object.update_buffers(self.dt)

        self.robot.update_buffers(self.dt)

        self.bowl.update_buffers(self.dt)
        self.tray.update_buffers(self.dt)

        for fruit in self.sliced_fruits:
            self.sliced_fruits[fruit]['object'].update_buffers(self.dt)

        # diable physics for care objects and fruits
        # self.banana.objects.enable_rigid_body_physics()
        # print("Banana Pos: ", self.banana.data.root_pos_w)
        # print("Berry Pos: ", self.berry.data.root_pos_w)
        # print("bowl Pos: ", self.bowl.data.root_pos_w)

        # print("[INFO] Observation: ", self._observation_manager.compute())
        
        # print("Default dof state: ", self.robot.get_default_dof_state()[0][0])
        # print("Dtype: ", type(self.robot.get_default_dof_state()))

        self.episode_count = 0

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # clone berry
        # from omni.isaac.cloner import GridCloner
        # berry_cloner = GridCloner(spacing=1e-4)
        # berry_cloner.define_base_env("/World/envs/env_0")
        # prim_utils.define_prim("/World/envs/env_0/Berry/Collisions")

        env_key = 'hospital_reduced' # random.choice(list(self.env_variations))
        if env_key == 'hospital' or env_key == 'hospital_reduced':
            self.robot_world_pose = self.robot_world_pose = {'translation':(1.69104, -0.77251, 0.931), 'orientation':(0.707, 0.0, 0.0, 0.707)} 
            self.dishes_world_pos = (1.69, -0.4, 0.931)
            self.ground_plane_z_pos = -0.03542
        else:
            self.robot_world_pose = {'translation':(0.0, 0.0, 0.0), 'orientation':(1, 0.0, 0.0, 0.0)}
            self.dishes_world_pos = (0.6, 0.3, 0.007)
            self.ground_plane_z_pos = -1.05
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position = self.ground_plane_z_pos)
        # init environment
        prim_utils.create_prim(self.template_env_ns + "/" + env_key.capitalize(), usd_path=self.env_variations[env_key])
        # bowl
        # prim_utils.create_prim(self.template_env_ns + "/Book", usd_path=self.cfg.book.usd_path)
        # prim_utils.create_prim(self.template_env_ns + "/SpecialDish", usd_path=self.cfg.special_dish.usd_path, translation=(0.4, 0.0, 0))
        # sliced apple dish
        # prim_utils.create_prim(self.template_env_ns + "/SlicedDish", usd_path=self.cfg.dish_sliced_apple.usd_path, position=(0.4, 0.1, 0.0))
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot", translation=self.robot_world_pose['translation'], orientation=self.robot_world_pose['orientation'])
        # # object
        self.object.spawn(self.template_env_ns + "/Object")
        # # tray
        # self.tray.spawn(self.template_env_ns + "/Tray", translation=(0.6, 0.3, 0.005))
        # # plate
        # self.plate.spawn(self.template_env_ns + "/Plate", translation=(0.6, -0.3, 0.006))
        # # berry
        # self.bowl.spawn(self.template_env_ns + "/Bowl", translation=self.dishes_world_pos)
        for care_object in self.feed_care_objects:
            self.feed_care_objects[care_object].spawn(self.template_env_ns + "/" + care_object.capitalize(), translation=self.out_of_scene_pos)
        for fruit in self.sliced_fruits:
            self.sliced_fruits[fruit]['object'].spawn(self.template_env_ns + "/" + fruit.capitalize() + "_0", translation=self.out_of_scene_pos)
        
            # Clone within base env before cloning to vectorized env
            # self._clone_scene(target_path="/World/envs/env_0/Rs_berry", source_path="/World/envs/env_0/Rs_berry_0", num_clones=10)
            self._clone_scene(target_path=self.template_env_ns + "/" + fruit.capitalize() , source_path=self.template_env_ns + "/" + fruit.capitalize() + "_0", num_clones=self.sliced_fruits[fruit]['num_slices'], spacing=0.00001)

        # Clone berry
        # envs_prim_paths = berry_cloner.generate_paths("/World/envs/env_0/Berry", num_paths=5)
        # berry_env_positions = berry_cloner.clone(source_prim_path="/World/envs/env_0/Berry", prim_paths=envs_prim_paths, replicate_physics=False)
        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._post_scoop_markers = StaticMarker(
                "/Visuals/pre_scoop_pos",
                self.num_envs,
                usd_path=self.cfg.post_scoop_goal_marker.usd_path,
                scale=self.cfg.post_scoop_goal_marker.scale,
            )
            # create point instancer to visualize the goal points
            self._goal_markers = StaticMarker(
                "/Visuals/object_goal",
                self.num_envs,
                usd_path=self.cfg.goal_marker.usd_path,
                scale=self.cfg.goal_marker.scale,
            )
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current",
                self.num_envs,
                usd_path=self.cfg.frame_marker.usd_path,
                scale=self.cfg.frame_marker.scale,
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command",
                    self.num_envs,
                    usd_path=self.cfg.frame_marker.usd_path,
                    scale=self.cfg.frame_marker.scale,
                )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        #print("[Debug] env_ids to reset: ", env_ids)
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # # -- object pose
        # self._randomize_object_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_initial_pose)
        # # # -- goal pose
        # self._randomize_object_desired_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_desired_pose)

        self._randomize_food_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.food_initial_pos)
        self._randomize_spoon_post_scoop_pose(env_ids=env_ids, cfg=self.cfg.randomization.post_scoop_pose)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller.reset_idx(env_ids)

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        # transform actions based on controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # set the controller commands
            self._ik_controller.set_command(self.actions[:, :-1])
            # use IK to convert to joint-space commands
            self.robot_actions[:, : self.robot.arm_num_dof] = self._ik_controller.compute(
                self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
                self.robot.data.ee_state_w[:, 3:7],
                self.robot.data.ee_jacobian,
                self.robot.data.arm_dof_pos,
            )
            # offset actuator command with position offsets
            dof_pos_offset = self.robot.data.actuator_pos_offset
            self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset[:, : self.robot.arm_num_dof]
            # we assume last command is gripper action so don't change that
            self.robot_actions[:, -1] = self.actions[:, -1]
        elif self.cfg.control.control_type == "default":
            self.robot_actions[:] = self.actions
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.object.update_buffers(self.dt)
        self.bowl.update_buffers(self.dt)
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- add information to extra if task completed
        target_position_error = torch.norm(self.robot.data.ee_state_w[:, 0:3] - torch.Tensor(self.cfg.randomization.food_initial_pos.position_default).to(self.device), dim=1)
        self.extras["is_success"] = torch.where(target_position_error < self.cfg.terminations.sucess_threshold, 1, self.reset_buf)
        # object_position_error = torch.norm(self.object.data.root_pos_w - self.object_des_pose_w[:, 0:3], dim=1)
        # self.extras["is_success"] = torch.where(object_position_error < 0.002, 1, self.reset_buf)
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # DEBUG
        # visiblities = torch.Tensor([False, True, True, True]).to(self.device)
        # self.mango.objects.set_visibilities(visiblities)
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre-processing of configuration parameters."""
        # set configuration for task-space controller
        if self.cfg.control.control_type == "inverse_kinematics":
            print("Using inverse kinematics controller...")
            # enable jacobian computation
            self.cfg.robot.data_info.enable_jacobian = True
            # enable gravity compensation
            self.cfg.robot.rigid_props.disable_gravity = True
            # set the end-effector offsets
            self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
            self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        else:
            print("Using default joint controller...")

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torchee
        # randomization
        # -- initial pose
        config = self.cfg.randomization.object_initial_pose
        for attr in ["position_uniform_min", "position_uniform_max"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))
        # -- desired pose
        config = self.cfg.randomization.object_desired_pose
        for attr in ["position_uniform_min", "position_uniform_max", "position_default", "orientation_default"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.object.initialize(self.env_ns + "/.*/Object")
        self.tray.initialize(self.env_ns + "/.*/Tray")
        self.plate.initialize(self.env_ns + "/.*/Plate")
        self.bowl.initialize(self.env_ns + "/.*/Bowl")
        
        for fruit in self.sliced_fruits:
            self.sliced_fruits[fruit]['object'].initialize(self.env_ns + "/.*" + "/" + fruit.capitalize() + "_.*")
        # self.banana.initialize(self.env_ns + "/.*/Banana_*")
        # self.berry.initialize(self.env_ns + "/.*/Rs_berry_*")
        # self.banana.initialize(self.env_ns + "/.*/Banana_1")
        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            self._ik_controller = DifferentialInverseKinematics(
                self.cfg.control.inverse_kinematics, self.robot.count, self.device
            )
            self.num_actions = self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        # commands
        self.object_des_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.post_scoop_pos = torch.zeros((self.num_envs, 3),)

    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        # apply to instance manager
        # -- post_scoop_pos
        # print("DEBUG:", self.object_des_pose_w[:, 0:3])
        marker_pose = torch.zeros((self.num_envs,7))
        marker_pose[:, :3] = torch.Tensor(self.post_scoop_pos).to(self.device) + self.envs_positions
        marker_pose[:, 3:7] = torch.Tensor(self.cfg.randomization.post_scoop_pose.orientation_default).to(self.device)
        self._post_scoop_markers.set_world_poses(marker_pose[:, :3], marker_pose[:, 3:7])
        # -- goal
        self._goal_markers.set_world_poses(self.object_des_pose_w[:, 0:3], self.object_des_pose_w[:, 3:7])
        # -- end-effector
        # self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
        # -- task-space commands
        if self.cfg.control.control_type == "inverse_kinematics":
            # convert to world frame
            ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
            ee_orientations = self._ik_controller.desired_ee_rot
            # set poses
            self._cmd_markers.set_world_poses(ee_positions, ee_orientations)

    """
    Helper functions - MDP.
    """

    def _randomize_food_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.FoodInitialPoseCfg):
        object_base_pos = torch.zeros(len(env_ids),3, device=self.device)
        if cfg.position_cat == "default":
            object_base_pos[:,:] = torch.Tensor(cfg.position_default)
        elif cfg.position_cat == "uniform":
            # min = torch.tensor([self.work_space['x']['min'], self.work_space['y']['min'], self.work_space['z']['min']], device=self.device)
            # max = torch.tensor([self.work_space['x']['max'], self.work_space['y']['max'], self.work_space['z']['max']], device=self.device)
            min = torch.tensor(cfg.position_uniform_min, device=self.device)
            max = torch.tensor(cfg.position_uniform_max, device=self.device)
            # min = torch.tensor([1.368, -1.734, 1], device=self.device)
            # max = torch.tensor([1.968, -1.476, 1.5], device=self.device)
            object_base_pos = (min - max) * torch.rand(len(env_ids),3, device=self.device) + max

        # Randomize the position of current serving platform and food
        # -- stock objects to initial pos
        self._stock_scene_objects(default_scene=True, env_ids =env_ids)
        # self.rand_object_key
        # self.rand_fruit_key
        object_root_state = self.feed_care_objects[self.rand_object_key].get_default_root_state(env_ids=env_ids)
        #print("DEBUG: length of env_id: ", len(env_ids))
        #print("DEBUG: length of env_id: ", object_root_state.shape)
        object_root_state[:, 0:3] = object_base_pos
        object_root_state[:, 0:3] += self.envs_positions[env_ids]
        self.feed_care_objects[self.rand_object_key].set_root_state(root_states=object_root_state, env_ids=env_ids)
        # root_bowl_state = self.bowl.get_default_root_state()
        # root_bowl_state[:, 0:3] = object_base_pos
        # root_bowl_state[:, 0:3] += self.envs_positions[env_ids]
        # self.bowl.set_root_state(root_states=root_bowl_state, env_ids=env_ids)
        # print("pose after randomization: ", self.feed_care_objects[self.rand_object_key].data.root_pos_w - self.envs_positions )
        # move the fruit with the feed object
        """DEBUGGING
        indexing by setting visiblities
        """
        # visib = torch.ones(self.num_envs * 100).to(self.device)
        # visib[:100] = False
        # self.sliced_fruits[self.rand_fruit_key]['object'].objects.set_visibilities(visib)

        # generate slices ids from env_ids
        env_ids_ = torch.zeros(len(env_ids),100, dtype=torch.int64, device=self.device)
        env_pos = torch.zeros(len(env_ids)*100, 3, device=self.device)
        base_object_pos = torch.zeros(len(env_ids)*100, 3, device=self.device)
        for env_ in range(len(env_ids)):
            #print("index: ", env_, " env_ids: ", env_ids, "env_pos: ", env_pos)
            start = env_ids[env_]*100
            end = (env_ids[env_]+1)*100
            env_ids_[env_] = torch.arange(start=env_ids[env_]*100 ,end=(env_ids[env_]+1)*100, dtype=torch.int64, device=self.device)
            env_pos[start:end, :] = self.envs_positions[env_ids[env_], :]
            base_object_pos[start:end, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[env_ids[env_], :]
        env_ids_fruit = env_ids_.flatten()
        base_object_pos[:, 0:3] -= env_pos
        fruit_state = self.sliced_fruits[self.rand_fruit_key]['object'].get_default_root_state(env_ids=env_ids_fruit)
        # print(f'fruit state all index: {fruit_state}')
        # env_pos = torch.zeros(400, 3, device=self.device)
        # env_pos[:100, :] = self.envs_positions[0, :]
        # env_pos[100:200, :] = self.envs_positions[1, :]
        # env_pos[200:300, :] = self.envs_positions[2, :]
        # env_pos[300:400, :] = self.envs_positions[3, :]
        # base_object_pos = torch.zeros(len(env_ids)*100, 3, device=self.device)
        # base_object_pos[:100, :]    = self.feed_care_objects[self.rand_object_key].data.root_pos_w[0, :]
        # base_object_pos[100:200, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[1, :]
        # base_object_pos[200:300, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[2, :]
        # base_object_pos[300:400, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[3, :]
        # base_object_pos[:, 0:3] -= env_pos
        fruit_offset_min = torch.tensor([-0.05, -0.05, 0.01], device=self.device)
        fruit_offset_max = torch.tensor([0.05, 0.05, 0.4], device=self.device)
        fruit_offset = sample_uniform(fruit_offset_min, fruit_offset_max, (len(env_ids)*100, 3), device=self.device)
        fruit_state[:, 0:3] = base_object_pos + fruit_offset #(torch.rand(400, 3, device=self.device).uniform_(-0.1, 0.1)) # sample it
    
        fruit_state[:, 0:3] += env_pos    # create new env_pos????
        self.sliced_fruits[self.rand_fruit_key]['object'].set_root_state(root_states=fruit_state, env_ids=env_ids_fruit)



    def _randomize_eef_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.EefInitialPoseCfg):
        '''
        case1: whole action space, relative to robot_base/world pose
        case2: Sampled action space, relative to food pose
        '''
        pass

    def _randomize_spoon_post_scoop_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.PostScoopPoseCfg):
        if cfg.position_cat == "default":
            self.post_scoop_pos = cfg.position_default
        elif cfg.position_cat == "random":
            # randomize the post scoop pose uniformly within given box
            pass

    def _randomize_food_physics_props(self, env_ids: torch.Tensor, cfg: RandomizationCfg.FoodPhysicsPropsCfg):
        '''props that can be captured by f/t sensor such as static_friction, dynamic_friction, restitution'''
        if cfg.physics_props_cat == "default":
            pass

    def _randomize_type_of_food(self, env_ids: torch.Tensor, cfg: RandomizationCfg.FoodTypeCfg):
        '''Eventhough, physics props of each food category may not be modelled correctly, this randomization helps the agent to reason about geometric features of each food catagories'''
        if cfg.type_cat == "default":
            fruit_key = cfg.food_default

    def _randomize_way_of_serving(self, env_ids: torch.Tensor, cfg: RandomizationCfg.WayOfServingCfg):
        '''  '''
        if cfg.way_of_serving_cat == "default":
            feed_object_key = cfg.feed_object_default

    def _check_termination(self) -> None:
        # access buffers from simulator
        object_pos = self.object.data.root_pos_w - self.envs_positions
        bowl_pose = self.feed_care_objects[self.rand_object_key].data.root_state_w[:, 0:3]
        bowl_pose[:, 0:3] = bowl_pose[:, 0:3]- self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when bowl displaced
        if self.cfg.terminations.bowl_displaced:
            bowl_displacement = torch.norm(bowl_pose - torch.Tensor(self.cfg.randomization.food_initial_pos.position_default).to(self.device), dim=1)
            #self.reset_buf = torch.where(bowl_displacement < 0.02, 1, self.reset_buf)
            self.reset_buf = torch.where(bowl_displacement > self.cfg.terminations.bowl_displacement_threshold, 1, self.reset_buf)
        # -- when reaching to bowl pose is successful
        if self.cfg.terminations.is_success:
            target_position_error = torch.norm(self.robot.data.ee_state_w[:, 0:3] - torch.Tensor(self.cfg.randomization.food_initial_pos.position_default).to(self.device), dim=1)
            self.reset_buf = torch.where(target_position_error < self.cfg.terminations.sucess_threshold, 1, self.reset_buf)
        # -- object fell off the table (table at height: 0.0 m)
        # if self.cfg.terminations.object_falling:
        #     self.reset_buf = torch.where(object_pos[:, 2] < -0.05, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
        # print("[DEBUG] reset buffer: ", self.reset_buf)

    def _randomize_object_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectInitialPoseCfg):
        """Randomize the initial pose of the object."""
        # get the default root state
        root_state = self.object.get_default_root_state(env_ids)
        # -- object root position
        if cfg.position_cat == "default":
            pass
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            root_state[:, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
        # -- object root orientation
        if cfg.orientation_cat == "default":
            pass
        elif cfg.orientation_cat == "uniform":
            # sample uniformly in SO(3)
            root_state[:, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(f"Invalid category for randomizing the object orientation '{cfg.orientation_cat}'.")
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # update object init pose
        self.object_init_pose_w[env_ids] = root_state[:, 0:7]
        # set the root state
        self.object.set_root_state(root_state, env_ids=env_ids)

    def _randomize_object_desired_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectDesiredPoseCfg):
        """Randomize the desired pose of the object."""
        # -- desired object root position
        if cfg.position_cat == "default":
            # constant command for position
            self.object_des_pose_w[env_ids, 0:3] = cfg.position_default
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            self.object_des_pose_w[env_ids, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the desired object positions '{cfg.position_cat}'.")
        # -- desired object root orientation
        if cfg.orientation_cat == "default":
            # constant position of the object
            self.object_des_pose_w[env_ids, 3:7] = cfg.orientation_default
        elif cfg.orientation_cat == "uniform":
            self.object_des_pose_w[env_ids, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(
                f"Invalid category for randomizing the desired object orientation '{cfg.orientation_cat}'."
            )
        # transform command from local env to world
        self.object_des_pose_w[env_ids, 0:3] += self.envs_positions[env_ids]
        
        self.task_variation(fruit=False)
        # fruit_key = random.choice(list(self.sliced_fruits))
        # print(self.sliced_fruits[fruit_key].count)
        # print("BANANA COUNT: ", self.banana.count)
        # print("Banan pos: ", self.banana.data.root_state_w.shape)

        # print("ROOT TENSOR shape: ", self.banana.get_default_root_state().shape)
        # print("default root state: ", self.bowl.get_default_root_state())

        '''
        1. pick random pos in workspace at each episode               -----> DONE
        2. pick random care object and move to picked pos  each 8 episodes     ------> DONE
        3. pick random sliced fruit each 5 episodes
        '''
        """ # stock objects to initial pos
        self._stock_scene_objects()

        min = torch.tensor([self.work_space['x']['min'], self.work_space['y']['min'], self.work_space['z']['min']], device=self.device)
        max = torch.tensor([self.work_space['x']['max'], self.work_space['y']['max'], self.work_space['z']['max']], device=self.device)
        # min = torch.tensor([1.368, -1.734, 1], device=self.device)
        # max = torch.tensor([1.968, -1.476, 1.5], device=self.device)
        object_base_pos = (min - max) * torch.rand(len(env_ids),3, device=self.device) + max

        if self.episode_count % 1 == 0:   # Randomize position of food
            pass
        if self.episode_count % 5 == 0:   # Randomize type of food
            pass
        if self.episode_count % 7 == 0:   # Randomize rigid body physics props
            pass
        if self.episode_count % 10 == 0:  # Randomize way of serving
            self.rand_object_key = random.choice(list(self.feed_care_objects))
        object_root_state = self.feed_care_objects[self.rand_object_key].get_default_root_state()
        object_root_state[:, 0:3] = object_base_pos
        object_root_state[:, 0:3] += self.envs_positions[env_ids]
        self.feed_care_objects[self.rand_object_key].set_root_state(root_states=object_root_state)
        # root_bowl_state = self.bowl.get_default_root_state()
        # root_bowl_state[:, 0:3] = object_base_pos
        # root_bowl_state[:, 0:3] += self.envs_positions[env_ids]
        # self.bowl.set_root_state(root_states=root_bowl_state, env_ids=env_ids)
        print("pose after randomization: ", self.feed_care_objects[self.rand_object_key].data.root_pos_w - self.envs_positions )
        if self.episode_count % 5 == 0: # Rand Fruits
            self.rand_fruit_key = random.choice(list(self.sliced_fruits))
            fruit_state = self.sliced_fruits[self.rand_fruit_key]['object'].get_default_root_state()

            env_pos = torch.zeros(400, 3, device=self.device)
            env_pos[:100, :] = self.envs_positions[0, :]
            env_pos[100:200, :] = self.envs_positions[1, :]
            env_pos[200:300, :] = self.envs_positions[2, :]
            env_pos[300:400, :] = self.envs_positions[3, :]

            base_object_pos = torch.zeros(400, 3, device=self.device)
            base_object_pos[:100, :]    = self.feed_care_objects[self.rand_object_key].data.root_pos_w[0, :]
            base_object_pos[100:200, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[1, :]
            base_object_pos[200:300, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[2, :]
            base_object_pos[300:400, :] = self.feed_care_objects[self.rand_object_key].data.root_pos_w[3, :]
            base_object_pos[:, 0:3] -= env_pos


            fruit_state[:, 0:3] = base_object_pos + (torch.rand(400, 3, device=self.device).uniform_(-0.1, 0.1))
        
            fruit_state[:, 0:3] += env_pos    # create new env_pos????
            self.sliced_fruits[self.rand_fruit_key]['object'].set_root_state(root_states=fruit_state)

            self.episode_count += 1




        # fruit_key = random.choice(list(self.sliced_fruits))
        # fruit_root_state = self.sliced_fruits[fruit_key]['object'].get_default_root_state()
        # fruit_root_state[:, 0:3] = base_bowl_pos + (torch.rand(400, 3, device=self.device).uniform_(-0.1, 0.1))


        # self.banana.objects.enable_rigid_body_physics()
        # print(self.envs_positions)
        # print("Root state: ", self.bowl.get_default_root_state())
        # banana_pos = torch.ones((200, 13), devive=self.device)
        # self.banana.set_root_state(root_states=banana_pos)

        """

    def _clone_scene(self, target_path: str, source_path: str, num_clones: int, spacing = 0.01):
        ''' 
        Clone each slice of fruits spawned on stage                      
         target path: "/World/envs/env_0/Object"
         source_path: "/World/envs/env_0/Object_0"
        '''
        # create a GridCloner instance
        scene_cloner = GridCloner(spacing)
        target_paths = scene_cloner.generate_paths(target_path, num_clones)
        scene_cloner.clone(source_prim_path=source_path, prim_paths=target_paths)
        ''' filter physics, cloned scene poses???'''

    def _stock_scene_objects(self, default_scene: bool=False, env_ids: torch.Tensor = None):
        """ 
        default_scene: False, set all object to stock pos:
        default_scene: True, set all object to stock pos except default scene objects set in randm cfg"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
            env_ids_fruit = torch.arange(len(env_ids)*100, dtype=torch.int64, device=self.device)
            env_pos = torch.zeros(400, 3, device=self.device)
            env_pos[:100, :] = self.envs_positions[0, :]
            env_pos[100:200, :] = self.envs_positions[1, :]
            env_pos[200:300, :] = self.envs_positions[2, :]
            env_pos[300:400, :] = self.envs_positions[3, :]
        else:
            env_pos = torch.zeros((len(env_ids)*100,3), device=self.device)
            #print("env_ids: ", env_ids)
            env_ids_ = torch.zeros(len(env_ids),100, dtype=torch.int64, device=self.device)
            for env_ in range(len(env_ids)):
                #print("index: ", env_, " env_ids: ", env_ids, "env_pos: ", env_pos)
                start = env_ids[env_]*100
                end = (env_ids[env_]+1)*100
                env_pos[start:end, :] = self.envs_positions[env_ids[env_], :]
                env_ids_[env_] = torch.arange(start=env_ids[env_]*100 ,end=(env_ids[env_]+1)*100, dtype=torch.int64, device=self.device) 
            env_ids_fruit = env_ids_.flatten()
            #print("env_ids_fruit: ", env_ids_fruit)
            
        bowl_stock_pos = torch.tensor([0, 0.5, 0.05], device= self.device) * torch.ones(len(env_ids), 3, device=self.device)
        plate_stock_pos = torch.tensor([0, 0.8, 0.05], device= self.device) * torch.ones(len(env_ids), 3, device=self.device)
        tray_stock_pos = torch.tensor([0, 1, 0.05], device= self.device) * torch.ones(len(env_ids), 3, device=self.device)
        self.stock_pos = {'bowl': bowl_stock_pos, 'plate':plate_stock_pos, 'tray':tray_stock_pos}

        # print("Default root state: ", root_bowl_state)
        # print("pose before: ", self.bowl.data.root_pos_w - self.envs_positions )
        # self.bowl.set_root_state(root_states=root_bowl_state, env_ids=env_ids)
        # print("pose after: ", self.bowl.data.root_pos_w - self.envs_positions )

        for feed_object in self.feed_care_objects:
            root_object_state = self.feed_care_objects[feed_object].get_default_root_state(env_ids=env_ids)
            if feed_object == self.cfg.randomization.way_of_serving.feed_object_default and default_scene is True:
                root_object_state[:, 0:3] = torch.tensor(self.cfg.randomization.food_initial_pos.position_default, device= self.device) * torch.ones(len(env_ids), 3, device=self.device)
            else:
                root_object_state[:, 0:3] = self.stock_pos[feed_object]
            root_object_state[:, 0:3] += self.envs_positions[env_ids]
            self.feed_care_objects[feed_object].set_root_state(root_states=root_object_state, env_ids=env_ids)


        fruit_offset_min = torch.tensor([-0.05, -0.05, 0.01], device=self.device)
        fruit_offset_max = torch.tensor([0.05, 0.05, 0.4], device=self.device)
        fruit_offset = sample_uniform(fruit_offset_min, fruit_offset_max, (len(env_ids)*100, 3), device=self.device)
        for sliced_fruit in self.sliced_fruits:
            root_fruit_state = self.sliced_fruits[sliced_fruit]['object'].get_default_root_state(env_ids=env_ids_fruit)
            if sliced_fruit == self.cfg.randomization.food_type.food_default and default_scene is True:
                base_bowl_pos = torch.tensor(self.cfg.randomization.food_initial_pos.position_default, device=self.device)
            else:
                base_bowl_pos = torch.tensor([0.0, 0.0, 0.1], device=self.device)

            root_fruit_state[:, 0:3] = base_bowl_pos + fruit_offset
            root_fruit_state[:, 0:3] += env_pos
            self.sliced_fruits[sliced_fruit]['object'].set_root_state(root_states=root_fruit_state, env_ids=env_ids_fruit)

    '''
        root_banana_state = self.banana.get_default_root_state()
        root_rs_berry_state = self.rs_berry.get_default_root_state()
        print ("root_shape", root_banana_state.shape)
        print ("root_shape", root_banana_state)
        print("World pose: ", self.banana.objects.get_world_poses())
        base_bowl_pos = torch.tensor([0.0, 0.0, 0.1], device=self.device)
        root_banana_state[:, 0:3] = base_bowl_pos + (torch.rand(400, 3, device=self.device).uniform_(-0.1, 0.1))
        root_rs_berry_state[:, 0:3] = base_bowl_pos + (torch.rand(400, 3, device=self.device).uniform_(-0.1, 0.1))

        env_pos = torch.zeros(400, 3, device=self.device)
        env_pos[:100, :] = self.envs_positions[0, :]
        env_pos[100:200, :] = self.envs_positions[1, :]
        env_pos[200:300, :] = self.envs_positions[2, :]
        env_pos[300:400, :] = self.envs_positions[3, :]

        root_banana_state[:, 0:3] += env_pos    # create new env_pos????
        root_rs_berry_state[:, 0:3] += env_pos
        root_states = {'banana': root_banana_state, 'rs_berry': root_rs_berry_state, 'mango': root_banana_state}
        root_state_key = random.choice(list(root_states))
        if root_state_key == 'banana':
            self.banana.set_root_state(root_states=root_states[root_state_key])
        if root_state_key == 'rs_berry':
            self.rs_berry.set_root_state(root_states=root_states[root_state_key])
        if root_state_key == 'mango':
            self.mango.set_root_state(root_states=root_states[root_state_key])

    '''

    def task_variation(self, fruit=False, plate=False, bowl=False):
        if fruit is True:
            fruit_key = random.choice(list(self.sliced_fruits))
            # remove fruit prim
            for num_slices in range(self.sliced_fruits[fruit_key]['num_slices']):
                if prim_utils.is_prim_path_valid(prim_path=self.template_env_ns + "/Fruit_" + str(num_slices)):
                    prim_utils.delete_prim(prim_path=self.template_env_ns + "/Fruit_" + str(num_slices))
                    print("Deleted ", self.template_env_ns + "/Fruit_" + str(num_slices))
            # spawn random fruit
            for num_slices in range(self.sliced_fruits[fruit_key]['num_slices']):
                self.sliced_fruits[fruit_key]['object'].spawn(self.template_env_ns + "/Fruit_" + str(num_slices), translation=self.dishes_world_pos)
                print(fruit_key, "spawned")

            # self.sim.reset()
            # for berry in range(self.num_berries):
            #     self.sliced_fruits[fruit_key].initialize(self.env_ns + "/.*/Fruit_" + str(berry))
            # # reset buffer
            # self.sliced_fruits[fruit_key].reset_buffers(self.dt)
            # if self.episode_count % 3 == 0:
            #     serving_way_key = random.choice(list(self.serving_way))
            #     for feed_objects in self.serving_way[serving_way_key]:
            #         print(feed_objects)
# 
            # self.episode_count += 1


class ScoopObservationManager(ObservationManager):
    """Observation Manager for scoop environment.

    Proprioceptive

    1. joint positions
    2. joint velocities
    3. Spoon positions/tool positions
    4. Bowl positions
    5. Food positions
    6. Desired post scoop positions
    7. Last actions
    11. joint forces

    Exteroceptive

    8. Camera on hand rgb
    9. Camera on hand depth
    9. Camera front rgb
    10. Camera front depth
    12. eef forces

    Steps   working on training scoop task on a state vectors

    1. Determine the relevant features of the environment: Identify the key variables or attributes that are important for the agent to make scooping decisions
    2. Determine the data type and range of the observation space: Decide on the data type (e.g., continuous, discrete) and range (e.g., minimum and maximum values) of the observation space.
    3. Implement the observation space in code: Creating a class that represents the observation space and defining methods for updating and accessing the observation.
    4. Normalize the observation space: In some cases, it may be helpful to normalize the observation space to ensure that all features are on the same scale.
    """

    def arm_dof_pos_scaled(self, env: ScoopEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 1],
        )

    def arm_dof_vel(self, env: ScoopEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def spoon_positions(self, env: ScoopEnv):
        """Current end-effector position of the arm."""
        spoon_pose = env.robot.data.ee_state_w[:, 0:7]
        #spoon_pose = env.robot.data.ee_state_b[:, 0:7]
        spoon_pose[:, 0:3] = spoon_pose[:, 0:3] - env.envs_positions
        # TODO trans to robot_base
        # print("[INFO] eef_pos: ", spoon_pose)
        return spoon_pose

    def bowl_positions(self, env: ScoopEnv):
        """Current bowl position."""
        bowl_pose = env.feed_care_objects[env.rand_object_key].data.root_state_w[:, 0:7]
        # print("bowl pos world: ", bowl_pose)
        bowl_pose[:, 0:3] = bowl_pose[:, 0:3]- env.envs_positions
        # print("bowl pos env: ", bowl_pose)
        # bowl_pose[:, 0:3] = bowl_pose[:, 0:3]- torch.Tensor([1.69104, -0.77251, 0.931]).to(self.device)
        # print("bowl pos base: ", bowl_pose)
        return bowl_pose
    def food_positions(self, env: ScoopEnv):
        """Current food postion."""
        food_pose = env.sliced_fruits[env.rand_fruit_key]['object'].data.root_state_w[:4, 0:7]
        food_pose[:, 0:3] = food_pose[:, 0:3] - env.envs_positions
        return food_pose

    def post_scoop_positions(self, env: ScoopEnv):
        """Desired post scoop position."""
        post_scoop_pose = env.object_des_pose_w
        post_scoop_pose[:, 0:3] = post_scoop_pose[:, 0:3] - env.envs_positions
        return post_scoop_pose

    def actions(self, env: ScoopEnv):
        """Last actions provided to env."""
        return env.actions
    
    def dof_torque(self, env:ScoopEnv):
        """DOF torques applied from the actuator model (after clipping)."""
        return env.robot.data.applied_torques
    
    def eef_jacobian(self, env:ScoopEnv):
        """Jacobian of the parent body of end-effector frame in simulation frame."""
        return env.robot.data.ee_jacobian
    
    def rgb_on_hand(self, env:ScoopEnv):
        return env.robot.data.arm_dof_vel
    
    def depth_on_hand(self,env: ScoopEnv):
        return env.robot.data.arm_dof_vel

    def rgb_front(self, env:ScoopEnv):
        return env.robot.data.arm_dof_vel

    def depth_front(self, env:ScoopEnv):
        return env.robot.data.arm_dof_vel

    def eef_force(self, env: ScoopEnv):
        return env.robot.data.arm_dof_vel


class ScoopRewardManager(RewardManager):
    """Reward manager for scoop environment."""

    def tracking_robot_position_l2(self, env: ScoopEnv):
        """Penalize tracking position error using L2-kernel."""
        # compute error
        return torch.sum(torch.square(env.bowl.data.root_pos_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
    
    def tracking_robot_position_exp(self, env: ScoopEnv, sigma: float):
        """Penalize tracking position error using exp-kernel."""
        # compute error
        error = torch.sum(torch.square(env.bowl.data.root_pos_w[:, :3] - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        # print(f'error to bowl base: {torch.exp(-error / sigma)}')
        return torch.exp(-error / sigma)
    
    def penalizing_robot_dof_velocity_l2(self, env: ScoopEnv):
        """Penalize large movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_robot_dof_acceleration_l2(self, env: ScoopEnv):
        """Penalize fast movements of the robot arm."""
        return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)

    def penalizing_action_rate_l2(self, env: ScoopEnv):
        """Penalize large variations in action commands."""
        return torch.sum(torch.square(env.actions - env.previous_actions), dim=1)

    def reaching_bowl_success(self, env: ScoopEnv, threshold: float):
        """Sparse reward if the goal is reached successfully."""
        error = torch.sum(torch.square(env.bowl.data.root_pos_w - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        return torch.where(error < threshold, 3.5, 0.0)
    
    def tracking_pre_scoop_pos_exp(self, env: ScoopEnv, sigma: float, prev_goal_threshold: float ):
        """Penalize tracking position error b/n spoon and pre_scoop_pos using exp-kernel."""
        pos_tracking_err = torch.sum(torch.square(torch.Tensor(env.post_scoop_pos).to(self.device) - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        # print(f'[DEBUG] pos_tracking_err: {pos_tracking_err}')
        pos_tracking_err = torch.exp(-pos_tracking_err / sigma)
        # check bowl base reach sucess condition
        error = torch.sum(torch.square(env.bowl.data.root_pos_w - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        sucess_ids = torch.where(error < prev_goal_threshold)
        reward_post_scoop = pos_tracking_err
        # reward a successful environments
        for env_id in range(env.num_envs):
            if not env_id in list(sucess_ids[0]):
                reward_post_scoop[env_id] = 0.0
        # print(f'[DEBUG] prev_goal_threshold: {prev_goal_threshold} post_scoop_pos: {torch.Tensor(env.post_scoop_pos).to(self.device)} pos_tracking_err: {pos_tracking_err} sucess_ids: {sucess_ids} reward_post_scoop: {reward_post_scoop}')
        return reward_post_scoop
    
    def tracking_pre_scoop_pos_l2(self, env: ScoopEnv, prev_goal_threshold: float):
        """Penalize tracking position error b/n spoon and pre_scoop_pos using exp-kernel."""
        pos_tracking_err = torch.sum(torch.square(torch.Tensor(env.post_scoop_pos).to(self.device) - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        # check bowl base reach sucess condition
        error = torch.sum(torch.square(env.bowl.data.root_pos_w - env.robot.data.ee_state_w[:, 0:3]), dim=1)
        sucess_ids = torch.where(error < prev_goal_threshold)
        reward_post_scoop = pos_tracking_err
        # reward a successful environments
        for env_id in range(env.num_envs):
            if not env_id in list(sucess_ids[0]):
                reward_post_scoop[env_id] = 0.0
        # print(f'[DEBUG] prev_goal_threshold: {prev_goal_threshold} post_scoop_pos: {torch.Tensor(env.post_scoop_pos).to(self.device)} pos_tracking_err: {pos_tracking_err} sucess_ids: {sucess_ids} reward_post_scoop: {reward_post_scoop}')
        return reward_post_scoop

    """
    Remove bonus for reach bowl sucess since exp kernel error is applied

    """
