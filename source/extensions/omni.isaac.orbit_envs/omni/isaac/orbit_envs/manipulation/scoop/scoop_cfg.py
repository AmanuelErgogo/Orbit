# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

#from omni.isaac.orbit.robots.config.xarm import XARM_ARM_WITH_XARM_GRIPPER_CFG
from omni.isaac.orbit.robots.config.xarm_spoon import XARM_ARM_WITH_SPOON_CFG

##
# Scene settings
##


@configclass
class TableCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

@configclass
class HospitalSceneCfg:
    """Properties for the hospital room."""
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/environments/geo_o_room_down.usd"
class HospitalSceneReducedCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/environments/geo_o_room_reduced.usd"

@configclass
class BookCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/others/book2.usd"

@configclass
class FruitsComboCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits.usd"

@configclass
class SpecialDishCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/special_dish_1.usd"

@configclass
class DishSlicedAppleCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/dish_sliced_apple.usd"

@configclass
class BowlCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/care_objects/bowl_instanceable.usd",
        scale=(0.023, 0.023, 0.015),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=100.0,
        max_linear_velocity=100.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.9, dynamic_friction=0.9, restitution=0.0, prim_path="/World/Materials/bowlMaterial"
    )

@configclass
class PlateCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/care_objects/plate_instanceable.usd",
        scale=(0.1, 0.1, 0.1),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/plateMaterial"
    )

@configclass
class TrayCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/care_objects/tray_instanceable.usd",
        scale=(0.1, 0.1, 0.1),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/trayMaterial"
    )


@configclass
class DexCubeCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.8, 0.8, 0.8),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )


@configclass
class FoodMacroniCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/macron_instanceable.usd",
        scale=(1, 1, 1),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/macaronlMaterial"
    )

@configclass
class FruitsBerryCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/berry_instanceable.usd",
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/fruit_berryMaterial"
    )
@configclass
class FruitsAppleCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits/apple_instanceable.usd",
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/appleMaterial"
    )

@configclass
class FruitsBananaCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits/banana_instanceable.usd",
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/bananaMaterial"
    )

@configclass
class FruitsMangoCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits/mango_instanceable.usd",
        scale=(0.048, 0.048, 0.048),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=10.0,
        max_linear_velocity=10.0,
        max_depenetration_velocity=10000.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.9, dynamic_friction=0.9, restitution=0.0, prim_path="/World/Materials/mangoMaterial"
    )

@configclass
class FruitsMelonCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits/melon_instanceable.usd",
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/melonMaterial"
    )

@configclass
class FruitsRsBerryCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path="/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/food/fruits/rs_berry_instanceable.usd",
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=False,
    )
    material_props = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/rs_berryMaterial"
    )


@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
    # scale of the asset at import
    scale = [0.8, 0.8, 0.8]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z

@configclass
class PostScoopGoalMarkerCfg:
    usd_path = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/eef_viz.usd"
    scale = [1e-3, 1e-3, 1e-3]


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scoop env scene.
    Properties of scene to be randomized(most to least essential)
        1. Initial position of food
        2. Initial position of endeffector/spoon
        3. Post scoop pose
        4. Way of serving
        5. Food type/geometry
        6. Physics props of food
    """

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [0.25, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.5, 0.25, 0.5]  # position (x,y,z)

    @configclass
    class ObjectDesiredPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.0, 0.5]  # position default (x,y,z)
        position_uniform_min = [0.25, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.5, 0.25, 0.5]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    @configclass
    class FoodInitialPoseCfg:
        """Randomization of food initial pose."""
        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        position_default = [1.700, -0.35, 0.9]
        # randomize position
        position_uniform_min = [1.521, -0.445, 0.9]  # position (x,y,z)
        position_uniform_max = [1.856, -0.238, 0.93]  # position (x,y,z)

    @configclass
    class EefInitialPoseCfg:
        """Randomization of eef initial pose."""
        pass

    @configclass
    class PostScoopPoseCfg:
        """ The pose of spoon at the end of scooping policy before the bite transfer """
         # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [1.650, -0.32, 1.2]  # position default (x,y,z)
        position_uniform_min = [0.25, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.5, 0.25, 0.5]  # position (x,y,z)
        # randomize orientation
        orientation_default = [0.910, 0.377, -0.006, 0.160]  # orientation default

    @configclass
    class FoodPhysicsPropsCfg:
        """Props that can be captured by f/t sensor such as static_friction, dynamic_friction, restitution"""
        # category
        physics_props_cat: str = "default"  # randomize physics props of food "default", "random"
        

    @configclass
    class FoodTypeCfg:
        """ """
        # category
        type_cat: str = "default"  # randomize food type: "default", "random"
        food_default = "mango"

    @configclass
    class WayOfServingCfg:
        """ """
        # category
        way_of_serving_cat: str = "default"  # randomize way of serving food: "default", "random"
        feed_object_default = "bowl"

    # initialize
    food_initial_pos: FoodInitialPoseCfg = FoodInitialPoseCfg()
    post_scoop_pose: PostScoopPoseCfg = PostScoopPoseCfg()
    food_physics_prop: FoodPhysicsPropsCfg = FoodPhysicsPropsCfg()
    food_type: FoodTypeCfg = FoodTypeCfg()
    way_of_serving: WayOfServingCfg = WayOfServingCfg()

    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """State vector based observation group."""
        # global group settings
        enable_corruption: bool = True
        arm_dof_pos_scaled = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        spoon_positions = {}
        bowl_positions = {}
        # food_positions = {}
        # post_scoop_positions = {}
        actions = {}
    
    @configclass
    class OracleJointTorqueCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        arm_dof_pos_scaled = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        dof_torque = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        spoon_positions = {}
        bowl_positions = {}
        food_positions = {}
        post_scoop_positions = {}
        actions = {}
    
    @configclass
    class RawImageCfg:
        """Observations for raw_image group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        arm_dof_pos_scaled = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        dof_torque = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.1, "max": 0.1}}
        # spoon_positions = {}
        # bowl_positions = {}
        # food_positions = {}
        rgb_on_hand = {}
        depth_on_hand = {}
        eef_force = {}
        post_scoop_positions = {}
        actions = {}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # oracle_joint_torque: OracleJointTorqueCfg = OracleJointTorqueCfg()
    # raw_image: RawImageCfg = RawImageCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_robot_position_l2 = {"weight": -1e-4}
    tracking_robot_position_exp = {"weight": 2.5, "sigma": 0.05}  # 0.25
    penalizing_robot_dof_velocity_l2 = {"weight": -0.02}  # -1e-4
    penalizing_robot_dof_acceleration_l2 = {"weight": -1e-5}
    penalizing_action_rate_l2 = {"weight": -0.1}
    reaching_bowl_success = {"weight": 3.5, "threshold": 2e-2}
    tracking_pre_scoop_pos_exp = {"weight": 2.5, "sigma": 0.05, "prev_goal_threshold": 2e-2}
    tracking_pre_scoop_pos_l2 = {"weight": -1e-4, "prev_goal_threshold": 2e-2}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    bowl_displaced = True # reset when bowl displaced
    episode_timeout = True  # reset when episode length ended
    object_falling = False  # reset when object falls off the table
    is_success = True  # reset when the bowl is reached -- temp condition
    bowl_displacement_threshold = 0.1 # 5e-2  # 0.1
    sucess_threshold = 1.8e-2 # 3e-2 #1.95e-2 # 1.8e-2


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_rel",
        ik_method="dls",
        position_command_scale=(0.1, 0.1, 0.1),
        rotation_command_scale=(0.1, 0.1, 0.1),
    )


##
# Environment configuration
##


@configclass
class ScoopEnvCfg(IsaacEnvCfg):
    """Configuration for the scoop environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4, env_spacing=8, episode_length_s=4.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=1.0 / 60.0,
        substeps=1,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=512 * 1024,
            gpu_total_aggregate_pairs_capacity=6 * 1024,
        ),
    )

    # Scene Settings

    '''
    Environments: {home: {type1:'', type2:'', type3:''}, hospital: {type1:'', type2:'', type3:''}, kitchen: {type1:'', type2:'', type3:''}}
    '''
    # -- table
    table: TableCfg = TableCfg()
    # -- hospital
    hospital: HospitalSceneCfg = HospitalSceneCfg()
    # -- hospital reduced
    hospital_reduced: HospitalSceneReducedCfg= HospitalSceneReducedCfg()

    # -- robot_spoon
    robot: SingleArmManipulatorCfg = XARM_ARM_WITH_SPOON_CFG
    '''robot: {with_spoon: '', with_fork: '', with_chopstic: ''}'''

    # -- feed care objects
    bowl: BowlCfg = BowlCfg()
    plate: PlateCfg = PlateCfg()
    tray: TrayCfg = TrayCfg()

    # sliced fruits
    berry: FruitsBerryCfg = FruitsBerryCfg()
    apple: FruitsAppleCfg = FruitsAppleCfg()
    banana: FruitsBananaCfg = FruitsBananaCfg()
    mango: FruitsMangoCfg = FruitsMangoCfg()
    melon: FruitsMelonCfg = FruitsMelonCfg()
    rs_berry: FruitsRsBerryCfg = FruitsRsBerryCfg()

    # food
    macron: FoodMacroniCfg = FoodMacroniCfg()
    # -- FruitsCombo
    fruits_combo: FruitsComboCfg = FruitsComboCfg()
    # -- special dish
    special_dish: SpecialDishCfg = SpecialDishCfg()
    # -- sliced apple dish
    dish_sliced_apple: DishSlicedAppleCfg = DishSlicedAppleCfg()

    # Distractors 
    # -- book
    book: BookCfg = BookCfg()
    object: DexCubeCfg = DexCubeCfg()

    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()
    post_scoop_goal_marker: PostScoopGoalMarkerCfg = PostScoopGoalMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
