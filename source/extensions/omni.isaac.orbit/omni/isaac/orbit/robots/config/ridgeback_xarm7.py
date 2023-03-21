# xarm7 mobile base configuration


from omni.isaac.orbit.actuators.config.xarm import XARM_GRIPPER_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg


from ..mobile_manipulator import MobileManipulatorCfg

_RIDGEBACK_XARM_WITH_XARM_GRIPPER_USD = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/xarm7_mobile.usd"


RIDGEBACK_XARM_WITH_GRIPPER_CFG = MobileManipulatorCfg(
    meta_info=MobileManipulatorCfg.MetaInfoCfg(
        usd_path=_RIDGEBACK_XARM_WITH_XARM_GRIPPER_USD,
        base_num_dof=3,
        arm_num_dof=7,
        tool_num_dof=6,
        tool_sites_names=["left_finger", "right_finger"],
    ),
    init_state=MobileManipulatorCfg.InitialStateCfg(
        dof_pos={
            # base
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # xarm7 arm
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "drive_joint": 0.0,
            ".*_inner_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            ".*_finger_joint": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=MobileManipulatorCfg.EndEffectorFrameCfg(
        body_name="link_tcp", pos_offset=(0.0, 0.0, 0.0), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    actuator_groups={
        "base": ActuatorGroupCfg(
            dof_names=["dummy_base_.*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=1000.0),
            control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 0.0}, damping={".*": 1e5}),
        ),
        "xarm7": ActuatorGroupCfg(
            dof_names=["joint[1-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "joint1": 0.0,
                    "joint2": 0.0,
                    "joint3": 0.0,
                    "joint4": 0.0,
                    "joint5": 0.0,
                    "joint6": 0.0,
                    "joint7": 0.0,
                },
            ),
        ),
        "xarm_gripper": XARM_GRIPPER_MIMIC_GROUP_CFG,
    },
)
"""Configuration of Franka arm with Franka Hand on a Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""
