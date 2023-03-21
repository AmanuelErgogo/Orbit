"""
xArm7 robot config
"""


from omni.isaac.orbit.actuators.config.xarm import XARM_GRIPPER_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..single_arm import SingleArmManipulatorCfg

# _XARM_WITH_GRIPPER_INSTANCEABLE_USD = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/xarm_model_gripper_defalt_init/xarm_model_gripper_defalt_init.usd"
_XARM_WITH_GRIPPER_INSTANCEABLE_USD = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/xarm_with_gripper/xarm_model_gripper_defalt_init.usd"

XARM_ARM_WITH_XARM_GRIPPER_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=_XARM_WITH_GRIPPER_INSTANCEABLE_USD,
        arm_num_dof=7,
        tool_num_dof=6,
        tool_sites_names=["left_finger", "right_finger"],
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
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
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(
        # body_name="xarm_gripper_base_link", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
        body_name="link_tcp", pos_offset=(0.0, 0.0, 0.0), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    actuator_groups={
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
"""Configuration of xArm7 with xArm Gripper using implicit actuator models."""
