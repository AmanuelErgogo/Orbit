"""
xArm7 robot with spoon config
"""


from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..single_arm import SingleArmManipulatorCfg

#_XARM_WITH_GRIPPER_INSTANCEABLE_USD = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/xarm_model_gripper_defalt_init/xarm_model_gripper_defalt_init.usd"
_XARM_WITH_SPOON_INSTANCEABLE_USD = "/home/aman/.local/share/ov/pkg/isaac_sim-2022.2.0/arcare/asset/robots/xarm_without_gripper/xarm7_spoon_bright.usd"

XARM_ARM_WITH_SPOON_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=_XARM_WITH_SPOON_INSTANCEABLE_USD,
        arm_num_dof=7,
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
            "joint1": 0.0,
            "joint2": -0.79,
            "joint3": 0.0,
            "joint4": 0.8045,
            "joint5": 0.0,
            "joint6": 1.473,
            "joint7": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(
        # body_name="xarm_gripper_base_link", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
        body_name="link_eef", pos_offset=(0.03, 0.0, 0.23), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    actuator_groups={
        "xarm7": ActuatorGroupCfg(
            dof_names=["joint[1-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 1250.0}, # 850
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
    },
)
"""Configuration of xArm7 with xArm Gripper using implicit actuator models."""
