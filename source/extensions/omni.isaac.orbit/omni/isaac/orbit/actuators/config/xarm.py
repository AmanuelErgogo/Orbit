# xArm gripper config

from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg

"""
Actuator Groups.
"""

XARM_GRIPPER_MIMIC_GROUP_CFG = GripperActuatorGroupCfg(
    dof_names=["drive_joint", ".*_inner_knuckle_joint", ".*_finger_joint", ".*right_outer_knuckle_joint"],
    model_cfg=ImplicitActuatorCfg(velocity_limit=2.0, torque_limit=3000.0),
    control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 1e6}, damping={".*": 1e5}),
    mimic_multiplier={
        "drive_joint": 1.0,  # mimicked joint
        ".*_inner_knuckle_joint": -1.0,
        ".*_finger_joint": 1.0,
        ".*right_outer_knuckle_joint": -1.0,
    },
    speed=0.01,
    open_dof_pos=0.85,
    close_dof_pos=-10,
)