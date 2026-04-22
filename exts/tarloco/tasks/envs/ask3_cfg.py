import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASK3_USD_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../assets/ask_3_description/usd/ask_3.usd",
)

ASK3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ASK3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "LF_HAA": 0.1,
            "LH_HAA": 0.1,
            "RF_HAA": -0.1,
            "RH_HAA": -0.1,
            "LF_HFE": 0.8,
            "RF_HFE": 0.8,
            "LH_HFE": 1.0,
            "RH_HFE": 1.0,
            ".*_KFE": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            effort_limit=25.0,
            velocity_limit=1000.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)
