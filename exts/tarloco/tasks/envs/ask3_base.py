import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

import exts.tarloco.envs.mdp as mdp
from exts.tarloco.utils.terrains_cfg import ROUGH_TERRAINS_CFG
from .base import (
    BaseLocomotionVelocityEnvCfg,
    EvaluationConfigMixin,
    RoughSceneCfg,
    RewardsCfg,
    EventCfg,
    TerminationsCfg,
    TerminationsEvalCfg,
)
from .ask3_cfg import ASK3_CFG


@configclass
class Ask3RoughSceneCfg(RoughSceneCfg):
    """Scene config using ASK-3 instead of Go1."""

    robot: ArticulationCfg = ASK3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
        update_period=0.0,
    )


@configclass
class Ask3RewardsCfg(RewardsCfg):
    """Rewards adapted for ASK-3 body names."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Link2"),
            "threshold": 1.0,
        },
    )


@configclass
class Ask3EventCfg(EventCfg):
    """Events adapted for ASK-3 body names."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "static_friction_range": (0.5, 0.8),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Ask3TerminationsCfg(TerminationsCfg):
    """Terminations adapted for ASK-3 body names."""

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_Link1"]),
            "threshold": 1.0,
        },
    )


@configclass
class BaseAsk3LocomotionVelocityEnvCfg(BaseLocomotionVelocityEnvCfg):
    """Base locomotion config for ASK-3 robot."""

    terrain_cfg_class: type = Ask3RoughSceneCfg

    rewards: Ask3RewardsCfg = Ask3RewardsCfg()
    events: Ask3EventCfg = Ask3EventCfg()
    terminations: Ask3TerminationsCfg = Ask3TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ASK-3 base link is named 'base', not 'trunk'
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # Domain randomization for ASK-3
        self.events.physics_material.params["static_friction_range"] = (0.15, 3.16)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 3.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 1.00)
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 10.0)
        self.events.add_base_mass.params["recompute_inertia"] = True
