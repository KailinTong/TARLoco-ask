from isaaclab.utils import configclass

from .base import EvaluationConfigMixin, FullObservationsCfg
from .ask3_base import BaseAsk3LocomotionVelocityEnvCfg


@configclass
class TarAsk3LocomotionVelocityRoughEnvCfg(BaseAsk3LocomotionVelocityEnvCfg):
    """TAR RNN training config for ASK-3 on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = FullObservationsCfg()
        self.observations.policy.history_length = 4
        self.observations.policy.flatten_history_dim = False
        del self.observations.policy.base_lin_vel
        del self.observations.policy.height_scan
        del self.observations.policy.base_external_force
        del self.observations.policy.feet_contact_z
        del self.observations.policy.contact_friction
        del self.observations.policy.base_mass
        self.observations.critic.history_length = 1
        self.observations.critic.flatten_history_dim = False


@configclass
class TarAsk3LocomotionVelocityRoughEnvEvalCfg(EvaluationConfigMixin, TarAsk3LocomotionVelocityRoughEnvCfg):
    """TAR RNN evaluation config for ASK-3 on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
