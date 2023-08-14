import torch
from functorch import vmap
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
import omni.isaac.core.utils.torch as torch_utils
from typing import Optional

class Omav(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/omav.usd"
    param_path: str = ASSET_PATH + "/usd/omav.yaml"

    def __init__(self, name: str = "Omav", cfg=None) -> None:
        super().__init__(name, cfg)
        self.action_spec = BoundedTensorSpec(-1, 1, 12 + 6, device=self.device)
        self.tilt_dof_indices = torch.arange(0, 6, device=self.device)
        self.rotor_dof_indices = torch.arange(6, 18, device=self.device)
        self.max_tilt_velocity = 10

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors + 6, device=self.device)
        self.state_spec = UnboundedContinuousTensorSpec(
            19 + self.num_rotors + 12, device=self.device
        )

    def initialize(self, prim_paths_expr: str = None):
        if not self.is_articulation:
            raise NotImplementedError
        super().initialize(prim_paths_expr)
        self.init_joint_positions = self._view.get_joint_positions()
        self.init_joint_velocities = self._view.get_joint_velocities()
    
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds, tilt_cmds = actions.expand(*self.shape, 18).split([12, 6], dim=-1)
        super().apply_action(rotor_cmds)

        velocity_targets = tilt_cmds.clamp(-1, 1) * self.max_tilt_velocity
        self._view.set_joint_velocity_targets(
            velocity_targets, joint_indices=self.tilt_dof_indices
        )

        return self.throttle.sum(-1)

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        env_ids = super()._reset_idx(env_ids, train)
        self._view.set_joint_positions(
            self.init_joint_positions[env_ids],
            env_indices=env_ids,
        )
        self._view.set_joint_velocities(
            self.init_joint_velocities[env_ids],
            env_indices=env_ids,
        )
        return env_ids

    def get_state(self, env=True):
        state = super().get_state(env)
        joint_positions = self.get_joint_positions()[..., self.tilt_dof_indices]
        joint_velocities = self.get_joint_velocities()[..., self.tilt_dof_indices]
        state = torch.cat([state, joint_positions, joint_velocities], dim=-1)
        assert not torch.isnan(state).any()
        return state

