from functorch import vmap

import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

import omni.isaac.core.objects as objects
import omni_drones.utils.kit as kit_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.debug_draw import _debug_draw

from omni_drones.utils.torch import euler_to_quaternion, normalize
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from ..utils import lemniscate, scale_time
from .utils import create_pendulum


class InvPendulumTrack(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reset_thres = self.cfg.task.reset_thres
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.future_traj_len = int(self.cfg.task.future_traj_len)
        self.bar_length = self.cfg.task.bar_length
        assert self.future_traj_len > 0

        self.drone.initialize()

        # create and initialize additional views
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
        )
        self.bar.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_len-1) + 9
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(obs_dim).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )

        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0[0], device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0[0], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.8, 1.8, 1.], device=self.device),
            torch.tensor([3.2, 3.2, 1.5], device=self.device)
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.7, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 3.], device=self.device)
        self.traj_t0 = torch.pi / 2

        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.alpha = 0.8
        stats_spec = CompositeSpec({
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "motion_smoothness": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        create_pendulum(f"/World/envs/env_0/{self.drone.name}_0", self.cfg.task.bar_length, 0.04)

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w
        
        t0 = torch.zeros(len(env_ids), device=self.device)
        drone_pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        drone_pos[..., 2] -= self.bar_length
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids], drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        self.info["payload_mass"][env_ids] = payload_mass

        self.stats[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_state[..., :13]
        payload_pos = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()

        # relative position and heading
        target_pos = self._compute_traj(self.future_traj_len, step_size=5)
        self.drone_payload_rpos = self.drone_state[..., :3] - payload_pos.unsqueeze(1)
        self.target_payload_rpos = target_pos - payload_pos.unsqueeze(1)

        obs = [
            self.drone_payload_rpos, # 3
            self.drone_state[..., 3:],
            self.target_payload_rpos.flatten(1).unsqueeze(1), # self.future_traj_len * 3
            self.payload_vels.unsqueeze(1), # 6
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        smoothness = (
            self.drone.get_linear_smoothness()
            + self.drone.get_angular_smoothness()
        )
        self.stats["motion_smoothness"].lerp_(smoothness, (1-self.alpha))

        return TensorDict({
            "drone.obs": obs,
            "stats": self.stats,
            "info": self.info,
        }, self.batch_size)

    def _compute_reward_and_done(self):        

        distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))

        reward_pos = torch.exp(-self.reward_distance_scale * distance)

        reward_bar_up = normalize(-self.drone_payload_rpos)[..., 2]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)
        
        reward = (
            reward_pos
            + reward_effort
            + reward_action_smoothness
        ).unsqueeze(-1)
        
        done_misbehave = (self.drone.pos[..., 2] < 0.2) | (reward_bar_up < 0.2)
        done_hasnan = torch.isnan(self.drone_state).any(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | done_misbehave
            | done_hasnan
            | (distance > self.reset_thres)
        )

        self._tensordict["return"] += reward

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )

        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos

