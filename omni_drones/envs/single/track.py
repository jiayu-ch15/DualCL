from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, scale_time

class Track(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reset_thres = self.cfg.task.reset_thres
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = self.cfg.task.reward_motion_smoothness_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.future_traj_len = int(self.cfg.task.future_traj_len)
        assert self.future_traj_len > 0
        self.intrinsics = self.cfg.task.intrinsics
        self.wind = self.cfg.task.wind

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        if self.wind:
            if randomization is not None:
                if "wind" in self.cfg.task.randomization:
                    cfg = self.cfg.task.randomization["wind"]
                    # for phase in ("train", "eval"):
                    wind_intensity_scale = cfg['train'].get("intensity", None)
                    self.wind_intensity_low = wind_intensity_scale[0]
                    self.wind_intensity_high = wind_intensity_scale[1]
            else:
                self.wind_intensity_low = 0
                self.wind_intensity_high = 2
            self.wind_w = torch.zeros(self.num_envs, 3, 8, device=self.device)
            self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)


        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_len-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(obs_dim).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
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
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.1, device=self.device)
        )
        self.origin = torch.tensor([0., 0., 2.], device=self.device)

        self.traj_t0 = torch.pi / 2
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_len, 3, device=self.device)

        self.alpha = 0.8
        stats_spec = CompositeSpec({
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "motion_smoothness": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        info_spec = self.drone.info_spec.to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        # self.info = self.drone.info
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

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
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.zeros(len(env_ids), device=self.device)
        pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        self.stats[env_ids] = 0.

        self.info[env_ids] = self.drone.info[env_ids]

        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
        if self.wind:
            self.wind_i[env_ids] = torch.rand(*env_ids.shape, 1, device=self.device) * (self.wind_intensity_high-self.wind_intensity_low) + self.wind_intensity_low
            self.wind_w[env_ids] = torch.randn(*env_ids.shape, 3, 8, device=self.device)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

        if self.wind:
            t = (self.progress_buf * self.dt).reshape(-1, 1, 1)
            self.wind_force = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
            wind_forces = self.drone.MASS_0 * self.wind_force
            wind_forces = wind_forces.unsqueeze(1).expand(*self.drone.shape, 3)
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()

        self.target_pos[:] = self._compute_traj(self.future_traj_len, step_size=5)
        
        self.rpos = self.target_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.root_state[..., 3:],
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        if self.intrinsics:
            obs.append(self.drone.get_info())

        obs = torch.cat(obs, dim=-1)

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.smoothness = (
            self.drone.get_linear_smoothness() 
            + self.drone.get_angular_smoothness()
        )
        self.stats["motion_smoothness"].lerp_(self.smoothness, (1-self.alpha))

        if hasattr(self, "info"):
            return TensorDict({
                "drone.obs": obs,
                "stats": self.stats,
                "info": self.info
            }, self.batch_size)
        else:
            return TensorDict({
                "drone.obs": obs,
                "stats": self.stats
            }, self.batch_size)

    def _compute_reward_and_done(self):
        # pos reward
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))
        
        reward_pose = torch.exp(-self.reward_distance_scale * distance)
        
        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)
        reward_motion_smoothness = self.reward_motion_smoothness_weight * (self.smoothness / 1000)

        # spin reward
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        reward = (
            reward_pose 
            + reward_pose * (reward_up + reward_spin) 
            + reward_effort
            + reward_action_smoothness
            + reward_motion_smoothness
        )
        self._tensordict["return"] += reward.unsqueeze(-1)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (self.drone.pos[..., 2] < 0.1)
            | (distance > self.reset_thres)
        ) 

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
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


