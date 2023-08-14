from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni_drones.utils.kit as kit_utils

import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from .utils import attach_payload

class PayloadHover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = self.cfg.task.reward_motion_smoothness_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding
        self.bar_length = self.cfg.task.bar_length

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 9
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
            reset_xform_properties=False,
        )
        self.payload.initialize()

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(obs_dim).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -2.5, 1.5], device=self.device),
            torch.tensor([2.5, 2.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )
        # randomly push the payload by applying a force
        push_force_scale = self.cfg.task.push_force_scale
        self.push_force_dist = D.Normal(
            torch.tensor([0., 0., 0.], device=self.device),
            torch.tensor(push_force_scale, device=self.device)/self.dt
        )
        push_interval = self.cfg.task.push_interval
        self.push_prob = (1 - torch.exp(-self.dt/torch.tensor(push_interval))).to(self.device)
        
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0[0], device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0[0], device=self.device)
        )
        self.payload_masses = self.payload.get_masses()

        self.target_pos = torch.tensor([0., 0., 1.], device=self.device)

        self.alpha = 0.8

        info_spec  = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        stats_spec = CompositeSpec({
            "pos_error": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "motion_smoothness": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        info_spec.update(self.drone.info_spec.to(self.device))
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
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
        attach_payload(f"/World/envs/env_0/{self.drone.name}_0", self.cfg.task.bar_length)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        pos = self.init_pos_dist.sample(env_ids.shape)
        rpy = self.init_rpy_dist.sample(env_ids.shape)
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        self.payload_masses[env_ids] = payload_mass

        self.stats[env_ids] = 0.

        self.info.update_at_(self.drone.info[env_ids], env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)
        self._push_payload()
    
    def _push_payload(self):
        env_mask = (torch.rand(self.num_envs, device=self.device) < self.push_prob).float()
        forces = self.push_force_dist.sample((self.num_envs,))
        forces = (
            forces.clamp_max(self.push_force_dist.scale * 3)
            * self.payload_masses
            * env_mask.unsqueeze(-1)
        )
        self.payload.apply_forces(forces)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.payload_pos = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()
        
        self.drone_payload_rpos = self.drone.pos - self.payload_pos.unsqueeze(1)
        self.target_payload_rpos = (self.target_pos - self.payload_pos.unsqueeze(1))

        obs = [
            self.drone_payload_rpos.flatten(1).unsqueeze(1),
            self.target_payload_rpos.flatten(1).unsqueeze(1),
            self.root_state[..., 3:],
            self.payload_vels.unsqueeze(1), # 6
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.target_distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.smoothness = (
            self.drone.get_linear_smoothness() 
            + self.drone.get_angular_smoothness()
        )
        self.stats["motion_smoothness"].lerp_(self.smoothness, (1-self.alpha))

        return TensorDict({
            "drone.obs": obs,
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pos reward
        reward_pose = torch.exp(-self.reward_distance_scale * self.target_distance)
        
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
            | (self.drone.pos[..., 2] < 0.2)
            | (self.payload_pos[..., 2] < 0.2).unsqueeze(-1)
        ) 
        
        return TensorDict(
            {
                "reward": {"drone.reward": reward.unsqueeze(-1)},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )


