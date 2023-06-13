from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni_drones.utils.torch import euler_to_quaternion
import torch
import torch.distributions as D
from omni.isaac.core.objects import DynamicCuboid
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.drone import MultirotorBase

from .utils import TransportationGroup, TransportationCfg


class TransportHover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_spin_weight = self.cfg.task.reward_spin_weight
        self.reward_swing_weight = self.cfg.task.reward_swing_weight
        self.reward_action_smoothness_weight = self.cfg.task.reward_action_smoothness_weight
        self.reward_motion_smoothness_weight = self.cfg.task.reward_motion_smoothness_weight
        self.reward_potential_weight = self.cfg.task.reward_potential_weight
        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.time_encoding = self.cfg.task.time_encoding

        self.safe_distance = self.cfg.task.safe_distance

        self.group.initialize()
        self.payload = self.group.payload_view
        
        self.payload_target_visual = RigidPrimView(
            "/World/envs/.*/payloadTargetVis",
            reset_xform_properties=False
        )
        self.payload_target_visual.initialize()
        
        self.init_poses = self.group.get_world_poses(clone=True)
        self.init_velocities = torch.zeros_like(self.group.get_velocities())
        self.init_joint_pos = self.group.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.group.get_joint_velocities())

        self.init_drone_poses = self.drone.get_world_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        drone_state_dim = self.drone.state_spec.shape[-1]
        payload_state_dim = 22
        if self.time_encoding:
            self.time_encoding_dim = 4
            payload_state_dim += self.time_encoding_dim
        
        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, drone_state_dim)).to(self.device),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)).to(self.device),
            "state_payload": UnboundedContinuousTensorSpec((1, payload_state_dim)).to(self.device)
        })

        state_spec = CompositeSpec(
            drones=UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)).to(self.device),
            payload=UnboundedContinuousTensorSpec((1, payload_state_dim)).to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec=state_spec
        )

        self.payload_target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0.sum(), device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0.sum(), device=self.device)
        )
        self.init_pos_dist = D.Uniform(
            torch.tensor([-3, -3, 1.], device=self.device),
            torch.tensor([3., 3., 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.payload_target_pos = torch.tensor([0., 0., 2.], device=self.device)
        self.payload_target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)

        self.alpha = 0.8
        
        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        stats_spec = CompositeSpec({
            "payload_pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "motion_smoothness_payload": UnboundedContinuousTensorSpec(1),
            "motion_smoothness_drone": UnboundedContinuousTensorSpec(self.drone.n),
            "action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        group_cfg = TransportationCfg(num_drones=self.cfg.task.num_drones)
        self.group = TransportationGroup(drone=self.drone, cfg=group_cfg)

        scene_utils.design_scene()

        DynamicCuboid(
            "/World/envs/env_0/payloadTargetVis",
            translation=torch.tensor([0., 0., 2.]),
            scale=torch.tensor([0.75, 0.5, 0.2]),
            color=torch.tensor([0.8, 0.1, 0.1]),
            size=2.01,
        )
        kit_utils.set_collision_properties(
            "/World/envs/env_0/payloadTargetVis",
            collision_enabled=False
        )
        kit_utils.set_rigid_body_properties(
            "/World/envs/env_0/payloadTargetVis",
            disable_gravity=True
        )

        self.group.spawn(translations=[(0, 0, 2.0)], enable_collision=False)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos = self.init_pos_dist.sample(env_ids.shape)
        rpy = self.init_rpy_dist.sample(env_ids.shape)
        rot = euler_to_quaternion(rpy)
        heading = torch_utils.quat_axis(rot, 0)
        # up = torch_utils.quat_axis(rot, 2)

        self.group._reset_idx(env_ids)
        self.group.set_world_poses(pos + self.envs_positions[env_ids], rot, env_ids)
        self.group.set_velocities(self.init_velocities[env_ids], env_ids)

        self.group.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.group.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        payload_target_rpy = self.payload_target_rpy_dist.sample(env_ids.shape)
        payload_target_rot = euler_to_quaternion(payload_target_rpy)
        payload_target_heading = torch_utils.quat_axis(payload_target_rot, 0)
        payload_masses = self.payload_mass_dist.sample(env_ids.shape)

        self.payload_target_heading[env_ids] = payload_target_heading

        self.payload.set_masses(payload_masses, env_ids)
        self.payload_target_visual.set_world_poses(
            orientations=payload_target_rot,
            env_indices=env_ids
        )

        self.info["payload_mass"][env_ids] = payload_masses.unsqueeze(-1).clone()
        self.stats[env_ids] = 0.
        distance = torch.cat([
            self.payload_target_pos-pos,
            payload_target_heading-heading,
        ], dim=-1).norm(dim=-1, keepdim=True)
        self.last_distance[env_ids] = distance

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]

        self.payload_pos, self.payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_vels = self.payload.get_velocities()
        self.payload_heading: torch.Tensor = torch_utils.quat_axis(self.payload_rot, axis=0)
        self.payload_up: torch.Tensor = torch_utils.quat_axis(self.payload_rot, axis=2)
        
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        self.drone_pdist = torch.norm(self.drone_rpos, dim=-1, keepdim=True)
        payload_drone_rpos = self.payload_pos.unsqueeze(1) - drone_pos

        payload_target_rpos = self.payload_target_pos - self.payload_pos
        self.target_payload_rpose = torch.cat([
            payload_target_rpos,
            self.payload_target_heading - self.payload_heading
        ], dim=-1)
        
        self.group.get_state()
        payload_state = [
            self.target_payload_rpose,
            self.payload_rot,  # 4
            payload_vels,  # 6
            self.payload_heading,  # 3
            self.payload_up, # 3
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            payload_state.append(t.expand(-1, self.time_encoding_dim))
        payload_state = torch.cat(payload_state, dim=-1).unsqueeze(1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-payload_drone_rpos, self.drone_states[..., 3:]], dim=-1
        ).unsqueeze(2) # [..., 1, state_dim]
        obs["state_others"] = torch.cat(
            [self.drone_rpos, self.drone_pdist, vmap(others)(self.drone_states[..., 3:13])], dim=-1
        ) # [..., n-1, state_dim + 1]
        obs["state_payload"] = payload_state.expand(-1, self.drone.n, -1).unsqueeze(2) # [..., 1, 22]

        state = TensorDict({}, self.num_envs)
        state["payload"] = payload_state # [..., 1, 22]
        state["drones"] = obs["state_self"].squeeze(2) # [..., n, state_dim]

        pos_error = torch.norm(payload_target_rpos, dim=-1, keepdim=True)
        heading_alignment = torch.sum(
            self.payload_heading * self.payload_target_heading, dim=-1, keepdim=True
        )
        self.stats["payload_pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.payload_up[:, 2].unsqueeze(-1), (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.motion_smoothness_payload = (
            self.group.get_linear_smoothness()
            + self.group.get_angular_smoothness()
        )
        self.stats["motion_smoothness_payload"].lerp_(self.motion_smoothness_payload, (1-self.alpha))
        self.motion_smoothness_drone = (
            self.drone.get_linear_smoothness()
            + self.drone.get_angular_smoothness()
        )
        self.stats["motion_smoothness_drone"].lerp_(self.motion_smoothness_drone, (1-self.alpha))

        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
            "info": self.info,
            "stats": self.stats
        }, self.num_envs)

    def _compute_reward_and_done(self):
        vels = self.payload.get_velocities()
        joint_positions = (
            self.group.get_joint_positions()[..., :16]
            / self.group.joint_limits[..., :16, 0].abs()
        )
        
        distance = torch.norm(self.target_payload_rpose, dim=-1, keepdim=True)
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(-distance * self.reward_distance_scale)

        up = self.payload_up[:, 2]
        reward_up = torch.square((up + 1) / 2).unsqueeze(-1)

        spinnage = vels[:, -3:].abs().sum(-1, keepdim=True)
        reward_spin = self.reward_spin_weight * torch.exp(-torch.square(spinnage))

        swing = vels[:, :3].abs().sum(-1, keepdim=True)
        reward_swing = self.reward_swing_weight * torch.exp(-torch.square(swing))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward_joint_limit = 0.5 * torch.mean(1 - torch.square(joint_positions), dim=-1)

        reward_action_smoothness = self.reward_action_smoothness_weight * -self.drone.throttle_difference
        reward_motion_smoothness = (self.reward_motion_smoothness_weight * self.motion_smoothness_drone / 1000)

        reward_potential = self.reward_potential_weight * (self.last_distance - distance)
        self.last_distance[:] = distance
        
        reward[:] = (
            reward_separation * (
                reward_pose 
                + reward_pose * (reward_up + reward_spin + reward_swing) 
                + reward_potential
                + reward_joint_limit
                + reward_action_smoothness.mean(1, True)
                + reward_motion_smoothness.mean(1, True)
                + reward_effort
            )
        ).unsqueeze(-1)

        done_hasnan = torch.isnan(self.drone_states).any(-1)
        done_fall = self.drone_states[..., 2] < 0.2

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1) 
            | done_fall.any(-1, keepdim=True)
            | done_hasnan.any(-1, keepdim=True)
        )

        self._tensordict["return"] += reward
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

