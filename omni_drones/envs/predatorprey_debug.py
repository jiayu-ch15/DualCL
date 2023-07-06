import torch
import numpy as np
import functorch
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time

import omni.isaac.core.objects as objects
# from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, VisualCylinder, FixedCylinder, DynamicCylinder
# from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
from omni_drones.views import RigidPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
# import omni_drones.utils.restart_sampling as rsp
from pxr import UsdGeom, Usd, UsdPhysics

# drones on land by default
# only cubes are available as walls
# clip state as walls

class PredatorPrey_debug(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        self.obstacle = []
        for i in range(self.num_obstacles):
            # self.obstacle.append(GeometryPrimView(prim_paths_expr="/World/envs/env_*/obstacle_{idx}".format(idx=i)))
            self.obstacle.append(RigidPrimView(prim_paths_expr="/World/envs/env_*/obstacle_{idx}".format(idx=i)))
        self.target = RigidPrimView("/World/envs/env_*/target", reset_xform_properties=False)
        self.target.initialize()

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0,cfg.env.num_envs))
        self.env_width = cfg.env.env_spacing/2.0
        self.radius = self.env_width
        self.progress_buf2 = self.progress_buf*1.0
        self.caught = self.progress_buf2*0
        self.returns = self.progress_buf2*0
        self.init_poses = self.drone.get_world_poses(clone=True)

        # CL
        # self.goals = self.create_goalproposal_mix()
       
        self.random_idx = torch.ones(self.num_envs, device=self.device)
        
        
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.num_agents, 
            UnboundedContinuousTensorSpec(13 + 6*self.num_agents + 3*self.num_obstacles + 1, device=self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )        
        
        self.task_config = TensorDict({
            "max_linvel": torch.ones(self.num_envs, 1, device=self.device)
        }, self.num_envs)

        self.vels = self.drone.get_velocities()
        self.init_pos_scale = torch.tensor([2., 2., 0], device=self.device) 
        self.init_pos_offset = torch.tensor([-1., -1., 0], device=self.device)
        

    def _design_scene(self):
        self.num_agents = self.cfg.num_agents
        self.num_obstacles = self.cfg.num_obstacles
        self.size_obstacle = self.cfg.size_obstacle

        self.v_low = self.cfg.v_drone*self.cfg.v_low
        self.v_high = self.cfg.v_drone*self.cfg.v_high
        self.v0 = torch.from_numpy(np.random.uniform(self.v_low, self.v_high, [self.num_envs, 1])).to(self.device)
        self.list_v0 = np.array(self.cfg.v_list)
        # cfg = RobotCfg()
        # cfg.rigid_props.max_linear_velocity = self.cfg.v_drone
        # self.drone: MultirotorBase = MultirotorBase.REGISTRY["Crazyflie"](cfg=cfg)

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        self.target_pos = torch.tensor([[0., 0.05, 0.5]], device=self.device)
        self.coll_times = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.success = 0
        self.mean_vel = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.collided = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.target_args0 = torch.zeros(self.num_envs, device=self.device)
        self.target_args_ = torch.zeros(self.num_envs, device=self.device)
        self.outside = torch.zeros(self.num_envs, device=self.device)
        self.xx = 0
        self.cl_level = self.cfg.vel0
        self.progress_std = 0
        self.sample_buffer = []
        self.caught_buffer = []

        self.test_count = 0

        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            # height=0.1,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )
    
        objects.VisualCylinder(
            prim_path="/World/envs/env_0/ground",
            name="ground",
            translation= torch.tensor([0., 0., 0.], device=self.device),
            radius=self.cfg.env.env_spacing/2.0,
            height=0.001,
            color=torch.tensor([0., 0., 0.]),
        )

        
        for i in range(self.num_obstacles):
            objects.DynamicCylinder(
                prim_path="/World/envs/env_0/obstacle_{idx}".format(idx=i),
                name="obstacle",
                color=torch.tensor([0., 0.1, 0.]),
                radius = self.size_obstacle,
                height = 2,
                visible = 1, # air wall
            )
            prim = objects.DynamicCylinder.prim
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            prim.GetAttribute("physics:kinematicEnabled").Set(True)

        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            disable_gravity=True
        )        

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        n = self.num_agents
        for i in range(n):
            translation = torch.zeros(n, 3)
            translation[:, 0] = i 
            translation[:, 1] = torch.arange(n)
            translation[:, 2] = 0.5
        self.drone.spawn(translation) #to make n drones
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        n = self.num_agents
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)

        # if self.cfg.test_list:
        #     v = self.list_v0[self.test_count%len(self.list_v0)]
        #     print("{vel}: {rate}".format(vel=self.list_v0[(self.test_count-1)%len(self.list_v0)], rate=self.success))
        #     self.v0 = torch.ones(self.num_envs, 1, device=self.device) * v * self.cfg.v_drone
        # elif self.cfg.fixed_vel:
        #     if self.success > 90:
        #         self.cl_level += 0.01
        #     self.v0 = torch.ones(self.num_envs, 1, device=self.device) * self.cfg.v_drone * self.cl_level
        # elif self.cfg.mixed:
        #     ran = np.arange(self.cfg.v_low, self.cfg.v_high+0.001, 0.05)
        #     self.class_success(self.v0.squeeze(-1)/self.cfg.v_drone, self.caught, ran, near=0.025)
        #     # if self.success > 90:
        #     #     self.cl_level += 0.01
        #     self.v0 = torch.from_numpy(np.random.uniform(self.v_low, self.v_high, [self.num_envs, 1])).to(self.device)
        #     # self.v0 = torch.from_numpy(np.random.uniform(self.cfg.vel0, self.cl_level, [self.num_envs, 1])).to(self.device)*self.cfg.v_drone 
        # else: # CL
        #     # 只从30%里采下一波70%的
        #     if self.test_count==0:
        #         self.v0 = torch.from_numpy(np.random.uniform(self.v_low, self.v_high, [self.num_envs, 1])).to(self.device)
        #     else:
        #         sample = self.v0[int(0.7*self.num_envs):]
        #         caught = self.caught[int(0.7*self.num_envs):]
        #         if len(self.sample_buffer) == 3:
        #             self.sample_buffer.pop(0)
        #             self.caught_buffer.pop(0)
        #         self.sample_buffer.append(sample.squeeze(-1)*1.0)
        #         self.caught_buffer.append(caught*1.0)
        #         # buffer攒3个episode再更新
        #         sample_ = torch.cat(self.sample_buffer, dim=-1)
        #         caught_ = torch.cat(self.caught_buffer, dim=-1)
        #         ran = np.arange(self.cfg.v_low, self.cfg.v_high+0.001, 0.05)
        #         score = self.class_success(sample_/self.cfg.v_drone, caught_, ran, near=0.025)                
        #         self.goals.update_buffer(sample_.cpu().tolist(), score.cpu().tolist())
        #         starts_all, _ = self.goals.restart_sampling()
        #         self.random_idx = torch.ones(self.num_envs, device=self.device) 
        #         for i,v in enumerate(starts_all):
        #             if v is None:
        #                 starts_all[i] = np.random.uniform(self.v_low, self.v_high)                 
        #             else:
        #                 self.random_idx[i] = 0
        #         starts_all = np.array(starts_all)
        #         self.v0 = torch.from_numpy(starts_all).to(self.device).unsqueeze(-1)
                
    
        pos = self.random_polar([self.num_envs, n, 3], [0, 1.0*self.radius])
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

        self.coll_times[env_ids] *= 0
        self.mean_vel[env_ids] *= 0
        self.collided[env_ids] *= 0

        #obstalces
        self.pos_obstacle = self.random_polar([self.num_envs, self.num_obstacles, 3], [0, self.radius])
        if self.num_obstacles > 0 and self.test_count % 1 ==0:
            for i in range(self.num_obstacles):
                self.obstacle[i].set_world_poses((self.envs_positions + self.pos_obstacle[:, i, :])[env_ids], indices=env_ids)
                # aaa = self.obstacle[i].get_net_contact_forces(indices=[0,1])
            pass

        self.target_pos = self.random_polar([self.num_envs, 3], [self.radius*0, self.radius*1.0])
        self.target_pos[..., 2] = 0.5
        self.target.set_world_poses((self.envs_positions + self.target_pos)[env_ids], env_indices=env_ids)

        target_vel = self.target.get_velocities()
        self.target.set_velocities(2*torch.rand_like(target_vel)-1, self.env_ids)

        self.test_count += 1

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)
        
        # alpha <= 1.0
        alpha = 1.0

        # disable some drones
        # actions[:, 1, :] *= 0
        # actions[:, 1, :] -= 3
        
        forces_target = self._get_dummy_policy_prey(norm=0)
        target_vel = self.target.get_velocities()
        vel = target_vel[..., :3] * 1.0
        target_vel[..., :3] = vel + 0.016 * forces_target * alpha
        target_vel[..., 2] = 0
        vel_norm = torch.norm(target_vel, dim=-1).unsqueeze(-1).expand_as(target_vel) + 1e-5
        target_vel = target_vel / vel_norm * torch.clamp(vel_norm, 0*vel_norm, self.v0.expand_as(target_vel))
        
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)

    def _compute_state_and_obs(self):
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        drone_state = self.drone.get_state()[..., :13]
        drone_state0 = drone_state*1.0
        drone_pos = drone_state0[..., :3]
        for i in range(self.num_agents - 1):
            drone_state = torch.cat((drone_state, drone_state0[..., (np.arange(0,self.num_agents) + i + 1)%self.num_agents, :3],
                                    drone_state0[..., (np.arange(0,self.num_agents) + i + 1)%self.num_agents, 7:10]
            ), -1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        prey_vel = self.target.get_velocities()[..., :3].unsqueeze(1).expand(-1,self.num_agents,-1)
        obstacle_pos = self.pos_obstacle.unsqueeze(1).expand(-1,self.num_agents,-1,-1) - drone_pos.unsqueeze(-2).expand(-1,-1,self.num_obstacles,-1)
        obs = torch.cat((drone_state, prey_pos, prey_vel, obstacle_pos.reshape(self.num_envs, self.num_agents, -1), \
            self.progress_buf2.unsqueeze(-1).unsqueeze(-1).expand(-1,self.num_agents, -1)/self.cfg.env.max_episode_length),-1)

        return TensorDict({
            "drone.obs": obs
        }, self.batch_size)

    # vel_obs
    # reward_average
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_world_poses(False)
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)

        target_dist = torch.norm(pos-prey_pos, dim=-1)

        catch_reward = 1.0 * (target_dist < 0.12) #sparse
        catch_reward = 1.0 * catch_reward.sum(-1).unsqueeze(-1).expand_as(catch_reward)

        # collison with obstacles
        coll_reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        
        for i in range(self.num_obstacles):
            relative_pos = pos[..., :2] - self.pos_obstacle[:, i, :2].unsqueeze(-2).expand_as(pos[..., :2])
            norm_r = torch.norm(relative_pos, dim=-1)
            if_coll = 1.0*(norm_r < 0.2)
            self.coll_times += if_coll
            coll_reward -= if_coll # sparse
            self.collided = 1.0 * ((self.collided + if_coll) > 0)

        # distance-based reward
        min_dist = (torch.min(target_dist, dim=-1)[0].unsqueeze(-1).expand_as(target_dist))
        distance_reward = - 1.0 * min_dist

        reward = 1.0*catch_reward + 5.0*coll_reward + distance_reward
        
        self._tensordict["return"] += reward.unsqueeze(-1)
        self.returns = self._tensordict["return"].squeeze(-1).sum(-1)*1.0

        self.progress_buf2 = self.progress_buf2 * (self.progress_buf > 0) + 1
        
        done  = (self.progress_buf2 >= self.max_episode_length).unsqueeze(-1) \
            | ((self.caught > 0).unsqueeze(-1) & (self.progress_buf2.unsqueeze(-1) > 1))*self.cfg.reset0
        
        
        caught = (catch_reward > 0) * 1.0
        self.caught = (self.progress_buf > 0) * ((self.caught + caught.any(-1)) > 0)
        self.progress_std = torch.std(self.progress_buf)

        self.random_success = torch.sum(self.random_idx*self.caught)/(torch.sum(self.random_idx)+1e-7)*100
        self.train_success = torch.sum(self.caught*(1-self.random_idx))/(torch.sum(1-self.random_idx)+1e-7)*100
        self.success = torch.sum(self.caught)/self.num_envs*100
        # num_level = len(self.goals.buffer)*torch.ones(self.num_envs, device=self.device)

        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done.any(-1),
            "caught": self.caught * 1.0,
            "return": self._tensordict["return"],
            "success": self.success*torch.ones(self.num_envs, device=self.device),
            "random_success": self.random_success*torch.ones(self.num_envs, device=self.device),
            "train_success": self.train_success*torch.ones(self.num_envs, device=self.device),
            # "mean_level": self.goals.mean_level/self.cfg.v_drone*torch.ones(self.num_envs, device=self.device),
            # "num_level": num_level,
            # "max_score": self.goals.max_score*torch.ones(self.num_envs, device=self.device),
            # "min_score": self.goals.min_score*torch.ones(self.num_envs, device=self.device),
            # "buffer_max": self.goals.buffer_max*torch.ones(self.num_envs, device=self.device),
            # "buffer_min": self.goals.buffer_min*torch.ones(self.num_envs, device=self.device),
            # "keep": self.goals.keep*torch.ones(self.num_envs, device=self.device),
            "collision_times": self.coll_times.sum(-1), #就是coll_reward
            "cl_level": self.cl_level*torch.ones(self.num_envs, device=self.device),
            "progress_std": self.progress_std**torch.ones(self.num_envs, device=self.device)
        }, self.batch_size)

    def _get_dummy_policy_prey(self, norm=1):

        pos, rot = self.drone.get_world_poses(False)
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        force = torch.zeros(self.num_envs, 3, device=self.device)

        # predators
        dist_pos = torch.norm(prey_pos - pos,dim=-1).unsqueeze(-1).expand(-1,-1,3)
        orient = self.normalize(prey_pos - pos)
        force = (orient/dist_pos).sum(-2)

        # arena
        norm_r = (torch.norm(prey_state[..., :2], dim=-1)+1e-5).unsqueeze(-1).expand(-1, 2)
        force_r = - (prey_state[..., :2]  / norm_r)* 1/(torch.relu(self.radius - norm_r) + 1e-20) * (norm_r > 0.1) # relu for constraint     
        force[..., :2] += force_r

        # obstacles        
        for i in range(self.num_obstacles):
            relative_pos = prey_state[..., :2] - self.pos_obstacle[:, i, :2]
            norm_r = (torch.norm(relative_pos, dim=-1)+1e-7).unsqueeze(-1).expand(-1, 2)
            force[..., :2] += 1.0*relative_pos  / norm_r * 1/(torch.abs(norm_r - self.size_obstacle) + 1e-7) # 1/10

        force[..., 2] = 0
        return force.type(torch.float32)
    
    # def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     self._pre_sim_step(tensordict)
    #     for _ in range(1):
    #         self.sim.step(self.enable_render)
    #     tensordict = TensorDict({"next": {}}, self.batch_size)
    #     tensordict["next"].update(self._compute_state_and_obs())
    #     tensordict["next"].update(self._compute_reward_and_done())
    #     # done决定重置，用buf2解决；buf代表第一次抓到的步数
    #     self.progress_buf += (1 - tensordict['next']['caught'])*1
    #     return tensordict

    def random_polar(self, size, radius):
        assert size[-1] == 3 and len(radius)==2
        orient = torch.rand(size, device=self.device) + torch.tensor([-0.5, -0.5, 0.], device=self.device)
        orient[..., 2] = 0.
        orient /= 1e-7 + torch.norm(orient, dim = -1).unsqueeze(-1).expand_as(orient)
        pos = orient * torch.from_numpy(np.random.uniform(radius[0], radius[1], size[:-1]+[1])).to(self.device).expand_as(orient)
        return pos.type(torch.float32)
    
    def get_orient(self, x):
        y = torch.norm(x, dim=-1).unsqueeze(-1).expand_as(x) 
        
    def normalize(self, x):
        y = x/(torch.norm(x, dim=-1).unsqueeze(-1).expand_as(x)+1e-12)
        return y

    # def create_goalproposal_mix(self):
    #     buffer_length = self.num_envs*50 #for debugging
    #     device = self.device
    #     proposal_batch = self.num_envs  #right
    #     goals = rsp.goal_proposal_return(
    #         config=None, env_name=None, scenario_name=None, critic_k=1, 
    #         buffer_capacity=buffer_length, proposal_batch=proposal_batch, device=device, score_type='value_error')
    #     return goals

    def plot_buffer(self, step=0.1):
        buffer = np.array(self.goals.buffer)
        if len(buffer) == 0:
            buffer = np.array([1.3])
        else:
            buffer /= self.cfg.v_drone
        level = np.linspace(1.3, 5.0, 38)
        times = [(abs(buffer - v) < 0.5*step).sum() for v in level]
        plt.figure()
        plt.plot(level, times)
        plt.savefig('buffer_num.png')
        # if 1:
        #     x = np.array(self.goals.buffer0)[...,0]
        #     y = np.array(self.goals.buffer_priority0)
        #     plt.figure()
        #     plt.plot(x, y, 'o')
        #     slope, intercept = np.polyfit(x, y, 1)
        #     y = slope*x + intercept
        #     figure2, = plt.plot(x, y, color='r')
        #     plt.savefig('buffer_reward.png')
        return {"buffer": wandb.Plotly(plt.gcf())}
    
    def print_success(self, x, suc, ran):
        assert x.size()[0]==suc.size()[0]
        result = []
        suc = (suc > 0)*1.0
        for v in ran:
            mask = torch.abs(x-v)<0.05
            sv = torch.sum(suc * mask.squeeze(-1)) / torch.sum(mask)*100
            result.append(sv)
            print(v, ': ', sv)
        return result
    
    def class_success2(self, x, suc, ran, near):
        assert x.size()==suc.size()
        L = x.size()[0]
        suc = (suc > 0)*1.0
        y = suc*0
        rate = []
        x1 = x.unsqueeze(0).expand(L, -1)
        x2 = x.unsqueeze(-1).expand(-1, L)
        x_in = torch.abs(x1 - x2) < near
        suc1 = suc.unsqueeze(0).expand(L, -1)
        suc_in = suc1 * x_in
        y = torch.sum(suc_in, -1)/(torch.sum(x_in, -1) + 1e-9)
        for v in ran:
            mask = torch.abs(x-v) < near
            sv = torch.sum(suc * mask.squeeze(-1)) / (torch.sum(mask) + 1e-9) * 100
            rate.append(sv.cpu())
        plt.figure()
        plt.plot(ran, rate)
        plt.savefig('buffer.png')
        plt.figure()
        plt.scatter(x.cpu().numpy(), y.cpu().numpy())
        plt.savefig('rate.png')
        return y
    
    def class_success(self, x, suc, ran, near):
        assert x.size()==suc.size()
        suc = (suc > 0)*1.0
        y = suc*0
        rate = []
        # 离散化
        for v in ran:
            mask = torch.abs(x-v) < near
            sv = torch.sum(suc * mask.squeeze(-1)) / (torch.sum(mask) + 1e-9)
            y += mask * sv
            rate.append(sv.cpu())
        plt.figure()
        plt.plot(ran, rate)
        plt.savefig('buffer.png')
        return y