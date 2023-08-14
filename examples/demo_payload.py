import os

import functorch
import hydra
import torch
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app

from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni.isaac.core.objects as objects
    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    drone_model = "Firefly"
    n = 4
    translations = [
        [-0.5, 0.5, 1.5],
        [0.5, 0.5, 1.5],
        [0.5, -0.5, 1.5],
        [-0.5, -0.5, 1.5],
    ]

    drone: MultirotorBase = MultirotorBase.REGISTRY[drone_model]()
    controller = drone.DEFAULT_CONTROLLER(
        dt=sim.get_physics_dt(), g=9.81, uav_params=drone.params
    ).to(sim.device)
    controller_state = TensorDict({}, n, device=sim.device)
    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0:3] = (
        torch.tensor(translations) + torch.tensor([3.0, 0.0, 0.7])
    ).to(cfg.sim.device)

    payload = objects.DynamicCuboid(
        prim_path="/World/envs/env_0/payload",
        position=(0.0, 0.0, 0.4),
        scale=(0.9, 0.9, 0.45),
        mass=3,
    )
    payload.set_collision_enabled(False)

    for i in range(n):
        unit = prim_utils.create_prim(
            f"/World/envs/env_0/Unit_{i}", translation=translations[i]
        )
        drone.spawn(
            translations=[(0.0, 0.0, 0.0)],
            prim_paths=[f"/World/envs/env_0/Unit_{i}/Firefly"],
        )
        scene_utils.create_rope(
            f"/World/envs/env_0/Unit_{i}/rope_{i}",
            from_prim=payload.prim,
            to_prim=f"/World/envs/env_0/Unit_{i}/Firefly/base_link",
        )

    scene_utils.design_scene()
    sim.reset()
    drone.initialize("/World/envs/env_0/Unit_.*/Firefly")
    payload.initialize()
    sim.pause()

    while simulation_app.is_running():
        if not sim.is_playing():
            sim.step()
            continue
        root_state = drone.get_state()[..., :13].squeeze(0)
        action, controller_state = functorch.vmap(controller)(
            root_state, control_target, controller_state
        )
        drone.apply_action(action)
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
