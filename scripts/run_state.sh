task_name="Spread"
algo_name="mappo"

CUDA_VISIBLE_DEVICES=0 python train.py headless=True \
task=$task_name algo=$algo_name \
eval_interval=80 \
wandb.run_name=${task_name}-${algo_name} \
wandb.entity="chenjy" \
task.env.num_envs=4096