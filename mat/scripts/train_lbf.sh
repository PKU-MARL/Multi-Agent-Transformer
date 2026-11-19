#!/bin/sh
env="LBF"
scenario="Foraging-8x8-2p-2f-coop-v3"
# Foraging-8x8-2p-2f-coop-v3
# Foraging-10x10-3p-3f-v3
# Foraging-15x15-4p-5f-v3
algo="mat"
exp="single"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_lbf.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 50 --num_env_steps 2000000 --lr 5e-4 --entropy_coef 0.01 --ppo_epoch 10 --clip_param 0.05 --use_eval --n_eval_rollout_threads 8 --eval_episodes 32 --eval_interval 25 --log_interval 25
