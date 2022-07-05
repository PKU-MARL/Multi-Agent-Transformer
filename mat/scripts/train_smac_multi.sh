#!/bin/sh
env="StarCraft2_multi"
train_maps="3s_vs_3z 3s_vs_4z 3m MMM 3s5z 8m_vs_9m 25m 10m_vs_11m 2s3z"
eval_maps="3s_vs_3z 3s_vs_4z 3m MMM 3s5z 8m_vs_9m 25m 10m_vs_11m 2s3z"
algo="mat"
exp="multi_task"
seed=1

echo "env is ${env}, train_maps is ${train_maps}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_smac_multi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --train_maps ${train_maps} --eval_maps ${eval_maps} --seed ${seed} --n_training_threads 16 --n_eval_rollout_threads 36 --n_rollout_threads 36 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 5e-4 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval
