#!/bin/bash
env="StarCraft2_multi"
maps=("27m_vs_30m" "8m" "5m_vs_6m" "MMM2" "1c3s5z" "2s_vs_1sc")
algo="mat"
exp="from_scratch_"
seed=1

for map in "${maps[@]}"
do
  exp_map=$exp$map
  echo "env is ${env}, train_maps is ${map}, algo is ${algo}, exp is ${exp_map}, seed is ${seed}"
  CUDA_VISIBLE_DEVICES=0 python train/train_smac_multi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp_map} --train_maps ${map} --eval_maps ${map} --seed ${seed} --n_training_threads 16 --n_eval_rollout_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --eval_interval 5 --num_env_steps 1000000 --lr 5e-4 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval
done
