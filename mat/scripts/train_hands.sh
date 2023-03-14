#!/bin/sh
env="hands"
task="ShadowHandCatchOver2Underarm"
#ShadowHandDoorCloseOutward
#ShadowHandDoorOpenInward
#ShadowHandCatchOver2Underarm
algo="mat"
exp="single"
seed=1

echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_hands.py --env_name ${env} --seed ${seed} --algorithm_name ${algo} --experiment_name ${exp} --task ${task} --n_rollout_threads 80 --lr 5e-5 --entropy_coef 0.001 --max_grad_norm 0.5 --eval_episodes 5 --log_interval 25 --n_training_threads 16 --num_mini_batch 1 --num_env_steps 50000000 --gamma 0.96 --ppo_epoch 5 --clip_param 0.2 --use_value_active_masks --add_center_xy --use_state_agent --use_policy_active_masks
