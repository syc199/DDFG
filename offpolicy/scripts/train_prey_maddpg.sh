#!/bin/sh
env="Predator_prey"
num_hare=0
num_agents=9
num_stags=6
algo="rmaddpg"
exp="p=-1"
seed_max=126
seed_min=126
name=""

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --num_hare ${num_hare} --num_stags ${num_stags} --seed ${seed} --episode_length 200 --actor_train_interval_step 1 --tau 0.005  --use_soft_update --lr 7e-4 --hard_update_interval_episode 200 --num_env_steps 2000000  --use_feature_normalization --miscapture_punishment -1 --eval_interval 2000 --num_eval_episodes 10 --reward_time -0.1 --use_wandb --batch_size 8 --capture_freezes
    echo "training is done!"
done
#CUDA_VISIBLE_DEVICES=5
