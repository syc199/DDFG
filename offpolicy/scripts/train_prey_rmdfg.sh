#!/bin/sh
env="Predator_prey"
num_hare=0
num_agents=9  #9
num_stags=6 #6
num_factor=6  #18
algo="rddfg_cent_rw"
exp="debug"
name=""
seed_max=126
seed_min=126

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --num_factor ${num_factor} --seed ${seed} --num_hare ${num_hare} --num_stags ${num_stags} --use_feature_normalization --episode_length 200 --use_soft_update  --hard_update_interval_episode 200 --num_env_steps 2000000 --n_training_threads 2  --cuda  --lamda 0 --msg_iterations 4  --use_dyn_graph --adj_output_dim 32  --eval_interval 2000 --num_eval_episodes 10 --adj_lambda 1.0 --buffer_size 5000 --miscapture_punishment -1.5 --reward_time 0 --highest_orders 3 --batch_size 8 --lr 1e-3 --adj_lr 1e-6 --train_adj_episode 8 --adj_buffer_size 8 --epsilon_anneal_time 50000 --gamma 0.98 --use_wandb --use_linear_lr_decay --entropy_coef 0.01 --capture_freezes --adj_anneal_time 400000 --use_reward_normalization --use_vfunction 

done
