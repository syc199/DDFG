#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="rddfg_cent_rw"
exp="debug"
name=""
seed_max=88
seed_min=88

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 python train/train_smac.py --env_name ${env} \
     --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
      --seed ${seed} --n_training_threads 4 --buffer_size 5000 --batch_size 8 --use_soft_update \
       --hard_update_interval_episode 200 --num_env_steps 2000000 \
       --log_interval 3000 --eval_interval 20000 --user_name ${name}\
      --lamda 0 --msg_iterations 4 --use_dyn_graph --adj_output_dim 64 \
     --highest_orders 2 --lr 1e-3 --adj_lr 1e-6 --train_adj_episode 8 --entropy_coef 0.01 --capture_freezes \
     --adj_buffer_size 8 --use_wandb --use_linear_lr_decay --adj_anneal_time 400000 --use_vfunction --use_global_all_local_state
    echo "training is done!"
done


