set -e

export MUJOCO_GL=egl

python record_sim_episodes.py --task_name sim_insertion_noisy_scripted --num_episodes 50
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_human
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_noisy_scripted

python visualize_episodes.py --dataset_dir ./data/sim_insertion_human --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_insertion_human_noisy_img --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_insertion_noisy_scripted --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_insertion_noisy_scripted_noisy_img --episode_idx all

python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_0_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_0_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0

###

python record_sim_episodes.py --task_name sim_transfer_cube_noisy_scripted --num_episodes 50
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_human
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_noisy_scripted

python visualize_episodes.py --dataset_dir ./data/sim_transfer_cube_human --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_transfer_cube_human_noisy_img --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_transfer_cube_noisy_scripted --episode_idx all
python visualize_episodes.py --dataset_dir ./data/sim_transfer_cube_noisy_scripted_noisy_img --episode_idx all

python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0