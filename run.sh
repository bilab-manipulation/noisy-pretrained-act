# python record_sim_episodes.py --task_name sim_transfer_cube_noisy_scripted --num_episodes 50
# python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_human
# python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_noisy_scripted
# 
# python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_human --episode_idx all
# python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_human_noisy_img --episode_idx all
# python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_noisy_scripted --episode_idx all
# python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_noisy_scripted_noisy_img --episode_idx all
# 
CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0