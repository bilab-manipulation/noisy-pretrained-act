export MUJOCO_GL=egl

# pre-training & training
## insertion
### baseline: 사전학습 없음
#### set wandb project name as np4a in imitate_episodes.py
#CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
#CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 1
#CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 2

### [노이즈 행동 + 이미지] 사전학습
#### set wandb project name as np4a_noisyact in imitate_episodes.py
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyact.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_noisy_scripted_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyact.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_noisyact_post_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_noisy_scripted_0/policy_last.ckpt

### [노이즈 행동 + 노이즈 이미지] 사전학습
#### set wandb project name as np4a_noisyactimg in imitate_episodes.py
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyactimg.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_noisy_scripted_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyactimg.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_noisyactimg_post_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_noisy_scripted_0_noisy_img/policy_last.ckpt

### [정상 행동 + 노이즈 이미지] 사전 학습
#### set wandb project name as np4a_noisyimg in imitate_episodes.py
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyimg.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
CUDA_VISIBLE_DEVICES=1 python imitate_episodes_noisyimg.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_noisyimg_post_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/ckpt/sim_insertion_human_0_noisy_img/policy_last.ckpt