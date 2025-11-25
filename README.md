# Improving Robot Imitation Learning with Noise-Pretrained Transformers
## 2025.11.25. memo for KTCP
- KCC 제출 버전의 ckpt에 대한 분석 추가(bi-stealth 4tb hdd의 data/archive에서 찾음)
- 당시 코드는 github 250513_overflow 커밋 또는 그 이전임(5월7일 제출)
- 기존 성능은 npact_eval 파일의 시트 1: https://docs.google.com/spreadsheets/d/1JjnU7BWnV-4ooRz0hF8CGxt8fmEbikqOVXssfeUUrpA/edit?gid=283176072#gid=283176072

- 0: baseline, 1: noisy act, 2: noisy act + img, 3: noisy img
- policy last (epoch idx 1999)로 다시 evaluation: KCC버전은 policy best 기준인듯
- 분석에 사용된 ckpt는 4.7GB정도, 전체는 몇백기가 나와서 data NAS에 보관(폴더명 KCC_KTCP)
- baseline2, insertion 등은 epoch idx가 1999가 아니라 2000이 저장되어있어서 논문에 포함하지 않았음

- TODO: 학습 평균, 최종 학습 성능, various noise distribution or magnitude, 학습 전 가중치(현재는 1 epoch 후에 가중치 저장) 저장 및 비교, seed control, 훈련 시간 비교

## 2025.10.20. New version: Noisy-Pretraining for Action; NP4A
- ACT -> ACT++ : TODO
- transfer cube task -> insertion task : reward settings of transfer cube task is quite strange and - insertion task is more cleary criterion of task success.
- evaluation after loss plateau
- success rate
- torch version for TITAN Xp
- wandb
- os mesa

- seed: mean

- eval: CUDA_VISIBLE_DEVICES=0 python imitate_episodes_eval.py --task_name sim_transfer_cube_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval

```
# prerequisites
#sudo apt install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev mesa-utils
export MUJOCO_GL=egl

# virtual env settings
conda create -n np4a python=3.11
conda activate np4a
pip install -r requirements.txt

# make submodule
cd ./noisy-pretrained-act/detr && pip install -e .

# generating own scripted data by using original record_sim_episodes.py are not good data because of pooly success rate.
# python record_sim_episodes.py --task_name sim_insertion_scripted --dataset_dir ./data/sim_insertion_scripted --num_episodes 50

# so, download good (teleoperated) public data from here. (sim_insertion_human)
https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link

# generating noisy data
## noisy action: using record_sim_episodes.py is okay because success rate is not important, so action is doesn't matter. 
python record_sim_episodes.py --task_name sim_insertion_noisy_scripted --num_episodes 50
## noisy image
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_human
## noisy action and noisy image: overwrite image in noisy action data.
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_noisy_scripted

# check files
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_human --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_human_noisy_img --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_noisy_scripted --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_noisy_scripted_noisy_img --episode_idx all

# find loss plateau
CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0

CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ./ckpt/sim_transfer_cube_scripted_0_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
CUDA_VISIBLE_DEVICES=2 python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ./ckpt/sim_transfer_cube_scripted_0_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
CUDA_VISIBLE_DEVICES=3 python imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir ./ckpt/sim_insertion_scripted_0_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
CUDA_VISIBLE_DEVICES=4 python imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir ./ckpt/sim_insertion_scripted_0_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0

CUDA_VISIBLE_DEVICES=3 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_long --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 100000 --lr 1e-5 --seed 0
CUDA_VISIBLE_DEVICES=4 python imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir ./ckpt/sim_insertion_scripted_long --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 100000 --lr 1e-5 --seed 0

# pre-training & training
## insertion
### baseline 1: 사전학습 없음, baseline 2: 사전학습 없음, 사전학습 epoch만큼의 추가 학습
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0
### [노이즈 행동 + 이미지] 사전학습
python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir ./ckpt/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_noisy_scripted/policy_last.ckpt
### [노이즈 행동 + 노이즈 이미지] 사전학습
python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir ./ckpt/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_noisy_scripted_noisy_img/policy_last.ckpt
### [정상 행동 + 노이즈 이미지] 사전 학습
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_human_noisy_img/policy_last.ckpt

## example of training command in bi-stealth server
CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 && CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 && CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted/policy_last.ckpt

CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted_noisy_img/policy_last.ckpt && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_noisy_img/policy_last.ckpt

# evaluation
python3 imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval

```

# Noisy-Pretrained Action Chunking with Transformers
## Installation
```
# make virtual environment
conda create -n npact python=3.11
conda activate npact
pip install -r requirements.txt

# make submodule
cd ./noisy-pretrained-act/detr && pip install -e .

# generating own scripted data by using original record_sim_episodes.py are not good data because of pooly success rate.
python record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir ./data/sim_transfer_cube_scripted --num_episodes 50

# so, download good data
https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link

# and generating noisy data
## noising action only
python record_sim_episodes.py --task_name sim_transfer_cube_noisy_scripted --num_episodes 50
python record_sim_episodes.py --task_name sim_insertion_noisy_scripted --num_episodes 50
## noising image only
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_human
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_human
## noising both action and image
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_noisy_scripted
python overwrite_img_with_noise.py --target_dataset_folder /sim_insertion_noisy_scripted

# check files
## baseline data
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_human --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_human_noisy_img --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_noisy_scripted --episode_idx all
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_insertion_noisy_scripted_noisy_img --episode_idx all

# pre-training & training
## transfer cube
### baseline 1: 사전학습 없음
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
### [노이즈 행동 + 이미지] 사전학습
python imitate_episodes.py --task_name sim_transfer_cube_noisy_scripted --ckpt_dir ./ckpt/sim_transfer_cube_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_noisy_scripted/policy_last.ckpt
### [노이즈 행동 + 노이즈 이미지] 사전학습
python imitate_episodes.py --task_name sim_transfer_cube_noisy_scripted --ckpt_dir ./ckpt/sim_transfer_cube_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_noisy_scripted_noisy_img/policy_last.ckpt
### [정상 행동 + 노이즈 이미지] 사전 학습
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_human_noisy_img/policy_last.ckpt
### baseline 2: 사전학습 없음, 사전학습 epoch만큼의 추가 학습
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_4 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 4000 --lr 1e-5 --seed 0

## insertion
### baseline 1: 사전학습 없음, baseline 2: 사전학습 없음, 사전학습 epoch만큼의 추가 학습
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0
### [노이즈 행동 + 이미지] 사전학습
python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir ./ckpt/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_noisy_scripted/policy_last.ckpt
### [노이즈 행동 + 노이즈 이미지] 사전학습
python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir ./ckpt/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_noisy_scripted_noisy_img/policy_last.ckpt
### [정상 행동 + 노이즈 이미지] 사전 학습
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir ./ckpt/sim_insertion_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_insertion_human_noisy_img/policy_last.ckpt

## example of training command in bi-stealth server
CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 && CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 && CUDA_VISIBLE_DEVICES=0 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted/policy_last.ckpt

CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_noisy_scripted --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_noisy_scripted_noisy_img/policy_last.ckpt && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0 --noisy_img True && CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 20000 --lr 1e-5 --seed 0 --path2ckpt /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_noisy_img/policy_last.ckpt

# evaluation
python3 imitate_episodes.py --task_name sim_insertion_human --ckpt_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/ckpt_sim_insertion/sim_insertion_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval


```




# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

