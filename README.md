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
## noising image only
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_human
## noising both action and image
python overwrite_img_with_noise.py --target_dataset_folder /sim_transfer_cube_noisy_scripted

# check files
## baseline data
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_human --episode_idx 0
## nosing action data
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_noisy_scripted --episode_idx 0
## noising image data
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_human_noisy_img --episode_idx 0
## noising both data
python visualize_episodes.py --dataset_dir /media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_noisy_scripted_noisy_img --episode_idx 0

# pre-training & training
## baseline 1: 사전학습 없음
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
## [노이즈 행동 + 이미지] 사전학습
python imitate_episodes.py --task_name sim_transfer_cube_noisy_scripted --ckpt_dir ./ckpt/sim_transfer_cube_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_1 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_noisy_scripted/policy_last.ckpt
## [노이즈 행동 + 노이즈 이미지] 사전학습
python imitate_episodes.py --task_name sim_transfer_cube_noisy_scripted --ckpt_dir ./ckpt/sim_transfer_cube_noisy_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_noisy_scripted_noisy_img/policy_last.ckpt
## [정상 행동 + 노이즈 이미지] 사전 학습
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --noisy_img True
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_3 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --path2ckpt ./ckpt/sim_transfer_cube_human_noisy_img/policy_last.ckpt
## baseline 2: 사전학습 없음, 사전학습 epoch만큼의 추가 학습
python imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_4 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 4000 --lr 1e-5 --seed 0

# evaluation
python3 imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir ./ckpt/sim_transfer_cube_human_0 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval


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

