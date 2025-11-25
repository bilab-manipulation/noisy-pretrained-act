import os
import torch
import numpy as np   # ← 추가
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

fig_tag = "baseline"
fig_tag = "noisyact"
fig_tag = "noisyactimg"
fig_tag = "noisyimg"

# --- 0. 저장 폴더 생성 ---
save_dir = f"compare_w/{fig_tag}"
os.makedirs(save_dir, exist_ok=True)

# --- 1. checkpoint 로드 함수 ---
def load_state_dict(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint

# --- 2. ckpt 경로 리스트 (여기만 수정) ---

if fig_tag == 'baseline':
    ckpt_paths = [
        # index 0: baseline
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_0/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_0/policy_last.ckpt",
    ]
elif fig_tag == 'noisyact':
    ckpt_paths = [
        # # index 1: noisy act
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_noisy_scripted/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_1/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_1/policy_last.ckpt",
    ]
elif fig_tag == 'noisyactimg':
    ckpt_paths = [
        # # index 2: noisy act + noisy img
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_noisy_scripted_noisy_img/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_2/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_2/policy_last.ckpt",
    ]
elif fig_tag == 'noisyimg':
    ckpt_paths = [
        # # index 3: noisy img
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_noisy_img/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_3/policy_epoch_0_seed_0.ckpt",
        "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/archive/kiise/sim_transfer_cube_human_3/policy_last.ckpt",
    ]


# checkpoint 개수 체크 (2 또는 3)
assert 2 <= len(ckpt_paths) <= 3, "Checkpoint는 2개 또는 3개만 사용할 수 있습니다."

# --- 모든 ckpt 로드 ---
state_dicts = [load_state_dict(p) for p in ckpt_paths]

# --- 3. 공통 키 계산 ---
common_keys = set(state_dicts[0].keys())
for sd in state_dicts[1:]:
    common_keys &= set(sd.keys())
common_keys = sorted(list(common_keys))

common_keys = ['model.action_head.weight']

print(f"공통 레이어 개수: {len(common_keys)}")

# --- 색상과 라벨 자동 생성 ---
colors = ["blue", "orange", "red"]
labels = [f"ckpt{i+1}" for i in range(len(state_dicts))]

if fig_tag == 'baseline':
    labels = ['after 1st epoch of post-training', 'after 2000th epoch of post-training']
    colors = ["#ff7f0e", "#2ca02c"]
    #colors = ["#f28e2b", "#e15759"]
else:
    labels = ['after 1st epoch of pretraining', 'after 1st epoch of post-training', 'after 2000th epoch of post-training']
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    #colors = ["#4e79a7", "#f28e2b", "#e15759"]


# --- 4. 레이어별 weight 비교 ---
for w_name in common_keys:
    print(f"{w_name}")

    # 모든 checkpoint에서 텐서인 경우만 처리
    tensors = []
    skip_flag = False
    for sd in state_dicts:
        w = sd[w_name]
        if torch.is_tensor(w):
            tensors.append(w.flatten().cpu().numpy())
        else:
            print(f"  (skip: not a tensor) {w_name}")
            skip_flag = True
            break
    if skip_flag:
        continue

    # 공통 bin 범위를 위해 전체 min/max 계산
    all_min = min(arr.min() for arr in tensors)
    all_max = max(arr.max() for arr in tensors)

    # ----- (1) 히스토그램: 한 그래프에 여러 히스토그램을 겹쳐 그리기 -----
    fig, ax = plt.subplots(figsize=(8, 4))

    num_bins = 100
    for i, arr in enumerate(tensors):
        ax.hist(
            arr,
            bins=num_bins,
            range=(all_min, all_max),
            alpha=0.4,               # 서로 겹치는 부분 보이도록 투명도
            color=colors[i],
            label=labels[i],
        )

    #ax.set_title(f"Weight Distribution: {w_name}")
    if fig_tag == 'baseline':
        title_name = 'Baseline'
    elif fig_tag == 'noisyact':
        title_name = 'Noisy action pretraining + post-training'
    elif fig_tag == 'noisyactimg':
        title_name = 'Noisy image & noisy action pretraining + post-training'
    elif fig_tag == 'noisyimg':
        title_name = 'Noisy image pretraining + post-training'
    ax.set_title(f"Weight distribution:\n{title_name}")
    ax.set_xlabel("Weight value in action-head layer")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 175)
    ax.grid(True)
    ax.legend(loc = 'upper right')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    # 파일명 안전하게 변환
    safe_name = w_name.replace("/", "_")
    plt.savefig(os.path.join(save_dir, f"{fig_tag}-{safe_name}.png"))
    plt.close(fig)

    # ----- (2) 추가: 표준편차 bar plot -----
    # 각 checkpoint에서 해당 레이어 weight의 표준편차 계산
    std_values = [np.std(arr) for arr in tensors]
    print(std_values)

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    x = np.arange(len(std_values))

    ax_bar.bar(x, std_values, color=colors[:len(std_values)], alpha = 0.4)
    ax_bar.set_xticks(x)
    #ax_bar.set_xticklabels(labels, rotation=20, ha='right')
    if fig_tag == 'baseline':
        labels = ['after 1st epoch\nof post-training', 'after 2000th epoch\nof post-training']
    else:
        labels = ['after 1st epoch\nof pretraining', 'after 1st epoch\nof post-training', 'after 2000th epoch\nof post-training']
    ax_bar.set_xticklabels(labels)

    ax_bar.set_title(f"Std of Weights: {w_name}")
    ax_bar.set_ylabel("Standard Deviation")
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig_bar.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fig_tag}-{safe_name}_std_bar.png"))
    plt.close(fig_bar)

print("완료!")
