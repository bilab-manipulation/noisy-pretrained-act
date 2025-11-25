import numpy as np
import matplotlib.pyplot as plt

# 글꼴 설정: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

value_list = []
value_list.extend([np.float32(0.00), np.float32(0.025565192), np.float32(0.023824966)])
value_list.extend([np.float32(0.025564754), np.float32(0.024064895), np.float32(0.0228798)])
value_list.extend([np.float32(0.025564628), np.float32(0.02464119), np.float32(0.023304267)])
value_list.extend([np.float32(0.025564525), np.float32(0.024696685), np.float32(0.023217488)])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ===== 그룹 설정 =====
group_size = 3
num_groups = len(value_list) // group_size

intra_gap = 0.8
inter_gap = 0.5

x = []
current_x = 0

for g in range(num_groups):
    for i in range(group_size):
        x.append(current_x)
        current_x += intra_gap
    current_x += inter_gap

bar_colors = [colors[i % group_size] for i in range(len(value_list))]

# 그룹 라벨
group_labels = ["Baseline", "Noisy image pretraining\n+ post-training", "Noisy action pretraining\n+ post-training", "Noisy image\n& Noisy action pretraining\n+ post-training"]

group_positions = []
pos = 0
for g in range(num_groups):
    center = pos + (group_size - 1) * intra_gap / 2
    group_positions.append(center)
    pos += group_size * intra_gap + inter_gap

# bar label
bar_labels = [
    'after 1st epoch of pretraining',
    'after 1st epoch of post-training',
    'after 2000th epoch of post-training'
]

# ===== Plot =====
plt.figure(figsize=(7, 5))
bars = plt.bar(x, value_list, color=bar_colors, alpha=0.4)

# bar 위 텍스트 라벨
for i, (rect, val) in enumerate(zip(bars, value_list)):
    label = bar_labels[i % 3]

# ===== Legend 추가 =====
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=colors[i], label=bar_labels[i], alpha=0.4) for i in range(3)]
plt.legend(handles=legend_handles, loc='upper right')

plt.xticks(group_positions, group_labels)
plt.xlabel("Learning strategy")
plt.ylabel("Std of action-head layer weights")
plt.title("Changes in weight distributions before and after training")
plt.ylim(0.0, 0.035)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('std_bar.png')
plt.close()