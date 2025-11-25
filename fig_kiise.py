import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 글꼴 설정: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

x = np.arange(4)
labels = ['Baseline\n(only post-training)', 'Noisy image pretraining\n+ post-training',
          'Noisy action pretraining\n+ post-training', 'Noisy image\n& Noisy action pretraining\n+ post-training']
values = [26, 32, 40, 40]

plt.figure(figsize=(8, 4))

# 1~3번째 bar: 단색
colors = ['green', 'red', 'blue']
colors = ["#C5E0B4", "#F8CBCD", "#BDD7EE"]

for i in range(3):
    plt.barh(i, values[i], color=colors[i])

# --- 4번째 bar: 위=빨강 / 아래=파랑 그라데이션 ---

value = values[3]

# 그라데이션용 colormap
cmap = LinearSegmentedColormap.from_list("red_blue_grad", ["#BDD7EE", "#DBD1CE", "#F8CBCD"])

# 그라데이션 데이터 (세로 방향)
grad = np.linspace(0, 1, 256).reshape(256, 1)

# bar의 y 영역
y_center = 3
height = 0.8
y0 = y_center - height/2
y1 = y_center + height/2

# 그라데이션 이미지를 bar 위에 덧씌움
plt.imshow(grad,
           extent=[0, value, y0, y1],  # x 범위=bar 길이, y 범위=bar 높이
           cmap=cmap,
           aspect='auto')

# 값 텍스트 표시
for i, value in enumerate(values):
    plt.text(value + 0.3, i, str(value), va='center')

plt.yticks(x, labels)
plt.xlabel('Task success rate (%)')
plt.ylim(-0.6, 3.6)   # 위/아래 여백
plt.xlim(0, 45)
plt.tight_layout()
plt.gca().invert_yaxis()

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('eval_fig_kiise.png', dpi=300)
plt.show()
