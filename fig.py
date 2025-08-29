import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
labels = ['Baseline', 'Baseline_2', 'Noisy image pretraining', 'Noisy action pretraining', 'Both noisy pretraining']
values = [30, 62, 38, 48, 36]
colors = ['black', 'black', 'black', 'black', 'black']

plt.barh(x, values, color=colors)
for i, value in enumerate(values):
    plt.text(value + 0.1, i, str(value), va='center')
plt.yticks(x, labels)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('eval_fig.png')