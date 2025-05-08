import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
years = ['Baseline', 'Noisy image pretraining', 'Noisy action pretraining', 'Both noisy pretraining']
values = [30, 38, 48, 36]
colors = ['green', 'red', 'blue', 'purple']

plt.bar(x, values, color=colors)
plt.xticks(x, years, rotation=45)
plt.tight_layout()
plt.savefig('eval_fig.png')