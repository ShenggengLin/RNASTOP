import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman', size=14)

np.random.seed(10)
# 999
data_gE = [[-576.07],
           [-576.57, -577.59, -577.92, -577.09, -575.91, -576.23, -577.68, -577.11, -577.94, -576.84, -575.9, -576.47],
           [-576.42, -576.73, -578.19, -577.61, -578.44, -577.34, -576.4, -577.36, -577.76, -579.21, -578.64, -579.47,
            -578.37, -577.33, -577.63, -578.08, -579.53, -578.96, -579.79, -578.69, -577.61, -576.82, -577.25, -578.7,
            -578.14, -578.96, -577.87, -576.81, -576.07, -577.53, -576.95, -577.78, -576.68, -576.07, -577.52, -576.95,
            -577.78, -576.68, -576.29, -576.64, -578.1, -577.52, -578.35, -577.25, -576.28],
           [-576.58, -578.05, -577.46, -578.29, -577.19, -576.56, -578.03, -577.44, -578.27, -577.17, -577.52, -578.98,
            -578.4, -579.23, -578.13, -577.5, -578.96, -578.38, -579.21, -578.11, -577.79, -579.24, -578.67, -579.5,
            -578.4, -577.77, -579.22, -578.65, -579.48, -578.38, -576.98, -578.43, -577.87, -578.69, -577.6, -576.97,
            -578.43, -577.86, -578.69, -577.59, -576.45, -577.93, -577.34, -578.16, -577.06, -576.44, -577.92, -577.33,
            -578.15, -577.05]]
# 999
data_COVID19 = [[-1397.09],
                [-1397.06, -1397.09, -1399.22],
                [-1397.06, -1399.19, -1399.22],
                [-1399.19]]

labels_gE = ['Base', 'One Mut', 'Two Mut', 'Three Mut']
labels_covid19 = ['Base', 'One Mut', 'Two Mut', 'Three Mut']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

bplot1 = ax1.boxplot(data_COVID19, labels=labels_covid19, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black'),
                     capprops={'linewidth': 1},
                     boxprops={'linewidth': 1},
                     whiskerprops={'linewidth': 1},
                     )  # , patch_artist=True)
ax1.set_title('COVID19', fontdict={'family': 'Times New Roman', 'size': 16})

ax1.set_ylabel('Minimum Free Energy', fontdict={'family': 'Times New Roman', 'size': 16})

ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)

bplot2 = ax2.boxplot(data_gE, labels=labels_gE, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black'),
                     capprops={'linewidth': 1},
                     boxprops={'linewidth': 1},
                     whiskerprops={'linewidth': 1})  # , patch_artist=True)
ax2.set_title('VZV gE', fontdict={'family': 'Times New Roman', 'size': 16})
ax2.set_ylabel('Minimum Free Energy', fontdict={'family': 'Times New Roman', 'size': 16})
# ax2.set_ylim(0.5, 1.0)
# ax2.tick_params(axis='x', rotation=45)
colors1 = ['#CEDFEF', '#B4B4D5', '#8481BA', '#614099']
colors2 = ['#CEDFEF', '#B4B4D5', '#8481BA', '#614099']

for patch, color in zip(bplot1['boxes'], colors1):
    patch.set_facecolor(color)

for patch, color in zip(bplot2['boxes'], colors2):
    patch.set_facecolor(color)

ax2.spines['bottom'].set_linewidth(1.5)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['top'].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)

plt.tight_layout()
# plt.show()
plt.savefig('./RNAopt_boxplot.png', dpi=500)
