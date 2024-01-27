import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['Public Test Dataset', 'Private Test Dataset']
DegScore = [0.392, 0.473]
DegScore_XGBoost = [0.359, 0.439]
Kazuki2 = [0.228, 0.343]
Nullrecurrent = [0.228, 0.342]
Our = [0.198, 0.296]

x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots()
rects1 = ax.bar(x - width * 2 + 0.03, DegScore, width, label='DegScore', color='#cb997e')
rects2 = ax.bar(x - width + 0.06, DegScore_XGBoost, width, label='DegScore-XGBoost', color='#ddbea9')
rects3 = ax.bar(x + 0.09, Kazuki2, width, label='Kazuki2', color='#fde8d5')
rects4 = ax.bar(x + width + 0.12, Nullrecurrent, width, label='Nullrecurrent', color='#b8b7a3')
rects5 = ax.bar(x + width * 2 + 0.15, Our, width, label='Our', color='#a5a58d')

ax.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 16})

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', ncol=3, prop={'family': 'Times New Roman', 'size': 12})

plt.ylim((0, 0.6))
yticks = [i * 0.05 for i in range(int(0.6 / 0.05) + 1)]
ax.set_yticks(yticks)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)

plt.gca().spines['bottom'].set_linewidth(1.5)  # 设置 x 轴线条宽度
plt.gca().spines['left'].set_linewidth(1.5)  # 设置 y 轴线条宽度
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height, 'Times New Roman', size=16),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()

plt.savefig('./model_perfor_kaggle.png', dpi=500)
