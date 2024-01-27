import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman', size=14)

x_lr = [4e-2, 4e-3, 4e-4, 4e-5]
y_lr_pub = [0.4186, 0.3835, 0.1982, 0.2047]
y_lr_pri = [0.4536, 0.4226, 0.2962, 0.3039]

x_bz = [16, 32, 64, 100]
y_bz_pub = [0.2001, 0.2071, 0.1982, 0.2006]
y_bz_pri = [0.2991, 0.3012, 0.2962, 0.3036]

x_cn = [3, 7, 11, 15]
y_cn_pub = [0.2072, 0.1982, 0.2090, 0.1936]
y_cn_pri = [0.3039, 0.2962, 0.2967, 0.3028]

x_dr = [0.1, 0.2, 0.3, 0.4]
y_dr_pub = [0.2137, 0.2076, 0.1982, 0.2217]
y_dr_pri = [0.3020, 0.2983, 0.2962, 0.3028]

x_eb = [32, 64, 128, 256]
y_eb_pub = [0.2205, 0.2243, 0.1982, 0.2213]
y_eb_pri = [0.3256, 0.3069, 0.2962, 0.3080]

x_el = [2, 3, 4, 5]
y_el_pub = [0.2142, 0.2132, 0.1982, 0.1997]
y_el_pri = [0.3014, 0.3038, 0.2962, 0.3054]

fig, axs = plt.subplots(2, 3)

ax11 = axs[0, 0]
ax11.plot(x_lr, y_lr_pub, color='#FA7F6F', marker='.')
ax11.plot(x_lr, y_lr_pri, color='#8ECFC9', marker='.')
ax11.set_title('Learning Rate', fontdict={'family': 'Times New Roman', 'size': 10})
ax11.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax11.set_ylim([0, 0.55])
ax11.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
ax11.set_xticklabels([0, 0.01, 0.02, 0.03, 0.04], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax11.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax11.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontdict={'family': 'Times New Roman', 'size': 10})

ax12 = axs[0, 1]
ax12.plot(x_bz, y_bz_pub, color='#FA7F6F', marker='.')
ax12.plot(x_bz, y_bz_pri, color='#8ECFC9', marker='.')
ax12.set_title('Batch Size', fontdict={'family': 'Times New Roman', 'size': 10})
ax12.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax12.set_ylim([0, 0.45])
ax12.set_xticks([16, 32, 64, 100])
ax12.set_xticklabels([16, 32, 64, 100], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax12.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax12.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontdict={'family': 'Times New Roman', 'size': 10})

ax13 = axs[0, 2]
ax13.plot(x_cn, y_cn_pub, color='#FA7F6F', marker='.')
ax13.plot(x_cn, y_cn_pri, color='#8ECFC9', marker='.')
ax13.set_title('Convolution Kernel Size', fontdict={'family': 'Times New Roman', 'size': 10})
ax13.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax13.set_ylim([0, 0.45])
ax13.set_xticks([3, 7, 11, 15])
ax13.set_xticklabels([3, 7, 11, 15], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax13.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax13.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontdict={'family': 'Times New Roman', 'size': 10})

ax21 = axs[1, 0]
ax21.plot(x_dr, y_dr_pub, color='#FA7F6F', marker='.')
ax21.plot(x_dr, y_dr_pri, color='#8ECFC9', marker='.')
ax21.set_title('Dropout Rate', fontdict={'family': 'Times New Roman', 'size': 10})
ax21.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax21.set_ylim([0, 0.45])
ax21.set_xticks([0.1, 0.2, 0.3, 0.4])
ax21.set_xticklabels([0.1, 0.2, 0.3, 0.4], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax21.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax21.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontdict={'family': 'Times New Roman', 'size': 10})

ax22 = axs[1, 1]
ax22.plot(x_eb, y_eb_pub, color='#FA7F6F', marker='.')
ax22.plot(x_eb, y_eb_pri, color='#8ECFC9', marker='.')
ax22.set_title('Embedding Dimension', fontdict={'family': 'Times New Roman', 'size': 10})
ax22.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax22.set_ylim([0, 0.45])
ax22.set_xticks([32, 64, 128, 256])
ax22.set_xticklabels([32, 64, 128, 256], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax22.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax22.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontdict={'family': 'Times New Roman', 'size': 10})

ax23 = axs[1, 2]
ax23.plot(x_el, y_el_pub, color='#FA7F6F', marker='.')
ax23.plot(x_el, y_el_pri, color='#8ECFC9', marker='.')
ax23.set_title('Encoder Layer', fontdict={'family': 'Times New Roman', 'size': 10})
ax23.set_ylabel('MCRMSE', fontdict={'family': 'Times New Roman', 'size': 10})
ax23.set_ylim([0, 0.45])
ax23.set_xticks([2, 3, 4, 5])
ax23.set_xticklabels([2, 3, 4, 5], rotation=30, fontdict={'family': 'Times New Roman', 'size': 10})
ax23.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax23.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontdict={'family': 'Times New Roman', 'size': 10})

plt.tight_layout()

plt.savefig('./hyper_pama_zhexiantu.png', dpi=500)
