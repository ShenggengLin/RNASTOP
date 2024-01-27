import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/GSE173083_188_withstrloop.csv')

rna_lengths = df['RNA length']
rna_degradation_rates = df['In-solution degradation coefficient']
rna_types = df['Group']
MEV_num = 0
eGFP_num = 0
cds_num = 0
utr_num = 0
for i in range(len(rna_types)):
    if 'MEV' in rna_types[i]:
        rna_types[i] = 'MEV'
        MEV_num += 1
    if 'eGFP' in rna_types[i]:
        rna_types[i] = 'eGFP'
        eGFP_num += 1
    if 'CDS (Nluc) variants' == rna_types[i]:
        rna_types[i] = 'NLuc CDS variants'
        cds_num += 1
    if 'UTR' in rna_types[i]:
        rna_types[i] = 'NLuc UTR variants'
        utr_num += 1
    if 'hHBB NlucP' == rna_types[i]:
        rna_types[i] = 'NLuc UTR variants'
        utr_num += 1
    if 'hHBB Nluc' == rna_types[i]:
        rna_types[i] = 'NLuc UTR variants'
        utr_num += 1

for i in range(len(rna_types)):
    print(rna_types[i])
print(MEV_num)
print(eGFP_num)
print(cds_num)
print(utr_num)

colors = {'MEV': '#8ECFC9', 'eGFP': '#FFBE7A', 'NLuc CDS variants': '#FA7F6F',
          'NLuc UTR variants': '#82B0D2'}  # Add more types if needed
rna_colors = [colors[type_] for type_ in rna_types]

plt.scatter(rna_degradation_rates, rna_lengths, c=rna_colors)

plt.xlabel('In-solution degradation coefficient', fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel('mRNA length', fontdict={'family': 'Times New Roman', 'size': 16})

plt.legend(
    handles=[plt.Line2D([0], [0], marker='o', markerfacecolor='#8ECFC9', label='MEV', markersize=10, color='#8ECFC9'),
             plt.Line2D([0], [0], marker='o', markerfacecolor='#FFBE7A', label='eGFP', markersize=10, color='#FFBE7A'),
             plt.Line2D([0], [0], marker='o', markerfacecolor='#FA7F6F', label='NLuc CDS variants', markersize=10,
                        color='#FA7F6F'),
             plt.Line2D([0], [0], marker='o', markerfacecolor='#82B0D2', label='NLuc UTR variants', markersize=10,
                        color='#82B0D2'),
             ], prop={'family': 'Times New Roman', 'size': 14}, loc='lower right')

plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)

plt.ylim((0, 2000))
yticks = [i * 500 for i in range(int(2000 / 500) + 1)]

plt.yticks(fontproperties='Times New Roman', size=15)
plt.xticks(fontproperties='Times New Roman', size=15)

plt.savefig('./PERSISI-seq dataset.png', dpi=500)
