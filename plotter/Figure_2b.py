import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')
from matplotlib.backends.backend_pdf import PdfPages

fig = plt.figure(figsize=(8,5),dpi=300)
# palette = mpl.color_sequences['tab20']
# colors = [palette[5],palette[7],palette[8]]
colors = [
"#01973F",
"#FA0404",
"#1601FA",
]
line_width = 2.0
legends = ['batch 1','batch 2','batch 3']
root = '../data/TJU data/'
batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
for i in range(3):
    batch = batchs[i]
    batch_root = os.path.join(root,batch)
    files = os.listdir(batch_root)
    for f in files:
        path = os.path.join(batch_root,f)
        data = pd.read_csv(path)
        capacity = data['capacity'].values
        plt.plot(capacity[1:],color=colors[i],alpha=1,linewidth=line_width)
plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')
custom_lines = [
    Line2D([0], [0], color=colors[0], linewidth=line_width),
    Line2D([0], [0], color=colors[1], linewidth=line_width),
    Line2D([0], [0], color=colors[2], linewidth=line_width)
]

custom_legend = plt.legend(custom_lines, legends, loc='upper right',
                           bbox_to_anchor=(1.0, 1), frameon=True, 
                           ncol=1, fontsize=16)

plt.ylim([1.4,3.6])
plt.yticks([1.5,2.0,2.5,3.0,3.5],['1.5','2.0','2.5','3.0','3.5'])
plt.tight_layout()
plt.show()
plt.savefig('../Figures/tju_trajectory.svg',format='svg')