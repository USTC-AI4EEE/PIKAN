import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
# plt.style.use(['science','nature'])
plt.style.use('own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')
from matplotlib.backends.backend_pdf import PdfPages

root = '../data/XJTU data/'
# pdf = PdfPages('xjtu_trajectory.pdf')
files = os.listdir(root)
fig = plt.figure(figsize=(8,6),dpi=300) #(8,5)
# colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
colors = [
'#80A6E2',
'#7BDFF2',
'#FBDD85',
'#F46F43',
'#403990',
'#CF3D3E'
]
markers = ['o','v','D','p','s','^']
legends = ['batch 1','batch 2','batch 3','batch 4','batch 5','batch 6']
batches = ['2C','3C','R2.5','R3','RW','satellite']
line_width = 2.0
for i in range(6):
    for f in files:
        if batches[i] in f:
            path = os.path.join(root,f)
            data = pd.read_csv(path)
            capacity = data['capacity'].values
            plt.plot(capacity[1:],color=colors[i],alpha=1,linewidth=line_width,
                     # linestyle=':',marker=markers[i],markersize=2,markevery=50
                    )
plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')
custom_lines = [
    # Line2D([0], [0], color=colors[0], linewidth=line_width,marker=markers[0],markersize=5),
    # Line2D([0], [0], color=colors[1], linewidth=line_width,marker=markers[1],markersize=5),
    # Line2D([0], [0], color=colors[2], linewidth=line_width,marker=markers[2],markersize=5),
    # Line2D([0], [0], color=colors[3], linewidth=line_width,marker=markers[3],markersize=5),
    # Line2D([0], [0], color=colors[4], linewidth=line_width,marker=markers[4],markersize=5),
    # Line2D([0], [0], color=colors[5], linewidth=line_width,marker=markers[5],markersize=5)
    Line2D([0], [0], color=colors[0], linewidth=line_width),
    Line2D([0], [0], color=colors[1], linewidth=line_width),
    Line2D([0], [0], color=colors[2], linewidth=line_width),
    Line2D([0], [0], color=colors[3], linewidth=line_width),
    Line2D([0], [0], color=colors[4], linewidth=line_width),
    Line2D([0], [0], color=colors[5], linewidth=line_width)
]

custom_legend = plt.legend(custom_lines, legends, loc='upper right',
                           bbox_to_anchor=(1.0, 1), frameon=True, handletextpad=0.4, columnspacing=0.6,
                           ncol=2, fontsize=18)


plt.ylim([1.55,2.05])
plt.tight_layout()
plt.show()
plt.savefig('../Figures/xjtu_trajectory_v2.svg',format='svg')