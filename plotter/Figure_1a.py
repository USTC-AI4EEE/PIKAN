import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('plotter/own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')


def plot():
    root = 'Data/XJTU data/'
    files = os.listdir(root)
    fig = plt.figure(figsize=(8,5),dpi=300) # v2: (8,6)
    colors = [
    '#80A6E2',
    '#7BDFF2',
    '#FBDD85',
    '#F46F43',
    '#403990',
    '#CF3D3E'
    ]
    legends = ['batch 1','batch 2','batch 3','batch 4','batch 5','batch 6']
    batches = ['2C','3C','R2.5','R3','RW','satellite']
    line_width = 2.0
    for i in range(6):
        for f in files:
            if batches[i] in f:
                path = os.path.join(root,f)
                data = pd.read_csv(path)
                capacity = data['capacity'].values
                plt.plot(capacity[1:],color=colors[i],alpha=1,linewidth=line_width)
    plt.xlabel('Cycle')
    plt.ylabel('Capacity (Ah)')
    custom_lines = [
        Line2D([0], [0], color=colors[0], linewidth=line_width),
        Line2D([0], [0], color=colors[1], linewidth=line_width),
        Line2D([0], [0], color=colors[2], linewidth=line_width),
        Line2D([0], [0], color=colors[3], linewidth=line_width),
        Line2D([0], [0], color=colors[4], linewidth=line_width),
        Line2D([0], [0], color=colors[5], linewidth=line_width)
    ]

    plt.legend(custom_lines, legends, loc='upper right',
                            bbox_to_anchor=(1.0, 1), frameon=True, handletextpad=0.4, columnspacing=0.6,
                            ncol=2, fontsize=18)

    plt.ylim([1.55,2.05])
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/xjtu_trajectory.svg',format='svg')


if __name__ == '__main__':
    plot()