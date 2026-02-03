import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# plt.style.use('own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')
from matplotlib.backends.backend_pdf import PdfPages

fig, ax = plt.subplots(figsize=(8,5),dpi=300)
line_width = 2.0
root = '../data/MIT data/'
capacity_curves = []
sample_counts = []
for batch in ['2017-05-12','2017-06-30','2018-04-12']:
    batch_root = os.path.join(root,batch)
    files = os.listdir(batch_root)
    for f in files:
        path = os.path.join(batch_root,f)
        data = pd.read_csv(path)
        capacity = data['capacity'].values
        capacity_curves.append(capacity[1:])
        sample_counts.append(len(capacity[1:]))

norm = plt.Normalize(min(sample_counts), max(sample_counts)) # 归一化到 [0, 1]（用于颜色映射）
cmap = plt.get_cmap('spring_r')  # 选择颜色映射（如 'viridis'，可替换为 'turbo' 'plasma' 'inferno' 'magma' 'cividis' 等）

for curve, count in zip(capacity_curves, sample_counts):
    ax.plot(curve,
             color=cmap(norm(count)),
             alpha=1,
             linewidth=line_width)
    
# 添加颜色条（Legend for sample counts）
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 关联数据
cbar = fig.colorbar(
    sm, 
    ax=ax, 
    # label='Number of Sample Points',
    shrink=1,       # 颜色条高度（已设置）
    aspect=30,        # 颜色条长宽比（值越大越窄）
    pad=0.02,         # 颜色条与主图的间距（默认0.05，减小为0.02）
)
# cbar.ax.set_ylabel('Number of Sample Points', fontsize=14)
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel('Cycle', fontsize=20)
ax.set_ylabel('Capacity (Ah)', fontsize=20)
ax.set_ylim([0.87,1.11])
ax.tick_params(labelsize=18)
plt.yticks([0.90,0.95,1.00,1.05,1.10],['0.90','0.95','1.00','1.05','1.10'])
plt.tight_layout()
plt.subplots_adjust(right=1.04)
plt.show()
plt.savefig('../Figures/mit_trajectory.svg',format='svg')