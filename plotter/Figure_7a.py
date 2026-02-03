import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
# plt.style.use(['science','nature'])
plt.style.use('../plotter/own_style_5.mplstyle')    # 1
plt.rc('font', family='Times New Roman')

root = '../Notebooks/viz_folder/'
files = os.listdir(root)
fig = plt.figure(figsize=(8,6),dpi=300)
true_values = np.load(root+'xjtu_batch_0_true_values.npy')
pikan_model_forecast = np.load(root+'xjtu_batch_0_pikan_model_forecast.npy')
pikan_symbolic_forecast = np.load(root+'xjtu_batch_0_pikan_symbolic_forecast.npy')
# colors = [
# "#3EE454",  
# "#13C8EC",
# "#1361F0",
# "#F35A26",
# "#F71432"
# ]
colors = ["#22d34e","#2ab2f7","#fd4d53"]
# markers = ['o','v','D','p','s','^']
# legends = ['batch 1','batch 2','batch 3','batch 4','batch 5','batch 6']
# batches = ['2C','3C','R2.5','R3','RW','satellite']
# line_width = 2.0
# for i in range(5):
#     for f in files:
#         if batches[i] in f:
#             path = os.path.join(root,f)
#             data = pd.read_csv(path)
#             capacity = data['capacity'].values
#             plt.plot(capacity[1:],color=colors[i],alpha=1,linewidth=line_width,
#                      # linestyle=':',marker=markers[i],markersize=2,markevery=50
#                     )

# Plot the second interval (entire dataset)
plt.plot(true_values, label='Real Data', color=colors[0], linewidth=2) # marker=markers[0],
plt.plot(pikan_model_forecast, label='PIKAN* Model', color=colors[1], linewidth=2)
plt.plot(pikan_symbolic_forecast, label='PIKAN* Symbolic', color=colors[2], linewidth=2)
# plt.ylim((0.775, 1.025))
# plt.yticks(np.arange(0.80, 1.05, 0.05))

plt.title('XJTU batch 1')
plt.xlabel('Sample Index')
plt.ylabel('SOH Value')
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('../Figures/PIKAN_xjtu_batch_0_true_vs_model_vs_symbolic_bold.svg',format='svg')
plt.show()