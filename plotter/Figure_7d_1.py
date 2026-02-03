import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use('../plotter/own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')

root = '../Notebooks/viz_folder/'
files = os.listdir(root)
fig = plt.figure(figsize=(8,6),dpi=300)
true_values = np.load(root+'hust_true_values.npy')
kan_model_forecast = np.load(root+'hust_kan_model_forecast.npy')
kan_symbolic_forecast = np.load(root+'hust_kan_symbolic_forecast.npy')
pikan_model_forecast = np.load(root+'hust_pikan_model_forecast.npy')
pikan_symbolic_forecast = np.load(root+'hust_pikan_symbolic_forecast.npy')
# colors = [
# "#3EE454",  
# "#13C8EC",
# "#1361F0",
# "#F35A26",
# "#F71432"
# ]
colors = ["#22d34e","#2ab2f7","#e172eb","#fd924b","#fd4d53"]
markers = ['o','v','D','p','s','^']
# Plot the second interval (entire dataset)
plt.plot(true_values, label='Real Data', color=colors[0], linewidth=1.5) # marker=markers[0],
plt.plot(kan_model_forecast, label='KAN Model', color=colors[1], linewidth=1.5)
plt.plot(kan_symbolic_forecast, label='KAN Symbolic', color=colors[2], linewidth=1.5)
plt.plot(pikan_model_forecast, label='PIKAN Model', color=colors[3], linewidth=1.5)
plt.plot(pikan_symbolic_forecast, label='PIKAN Symbolic', color=colors[4], linewidth=1.5)
# plt.set(ylim=(0.81, 1.04),yticks=np.arange(0.85, 1.00, 0.05))

plt.title('HUST dataset')
plt.xlabel('Sample Index')
plt.ylabel('SOH Value')
plt.legend(fontsize=16,loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('../Figures/hust_true_vs_model_vs_symbolic.svg',format='svg')
plt.show()