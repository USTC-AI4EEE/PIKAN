import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plotter/own_style_1.mplstyle')    # 5
plt.rc('font', family='Times New Roman')


def plot():
    root = 'Notebooks/viz_folder/'
    files = os.listdir(root)
    fig = plt.figure(figsize=(8,6),dpi=300)
    true_values = np.load(root+'tju_batch_2_true_values.npy')
    pikan_model_forecast = np.load(root+'tju_batch_2_pikan_model_forecast.npy')
    pikan_symbolic_forecast = np.load(root+'tju_batch_2_pikan_symbolic_forecast.npy')
    colors = ["#22d34e","#2ab2f7","#fd4d53"]

    # Plot the second interval (entire dataset)
    plt.plot(true_values, label='Real Data', color=colors[0], linewidth=2) # marker=markers[0],
    plt.plot(pikan_model_forecast, label='PIKAN* Model', color=colors[1], linewidth=2)
    plt.plot(pikan_symbolic_forecast, label='PIKAN* Symbolic', color=colors[2], linewidth=2)
    # plt.set(ylim=(0.81, 1.04),yticks=np.arange(0.85, 1.00, 0.05))

    plt.title('TJU batch 3')
    plt.xlabel('Sample Index')
    plt.ylabel('SOH Value')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figures/PIKAN_tju_batch_2_true_vs_model_vs_symbolic.svg',format='svg')
    plt.show()


if __name__ == '__main__':
    plot()