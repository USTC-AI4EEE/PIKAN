import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
# plt.style.use(['science','nature'])
plt.style.use('../plotter/own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')

def relative_estimation_error_abs(true_values, estimated_values, eps=1e-10):
    """
    计算相对估计误差绝对值数组，处理真实值为零的情况
    
    参数:
    true_values (np.ndarray): 真实值数组
    estimated_values (np.ndarray): 估计值数组
    eps (float): 极小值，用于避免除零错误
    
    返回:
    np.ndarray: 相对估计误差绝对值数组（百分比）
    """
    # 避免除零：当真实值接近0时，用eps代替
    safe_true = np.where(np.abs(true_values) < eps, eps, true_values)
    return np.abs((true_values - estimated_values) / safe_true) * 100

root = '../Notebooks/viz_folder/'
files = os.listdir(root)
# xjtu_batch_0
xjtu_batch_0_true_values = np.load(root+'xjtu_batch_0_true_values.npy')
xjtu_batch_0_kan_model_forecast = np.load(root+'xjtu_batch_0_kan_model_forecast.npy')
xjtu_batch_0_kan_symbolic_forecast = np.load(root+'xjtu_batch_0_kan_symbolic_forecast.npy')
xjtu_batch_0_pikan_model_forecast = np.load(root+'xjtu_batch_0_pikan_model_forecast.npy')
xjtu_batch_0_pikan_symbolic_forecast = np.load(root+'xjtu_batch_0_pikan_symbolic_forecast.npy')
xjtu_batch_0_kan_model_error = relative_estimation_error_abs(xjtu_batch_0_true_values,xjtu_batch_0_kan_model_forecast).ravel()  # 展平为一维数组
xjtu_batch_0_kan_symbolic_error = relative_estimation_error_abs(xjtu_batch_0_true_values,xjtu_batch_0_kan_symbolic_forecast).ravel()  # 展平为一维数组
xjtu_batch_0_pikan_model_error = relative_estimation_error_abs(xjtu_batch_0_true_values,xjtu_batch_0_pikan_model_forecast).ravel()  # 展平为一维数组
xjtu_batch_0_pikan_symbolic_error = relative_estimation_error_abs(xjtu_batch_0_true_values,xjtu_batch_0_pikan_symbolic_forecast).ravel()  # 展平为一维数组
# tju_batch_2
tju_batch_2_true_values = np.load(root+'tju_batch_2_true_values.npy')
tju_batch_2_kan_model_forecast = np.load(root+'tju_batch_2_kan_model_forecast.npy')
tju_batch_2_kan_symbolic_forecast = np.load(root+'tju_batch_2_kan_symbolic_forecast.npy')
tju_batch_2_pikan_model_forecast = np.load(root+'tju_batch_2_pikan_model_forecast.npy')
tju_batch_2_pikan_symbolic_forecast = np.load(root+'tju_batch_2_pikan_symbolic_forecast.npy')
tju_batch_2_kan_model_error = relative_estimation_error_abs(tju_batch_2_true_values,tju_batch_2_kan_model_forecast).ravel()  # 展平为一维数组
tju_batch_2_kan_symbolic_error = relative_estimation_error_abs(tju_batch_2_true_values,tju_batch_2_kan_symbolic_forecast).ravel()  # 展平为一维数组
tju_batch_2_pikan_model_error = relative_estimation_error_abs(tju_batch_2_true_values,tju_batch_2_pikan_model_forecast).ravel()  # 展平为一维数组
tju_batch_2_pikan_symbolic_error = relative_estimation_error_abs(tju_batch_2_true_values,tju_batch_2_pikan_symbolic_forecast).ravel()  # 展平为一维数组
# mit
mit_true_values = np.load(root+'mit_true_values.npy')
mit_kan_model_forecast = np.load(root+'mit_kan_model_forecast.npy')
mit_kan_symbolic_forecast = np.load(root+'mit_kan_symbolic_forecast.npy')
mit_pikan_model_forecast = np.load(root+'mit_pikan_model_forecast.npy')
mit_pikan_symbolic_forecast = np.load(root+'mit_pikan_symbolic_forecast.npy')
mit_kan_model_error = relative_estimation_error_abs(mit_true_values,mit_kan_model_forecast).ravel()  # 展平为一维数组
mit_kan_symbolic_error = relative_estimation_error_abs(mit_true_values,mit_kan_symbolic_forecast).ravel()  # 展平为一维数组
mit_pikan_model_error = relative_estimation_error_abs(mit_true_values,mit_pikan_model_forecast).ravel()  # 展平为一维数组
mit_pikan_symbolic_error = relative_estimation_error_abs(mit_true_values,mit_pikan_symbolic_forecast).ravel()  # 展平为一维数组
# hust
hust_true_values = np.load(root+'hust_true_values.npy')
hust_kan_model_forecast = np.load(root+'hust_kan_model_forecast.npy')
hust_kan_symbolic_forecast = np.load(root+'hust_kan_symbolic_forecast.npy')
hust_pikan_model_forecast = np.load(root+'hust_pikan_model_forecast.npy')
hust_pikan_symbolic_forecast = np.load(root+'hust_pikan_symbolic_forecast.npy')
hust_kan_model_error = relative_estimation_error_abs(hust_true_values,hust_kan_model_forecast).ravel()  # 展平为一维数组
hust_kan_symbolic_error = relative_estimation_error_abs(hust_true_values,hust_kan_symbolic_forecast).ravel()  # 展平为一维数组
hust_pikan_model_error = relative_estimation_error_abs(hust_true_values,hust_pikan_model_forecast).ravel()  # 展平为一维数组
hust_pikan_symbolic_error = relative_estimation_error_abs(hust_true_values,hust_pikan_symbolic_forecast).ravel()  # 展平为一维数组

dataset = ['XJTU batch 1', 'TJU batch 3', 'MIT', 'HUST']
methods = ['KAN Model', 'KAN Symbolic', 'PIKAN Model', 'PIKAN Symbolic']

# 每个阻尼参数下不同方法的估计误差数据
data = {
    'XJTU batch 1': [
        xjtu_batch_0_kan_model_error,  # KAN Model
        xjtu_batch_0_kan_symbolic_error,    # KAN Symbolic
        xjtu_batch_0_pikan_model_error,  # PIKAN Model
        xjtu_batch_0_pikan_symbolic_error # PIKAN Symbolic
    ],
    'TJU batch 3': [
        tju_batch_2_kan_model_error,  # KAN Model
        tju_batch_2_kan_symbolic_error,    # KAN Symbolic
        tju_batch_2_pikan_model_error,  # PIKAN Model
        tju_batch_2_pikan_symbolic_error # PIKAN Symbolic
    ],
    'MIT': [
        mit_kan_model_error,  # KAN Model
        mit_kan_symbolic_error,    # KAN Symbolic
        mit_pikan_model_error,  # PIKAN Model
        mit_pikan_symbolic_error # PIKAN Symbolic
    ],
    'HUST': [
        hust_kan_model_error,  # KAN Model
        hust_kan_symbolic_error,    # KAN Symbolic
        hust_pikan_model_error,  # PIKAN Model
        hust_pikan_symbolic_error # PIKAN Symbolic
    ]
}

# 设置参数
n_categories = len(dataset)  # 4个类别（D1-D4）
n_methods = len(methods)     # 4种方法
category_gap = 4             # 类别之间的间距（越大间距越宽）
box_width = 0.8              # 单个箱线图的宽度（越小越窄）

# 计算每个箱线图的位置
# 同一类别内的箱线图在 [i*category_gap, (i+1)*category_gap) 区间内紧凑排列
positions = []
for i in range(n_categories):  # 遍历每个类别（D1-D4）
    # 同一类别内的位置：从 i*category_gap 开始，等间隔排列n_methods个箱线图
    start = i * category_gap
    step = box_width * 1.2  # 同一类别内箱线图的间隔（避免重叠）
    for j in range(n_methods):  # 遍历每个方法
        positions.append(start + j * step)

# 绘制箱线图
fig, ax = plt.subplots(figsize=(12, 6))
box_plot = ax.boxplot(
    [data[d][i]*100 for d in dataset for i in range(n_methods)],  # 数据列表
    positions=positions,  # 自定义位置
    widths=box_width,     # 箱线图宽度
    patch_artist=True,    # 允许填充颜色
    vert=True             # 垂直箱线图
)

# 设置颜色（与你的代码一致）
colors = ["#13C8EC", "#1361F0", "#F35A26", "#F71432"]
for patch, color in zip(box_plot['boxes'], colors * n_categories):
    patch.set_facecolor(color)

# 调整x轴刻度（只显示类别名称，不显示具体位置）
ax.set_xticks([i * category_gap + (n_methods - 1) * step / 2 for i in range(n_categories)])
ax.set_xticklabels(dataset)  

# 设置 y 轴为对数刻度
ax.set_yscale('log')
ax.set_xlabel(None)
ax.set_ylabel('Relative Estimation Error (%)')
ax.legend(handles=[plt.Rectangle((0,0),1,1,fc=colors[i]) for i in range(n_methods)],
          labels=methods, loc='lower left',fontsize=16,ncol=2)
plt.tight_layout()
plt.savefig('../Figures/estimation_error_of_kan_pikan_model_symbolic.svg',format='svg')
plt.show()